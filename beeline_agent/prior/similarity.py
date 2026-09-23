"""
Насколько история смен тарифов похожа на аудиторию кампаний и сколько ей доверять.

    python prior/similarity.py [--splits 30]

1. Сравнение распределений (история до смены vs customer_profile): ARPU-сегменты, тарифы,
   data/call-сегменты, ARPU внутри сегментов, потребление. Мера сходства — overlap
   (1 − полусумма модулей разностей долей): 1 = распределения совпадают, 0 = не пересекаются.
2. «Переносимость» эффектов. Историю делим на «похожих на аудиторию» и «непохожих»
   (вес = доля такой комбинации сегментов в аудитории / в истории), считаем q по каждой части
   и смотрим, как q «похожих» зависит от q «непохожих»: наклон β — какую долю исторического
   эффекта разумно переносить (это и есть PRIOR_SHRINK в agent.py), разброс остатков — сколько
   неопределённости добавлять (PRIOR_MODEL_SD). Для сравнения — тот же расчёт на случайном
   разбиении (там расхождение даёт только шум выборки).

Выход: prior/reports/similarity.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from history import KEYS, ROOT, eb_shrink, load_history, pooled_sigma2  # noqa: E402

PRE_MONTHS = ["2026-07-01", "2026-08-01", "2026-09-01"]    # 3 месяца до смены (смена — 2026-10)
CONV_ALPHA = 10.0


def overlap(p: pd.Series, q: pd.Series) -> float:
    idx = p.index.union(q.index)
    return float(1 - 0.5 * (p.reindex(idx, fill_value=0) - q.reindex(idx, fill_value=0)).abs().sum())


def history_features(hist: pd.DataFrame) -> pd.DataFrame:
    tr = pd.read_csv(ROOT / "data" / "traffic.csv", usecols=["ID_NUMBER", "time_key", "DATA_VOLUME",
                                                             "OUT_LOC_ONNET_MIN", "OUT_LOC_OFFNET_MIN"])
    tr = tr[tr["time_key"].isin(PRE_MONTHS)].groupby("ID_NUMBER").mean(numeric_only=True)
    h = hist.merge(tr, left_on="ID_NUMBER", right_index=True, how="left")
    dv = h["DATA_VOLUME"]
    h["data_segment"] = np.where(dv.isna(), "NA", np.where(dv <= 0, "NON_USER", np.where(dv <= 2000, "LITE", "HEAVY")))
    mins = (h["OUT_LOC_ONNET_MIN"] + h["OUT_LOC_OFFNET_MIN"]).fillna(0)
    h["call_segment"] = np.where(mins < 100, "LOW", np.where(mins <= 400, "MEDIUM", "HIGH"))
    return h


def q_table(d: pd.DataFrame, price: pd.Series, median_price: float) -> pd.DataFrame:
    cells = d.groupby(KEYS)["pct"].agg(n="size", est="mean").reset_index()
    cells, _ = eb_shrink(cells, pooled_sigma2(d), price, median_price)
    tot = cells.groupby(["tariff_from", "arpu_segment"])["n"].transform("sum")
    share = d.groupby(["arpu_segment", "tariff_to"]).size() / d.groupby("arpu_segment").size()
    p0 = np.array([share.get((s, t), 0.0) for s, t in zip(cells["arpu_segment"], cells["tariff_to"])])
    cells["q"] = cells["shrunk"] * (cells["n"] + CONV_ALPHA * p0) / (tot + CONV_ALPHA)
    return cells[KEYS + ["n", "q"]]


def transfer(a: pd.DataFrame, b: pd.DataFrame, price, median_price, min_n=10) -> dict:
    """Регрессия q(b) на q(a) по связкам, где в обеих частях >= min_n наблюдений."""
    m = q_table(a, price, median_price).merge(q_table(b, price, median_price), on=KEYS, suffixes=("_a", "_b"))
    m = m[(m["n_a"] >= min_n) & (m["n_b"] >= min_n)]
    w = np.minimum(m["n_a"], m["n_b"]).values.astype(float)
    x, y = m["q_a"].values, m["q_b"].values
    xm, ym = np.average(x, weights=w), np.average(y, weights=w)
    beta = np.sum(w * (x - xm) * (y - ym)) / np.sum(w * (x - xm) ** 2)
    alpha = ym - beta * xm
    resid = y - (alpha + beta * x)
    return {"beta": float(beta), "alpha": float(alpha), "resid_sd": float(np.sqrt(np.average(resid ** 2, weights=w))),
            "corr": float(np.corrcoef(x, y)[0, 1]), "sign_agree": float((np.sign(x) == np.sign(y)).mean()),
            "cells": int(len(m))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=int, default=30)
    args = parser.parse_args()

    hist, _ = load_history()
    hist = history_features(hist)
    prof = pd.read_csv(ROOT / "customer_profile.csv")
    price = pd.read_csv(ROOT / "data" / "dict_tariff.csv").set_index("tariff_plan_code")["price_tariff"]
    median_price = float(price.median())

    # --- 1. распределения ------------------------------------------------------
    share = lambda s: s.fillna("NA").value_counts(normalize=True)  # noqa: E731
    comp = {}
    for name, a_col, h_col in [("arpu_segment", "arpu_segment", "arpu_segment"),
                               ("data_segment", "data_segment", "data_segment"),
                               ("call_segment", "call_segment", "call_segment"),
                               ("tariff", "current_tariff", "tariff_from")]:
        pa, ph = share(prof[a_col]), share(hist[h_col])
        comp[name] = {"overlap": overlap(pa, ph),
                      "audience": pa.round(4).to_dict(), "history": ph.round(4).to_dict()}
    q = [0.1, 0.25, 0.5, 0.75, 0.9]
    arpu_within = {}
    for seg in ["LOW", "MID", "HIGH"]:
        a = prof.loc[prof["arpu_segment"] == seg, "ARPU_3m_avg"]
        h = hist.loc[hist["arpu_segment"] == seg, "AVG_ARPU_PREV_3M"]
        arpu_within[seg] = {"audience": {str(k): round(v, 1) for k, v in a.quantile(q).items()},
                            "history": {str(k): round(v, 1) for k, v in h.quantile(q).items()}}
    comp["arpu_within_segment"] = arpu_within
    comp["mean_arpu"] = {"audience_ARPU_3m_avg": float(prof["ARPU_3m_avg"].mean()),
                         "history_AVG_ARPU_PREV_3M_all_rows": float(pd.read_csv(ROOT / "data" / "change_tariff.csv")
                                                                    ["AVG_ARPU_PREV_3M"].mean()),
                         "history_used_rows": float(hist["AVG_ARPU_PREV_3M"].mean())}
    comp["median_usage"] = {c: {"audience": float(prof[c].median()), "history": float(hist[c].median())}
                            for c in ["DATA_VOLUME", "OUT_LOC_ONNET_MIN", "OUT_LOC_OFFNET_MIN"]}
    cells_aud = prof.groupby(["current_tariff", "arpu_segment"]).size()
    cells_hist = hist.groupby(["tariff_from", "arpu_segment"]).size()
    covered = cells_aud[cells_aud.index.isin(cells_hist[cells_hist >= 30].index)].sum() / cells_aud.sum()
    comp["audience_share_in_cells_with_30plus_history_switches"] = float(covered)

    # --- 2. переносимость эффектов --------------------------------------------
    key = ["arpu_segment", "data_segment", "call_segment"]
    pa = prof.assign(data_segment=prof["data_segment"].fillna("NA")).groupby(key).size() / len(prof)
    ph = hist.groupby(key).size() / len(hist)
    ratio = (pa / ph).fillna(0)
    hist["w_aud"] = hist.set_index(key).index.map(ratio).fillna(0).values
    rng = np.random.default_rng(11)
    shift_runs, random_runs = [], []
    ids = hist["ID_NUMBER"].unique()
    for _ in range(args.splits):
        # похожие на аудиторию vs непохожие (ранги с случайным разбиением одинаковых весов)
        jitter = rng.random(len(hist)) * 1e-6
        like = (hist["w_aud"] + jitter) >= np.median(hist["w_aud"] + jitter)
        shift_runs.append(transfer(hist[~like], hist[like], price, median_price))
        a_ids = rng.choice(ids, size=len(ids) // 2, replace=False)
        in_a = hist["ID_NUMBER"].isin(a_ids)
        random_runs.append(transfer(hist[in_a], hist[~in_a], price, median_price))
    summ = lambda runs: {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}  # noqa: E731
    tr_shift, tr_rand = summ(shift_runs), summ(random_runs)
    upper = float(np.clip(tr_shift["beta"], 0.0, 1.0))
    result = {"distributions": comp,
              "transfer_history_to_audience_like": tr_shift,
              "transfer_random_split": tr_rand,
              "recommendation": {
                  "PRIOR_SHRINK_upper_bound_from_data": round(upper, 2),
                  "PRIOR_SHRINK_suggested": "0.7-0.8",
                  "PRIOR_MODEL_SD_in_q_units": round(tr_shift["resid_sd"], 3),
                  "comment": ("β — доля исторического q, которая переносится на похожих на аудиторию абонентов "
                              "внутри истории. Реальная аудитория отличается сильнее (организаторы прямо пишут, "
                              "что эффекты другие), поэтому β — верхняя граница доверия, а 0.7-0.8 — запас.")}}
    out = ROOT / "prior" / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "similarity.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    print("Сходство распределений (overlap, 1 = совпадают):")
    for k in ["arpu_segment", "data_segment", "call_segment", "tariff"]:
        print(f"  {k:13s} {comp[k]['overlap']:.2f}")
    print(f"  средний ARPU: аудитория {comp['mean_arpu']['audience_ARPU_3m_avg']:,.0f} vs история "
          f"{comp['mean_arpu']['history_AVG_ARPU_PREV_3M_all_rows']:,.0f}")
    print(f"  аудитория в ячейках с 30+ сменами в истории: {covered:.1%}")
    print(f"Переносимость q: история→похожие на аудиторию β={tr_shift['beta']:.2f} (corr {tr_shift['corr']:.2f}, "
          f"знак {tr_shift['sign_agree']:.0%}, остаток sd {tr_shift['resid_sd']:.3f}); "
          f"случайное разбиение β={tr_rand['beta']:.2f} (corr {tr_rand['corr']:.2f})")
    print(f"Доверие к истории по данным — не выше β≈{upper:.2f}; реальная аудитория отличается сильнее → "
          f"PRIOR_SHRINK 0.7–0.8, PRIOR_MODEL_SD ≈ {tr_shift['resid_sd']:.3f} в единицах q")


if __name__ == "__main__":
    main()
