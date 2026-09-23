"""
Какая оценка эффекта связки надёжнее: среднее, медиана, усечённое среднее или сжатие (EB)?

    python prior/evaluate_estimators.py [--splits 50]

Split-half: историю много раз делим пополам по абонентам. Сравниваются: среднее, медиана,
усечённые средние, винзоризация, сжатие к ценовой регрессии (eb_mean) и иерархия
«уровень страты + контраст цели» (hier, prior/hier.py). По половине A считаем оценку
каждой связки (тариф_откуда, сегмент, тариф_куда), по половине B — то, что считает среда
(среднее клипнутого Δ%), и смотрим ошибку. Отдельно по связкам с 1–5, 6–20 и 21+ наблюдениями.
Для q = Δ% × доля перехода сравниваем сырую долю и сглаженную.

Выход: prior/reports/estimator_comparison.csv (+ печать сводки).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hier  # noqa: E402
from history import KEYS, ROOT, eb_shrink, load_history, pooled_sigma2, trimmed_mean  # noqa: E402

BUCKETS = [(1, 5, "1-5"), (6, 20, "6-20"), (21, 10 ** 9, "21+")]
TARIFFS = None
CONV_ALPHA = 10.0


def cell_estimates(a: pd.DataFrame, price, median_price) -> pd.DataFrame:
    g = a.groupby(KEYS)["pct"]
    cells = g.agg(n="size", mean="mean", median="median").reset_index()
    cells["trim10"] = g.apply(lambda x: trimmed_mean(x.values, 0.10)).values
    cells["trim20"] = g.apply(lambda x: trimmed_mean(x.values, 0.20)).values
    lo, hi = a.groupby("arpu_segment")["pct"].quantile(0.05), a.groupby("arpu_segment")["pct"].quantile(0.95)
    wins = a["pct"].clip(a["arpu_segment"].map(lo), a["arpu_segment"].map(hi))
    cells["winsor"] = a.assign(w=wins).groupby(KEYS)["w"].mean().values
    s2 = pooled_sigma2(a)
    eb, _ = eb_shrink(cells.assign(est=cells["mean"]), s2, price, median_price)
    cells = cells.merge(eb[KEYS + ["shrunk"]].rename(columns={"shrunk": "eb_mean"}), on=KEYS)
    eb2, _ = eb_shrink(cells.assign(est=cells["trim10"]), s2, price, median_price)
    cells = cells.merge(eb2[KEYS + ["shrunk"]].rename(columns={"shrunk": "eb_trim10"}), on=KEYS)
    # иерархия: уровень страты + контраст цели (prior/hier.py)
    model = hier.fit(a, TARIFFS)
    cells["hier"] = [hier.predict(model, f, s, t)["pct"]
                     for f, s, t in zip(cells["tariff_from"], cells["arpu_segment"], cells["tariff_to"])]
    # доля переходов: сырая и сглаженная к структуре переходов сегмента
    tot = a.groupby(["tariff_from", "arpu_segment"]).size().rename("N").reset_index()
    cells = cells.merge(tot, on=["tariff_from", "arpu_segment"])
    seg_share = a.groupby(["arpu_segment", "tariff_to"]).size() / a.groupby("arpu_segment").size()
    p0 = [seg_share.get((s, t), 0.0) for s, t in zip(cells["arpu_segment"], cells["tariff_to"])]
    cells["conv_raw"] = cells["n"] / cells["N"]
    cells["conv_smooth"] = (cells["n"] + CONV_ALPHA * np.array(p0)) / (cells["N"] + CONV_ALPHA)
    return cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=int, default=50)
    args = parser.parse_args()

    global TARIFFS
    hist, _ = load_history()
    TARIFFS = pd.read_csv(ROOT / "data" / "dict_tariff.csv")
    price = TARIFFS.set_index("tariff_plan_code")["price_tariff"]
    median_price = float(price.median())
    estimators = ["mean", "median", "trim10", "trim20", "winsor", "eb_mean", "eb_trim10", "hier"]

    rng = np.random.default_rng(7)
    ids = hist["ID_NUMBER"].unique()
    rows = []
    for split in range(args.splits):
        a_ids = rng.choice(ids, size=len(ids) // 2, replace=False)
        in_a = hist["ID_NUMBER"].isin(a_ids)
        A, B = hist[in_a], hist[~in_a]
        est = cell_estimates(A, price, median_price)
        tgt = B.groupby(KEYS)["pct"].agg(n_b="size", mean_b="mean").reset_index()
        tot_b = B.groupby(["tariff_from", "arpu_segment"]).size().rename("N_b").reset_index()
        tgt = tgt.merge(tot_b, on=["tariff_from", "arpu_segment"])
        tgt["q_b"] = tgt["mean_b"] * tgt["n_b"] / tgt["N_b"]
        m = est.merge(tgt, on=KEYS)
        m = m[m["n_b"] >= 5]                      # цель должна быть не слишком шумной
        for lo_n, hi_n, label in BUCKETS + [(1, 10 ** 9, "all")]:
            mm = m[(m["n"] >= lo_n) & (m["n"] <= hi_n)]
            if len(mm) < 3:
                continue
            w = mm["n_b"]
            for e in estimators:
                err = mm[e] - mm["mean_b"]
                for conv in ["conv_raw", "conv_smooth"]:
                    q = mm[e] * mm[conv]
                    rows.append({"split": split, "bucket": label, "estimator": e, "conv": conv,
                                 "rmse_pct": float(np.sqrt(np.average(err ** 2, weights=w))),
                                 "bias_pct": float(np.average(err, weights=w)),
                                 "rmse_q": float(np.sqrt(np.average((q - mm["q_b"]) ** 2, weights=w))),
                                 "corr_q": float(np.corrcoef(q, mm["q_b"])[0, 1]) if len(mm) > 3 else np.nan,
                                 "sign_agree": float((np.sign(mm[e]) == np.sign(mm["mean_b"])).mean()),
                                 "cells": int(len(mm))})
    res = (pd.DataFrame(rows).groupby(["bucket", "estimator", "conv"])
           .agg(rmse_pct=("rmse_pct", "mean"), bias_pct=("bias_pct", "mean"), rmse_q=("rmse_q", "mean"),
                corr_q=("corr_q", "mean"), sign_agree=("sign_agree", "mean"), cells=("cells", "mean"))
           .reset_index())
    out = ROOT / "prior" / "reports"
    out.mkdir(parents=True, exist_ok=True)
    res.round(4).to_csv(out / "estimator_comparison.csv", index=False)

    pd.set_option("display.width", 200)
    view = res[res["conv"] == "conv_smooth"].pivot(index="estimator", columns="bucket", values="rmse_pct")
    print("RMSE оценки Δ% против половины B (меньше — лучше):")
    print(view[["1-5", "6-20", "21+", "all"]].round(3).sort_values("all").to_string())
    qv = res[res["bucket"] == "all"].pivot(index="estimator", columns="conv", values="rmse_q")
    print("\nRMSE q = Δ% × доля перехода (все связки):")
    print(qv.round(4).sort_values("conv_smooth").to_string())


if __name__ == "__main__":
    main()
