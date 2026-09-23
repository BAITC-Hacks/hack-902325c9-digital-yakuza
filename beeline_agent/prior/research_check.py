"""
Проверка утверждений исследования команды на данных (воспроизводит таблицу «0.» из prior/README.md).

    python prior/research_check.py      # ~15 с, результат: prior/reports/research_check.json

Проверяется: структура истории, окно «до смены», хвосты отношения ARPU, плацебо (регрессия к среднему),
надёжность контрастов внутри страты, покрытие троек, сдвиг распределений (adversarial AUC),
механизм «перерасход → больший пакет», модель шума пилота из ТЗ.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from history import ROOT, load_history  # noqa: E402

PRE = ["2026-07-01", "2026-08-01", "2026-09-01"]
EARLY = ["2026-04-01", "2026-05-01", "2026-06-01"]
BINS, LABELS = [-np.inf, 1000, 5000, np.inf], ["LOW", "MID", "HIGH"]


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def norm_ppf(p):
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if norm_cdf(mid) < p else (lo, mid)
    return (lo + hi) / 2


def contrast_reliability(d: pd.DataFrame, y: str, splits=30, min_n=5) -> dict:
    """split-half корреляция контрастов (y − среднее страты) на трёх уровнях + поправка Спирмена–Брауна."""
    levels = {"target": ["tariff_to"], "seg_target": ["arpu_segment", "tariff_to"],
              "cell": ["tariff_from", "arpu_segment", "tariff_to"]}
    rng = np.random.default_rng(0)
    ids = d["ID_NUMBER"].unique()
    out = {k: [] for k in levels}
    for _ in range(splits):
        a = set(rng.choice(ids, len(ids) // 2, replace=False))
        parts = []
        for part in (d[d["ID_NUMBER"].isin(a)], d[~d["ID_NUMBER"].isin(a)]):
            part = part.assign(c=part[y] - part.groupby(["tariff_from", "arpu_segment"])[y].transform("mean"))
            parts.append(part)
        for k, keys in levels.items():
            ga, gb = (p.groupby(keys)["c"].agg(["mean", "size"]) for p in parts)
            j = ga.join(gb, lsuffix="_a", rsuffix="_b", how="inner")
            j = j[(j["size_a"] >= min_n) & (j["size_b"] >= min_n)]
            out[k].append(np.corrcoef(j["mean_a"], j["mean_b"])[0, 1])
    return {k: {"r": round(float(np.mean(v)), 3), "spearman_brown": round(2 * np.mean(v) / (1 + np.mean(v)), 3)}
            for k, v in out.items()}


def adversarial_auc(hist_ids: pd.Series, profile: pd.DataFrame, ct: pd.DataFrame, tr: pd.DataFrame) -> dict:
    feats = ["DATA_VOLUME", "LTE_DATA_VOLUME", "OUT_LOC_ONNET_MIN", "OUT_LOC_OFFNET_MIN", "OUT_LOC_OFFNET_PAID_MIN",
             "OUT_INTER_MIN", "OUT_LOCAL_ONNET_SMS_AMT", "COUNT_CONTACT", "SUM_TRANSACT_CONTACT",
             "AVG_DURATION_CONTACT", "COUNT_BASE_STATION"]
    h = tr[tr["time_key"].isin(PRE)].groupby("ID_NUMBER")[feats].mean()
    h = (ct.drop_duplicates("ID_NUMBER").set_index("ID_NUMBER")[["AVG_ARPU_PREV_3M"]]
         .rename(columns={"AVG_ARPU_PREV_3M": "ARPU"}).join(h, how="inner"))
    a = profile.set_index("ID_NUMBER")[["ARPU_3m_avg"] + feats].rename(columns={"ARPU_3m_avg": "ARPU"})
    X = pd.concat([h.assign(y=0), a.assign(y=1)])
    y = X["y"].values
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    tr_i, te_i = idx[: len(X) // 2], idx[len(X) // 2:]

    def design(cols):
        Z = np.log1p(X[cols].clip(lower=0).fillna(0)).values
        return np.c_[np.ones(len(Z)), (Z - Z.mean(0)) / (Z.std(0) + 1e-9)]

    def fit(M, yy):
        w = np.zeros(M.shape[1])
        for _ in range(25):
            p = 1 / (1 + np.exp(-M @ w))
            H = M.T @ (M * (p * (1 - p))[:, None]) + 1e-3 * np.eye(len(w))
            w += np.linalg.solve(H, M.T @ (yy - p) - 1e-3 * w)
        return w

    def auc(s, yy):
        r = pd.Series(s).rank().values
        n1 = yy.sum()
        return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(yy) - n1)))

    res = {}
    for name, cols in [("arpu_only", ["ARPU"]), ("all_features", ["ARPU"] + feats)]:
        M = design(cols)
        w = fit(M[tr_i], y[tr_i])
        res[name] = round(auc(M[te_i] @ w, y[te_i]), 3)
    M = design(["ARPU"] + feats)
    p = 1 / (1 + np.exp(-M @ fit(M, y)))
    wts = (p / (1 - p))[y == 0]
    res["importance_weight_ess_share"] = round(float(wts.sum() ** 2 / (wts ** 2).sum() / len(wts)), 3)
    return res


def main():
    ct = pd.read_csv(ROOT / "data" / "change_tariff.csv")
    am = pd.read_csv(ROOT / "data" / "arpu_monthly.csv")
    tr = pd.read_csv(ROOT / "data" / "traffic.csv")
    prof = pd.read_csv(ROOT / "customer_profile.csv")
    tariffs = pd.read_csv(ROOT / "data" / "dict_tariff.csv")
    r = {}

    ids = {"change_tariff": set(ct["ID_NUMBER"]), "arpu_monthly": set(am["ID_NUMBER"]), "traffic": set(tr["ID_NUMBER"])}
    r["ids"] = {k: len(v) for k, v in ids.items()}
    r["ids"]["in_all_three"] = len(set.intersection(*ids.values()))
    r["ids"]["overlap_with_audience"] = len(set.union(*ids.values()) & set(prof["ID_NUMBER"]))
    r["change_months"] = sorted(ct["TIME_KEY"].str[:7].unique().tolist())

    pre = am[am["TIME_KEY"].isin(PRE)].groupby("ID_NUMBER")["ARPU_1M"].mean()
    m = ct.drop_duplicates().merge(pre.rename("m"), left_on="ID_NUMBER", right_index=True, how="left")
    r["prev_equals_mean_jul_sep_share"] = round(float(((m["m"] - m["AVG_ARPU_PREV_3M"]).abs() < 0.01).mean()), 4)

    ratio = ct["AVG_ARPU_NEXT_3M"] / ct["AVG_ARPU_PREV_3M"].where(ct["AVG_ARPU_PREV_3M"] > 0)
    r["ratio_tails"] = {"le_0": round(float((ratio <= 0).mean()), 3), "ge_10x": round(float((ratio >= 10).mean()), 3),
                        "prev_le_0": int((ct["AVG_ARPU_PREV_3M"] <= 0).sum()), "next_eq_0": int((ct["AVG_ARPU_NEXT_3M"] == 0).sum())}

    # плацебо: апрель–июнь → июль–сентябрь, смен тарифа нет
    piv = am.drop_duplicates().groupby(["ID_NUMBER", "TIME_KEY"])["ARPU_1M"].mean().unstack()
    p1, p2 = piv[EARLY].mean(axis=1), piv[PRE].mean(axis=1)
    ok = p1.notna() & p2.notna()
    seg = pd.cut(p1[ok], BINS, labels=LABELS)
    lr = np.log(p2[ok].clip(lower=0) + 1) - np.log(p1[ok].clip(lower=0) + 1)
    pc = ((p2[ok] - p1[ok]) / p1[ok].where(p1[ok] >= 100)).clip(-1, 3)
    hist, _ = load_history()
    r["placebo"] = {"log1p_ratio": lr.groupby(seg, observed=True).mean().round(3).to_dict(),
                    "clipped_pct": pc.groupby(seg, observed=True).mean().round(3).to_dict(),
                    "real_switches_clipped_pct": hist.groupby("arpu_segment")["pct"].mean().round(3).to_dict(),
                    "high_downsell_share_placebo": round(float((pc[seg == "HIGH"] < -0.1).mean()), 3)}

    r["contrast_reliability_pct"] = contrast_reliability(hist, "pct")
    d = ct.drop_duplicates()
    d = d[d["AVG_ARPU_PREV_3M"] > 0].rename(columns={"tariff_plan_code_from": "tariff_from", "tariff_plan_code_to": "tariff_to"})
    d["arpu_segment"] = pd.cut(d["AVG_ARPU_PREV_3M"], BINS, labels=LABELS).astype(str)
    y = np.log(d["AVG_ARPU_NEXT_3M"].clip(lower=0) + 1) - np.log(d["AVG_ARPU_PREV_3M"] + 1)
    d["ylog"] = y.clip(*y.quantile([0.01, 0.99]))
    r["contrast_reliability_log"] = contrast_reliability(d, "ylog")

    aud = prof.dropna(subset=["current_tariff", "arpu_segment"]).groupby(["current_tariff", "arpu_segment"]).size()
    n = hist.groupby(["tariff_from", "arpu_segment", "tariff_to"]).size()
    codes = tariffs["tariff_plan_code"].tolist()
    cnt = np.array([n.get((f, s, t), 0) for (f, s) in aud.index for t in codes if t != f])
    r["coverage"] = {"targets_in_history": int(hist["tariff_to"].nunique()), "relevant_triples": int(len(cnt)),
                     "zero_obs_share": round(float((cnt == 0).mean()), 3), "ge30_share": round(float((cnt >= 30).mean()), 3)}
    r["shift"] = {"high_share": {"audience": round(float((prof["arpu_segment"] == "HIGH").mean()), 3),
                                 "history": round(float((hist["arpu_segment"] == "HIGH").mean()), 3)},
                  "adversarial": adversarial_auc(hist["ID_NUMBER"], prof, ct, tr)}

    # перерасход трафика → больший пакет данных
    pre_tr = tr[tr["time_key"].isin(PRE)].groupby("ID_NUMBER")[["DATA_VOLUME", "OUT_LOC_OFFNET_PAID_MIN"]].mean()
    t = tariffs.set_index("tariff_plan_code")
    mins = t["Min_another_operator_in_PKG"] + t["Min_another_operator_and_city_in_PKG"]
    h2 = hist.merge(pre_tr, left_on="ID_NUMBER", right_index=True, how="inner")
    over = h2["DATA_VOLUME"] > h2["tariff_from"].map(t["Data_in_PKG"])
    bigger = h2["tariff_to"].map(t["Data_in_PKG"]) > h2["tariff_from"].map(t["Data_in_PKG"])
    paid = h2["OUT_LOC_OFFNET_PAID_MIN"] > 1
    more_min = h2["tariff_to"].map(mins) > h2["tariff_from"].map(mins)
    r["overage"] = {"data_over_to_bigger_data": round(float(bigger[over].mean()), 3),
                    "no_over_to_bigger_data": round(float(bigger[~over].mean()), 3),
                    "paid_min_to_more_min": round(float(more_min[paid].mean()), 3),
                    "no_paid_to_more_min": round(float(more_min[~paid].mean()), 3)}

    # шум пилота из ТЗ: 1/4 ошибок знака на 30, 1/25 на 200
    snr30 = -norm_ppf(0.25) / math.sqrt(30)
    snr200 = -norm_ppf(0.04) / math.sqrt(200)
    snr = (snr30 + snr200) / 2
    r["pilot_noise"] = {"snr_per_customer_from_30": round(snr30, 4), "snr_per_customer_from_200": round(snr200, 4),
                        "wrong_sign_at_100": round(norm_cdf(-snr * 10), 3),
                        "n_for_90pct": round((norm_ppf(0.90) / snr) ** 2), "n_for_95pct": round((norm_ppf(0.95) / snr) ** 2),
                        "implied_effect_at_std_0.804": round(snr * 0.804, 3)}

    out = ROOT / "prior" / "reports"
    out.mkdir(exist_ok=True)
    (out / "research_check.json").write_text(json.dumps(r, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps(r, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
