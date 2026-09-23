"""
Шаг 4. Насколько можно доверять априору: split-half проверка на истории.

    python analytics/validate_prior.py [--out analytics/output] [--splits 30]

Историю случайно делим по абонентам пополам (A/B), оцениваем эффекты переходов по A и
сравниваем с тем, что видно в B:
  * raw      — простое среднее по ячейке (так устроена мок-модель среды);
  * shrunk   — среднее, сжатое к регрессии по цене (empirical Bayes, как в build_prior.py).
Метрики: взвешенная RMSE по pct, корреляция base = pct·conv, совпадение знака, калибровка
интервалов (доля |z| < 1.96 должна быть ≈ 95%).

Важно: это проверяет только выборочный шум внутри истории. Сдвиг между историей и реальной
аудиторией (главный риск по ТЗ) так не измерить — его закрывают пилоты.
Выход: --out/prior/validation.json
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from common import DEFAULT_DATA_DIR, DEFAULT_OUT_DIR, write_json

KEY = ["tariff_plan_code_from", "tariff_plan_code_to", "arpu_segment"]


def cell_stats(h: pd.DataFrame) -> pd.DataFrame:
    g = h.groupby(KEY)["pct"].agg(n="size", mean="mean").reset_index()
    tot = h.groupby(["tariff_plan_code_from", "arpu_segment"]).size().rename("N").reset_index()
    g = g.merge(tot, on=["tariff_plan_code_from", "arpu_segment"])
    g["conv"] = g["n"] / g["N"]
    return g


def shrink(h: pd.DataFrame, cells: pd.DataFrame, price: pd.Series, median_price: float) -> pd.DataFrame:
    out = []
    for seg, cs in cells.groupby("arpu_segment"):
        hs = h[h["arpu_segment"] == seg]
        g = hs.groupby(["tariff_plan_code_from", "tariff_plan_code_to"])["pct"]
        sigma2 = float((g.var(ddof=1).fillna(0) * (g.size() - 1)).sum() / max(len(hs) - g.ngroups, 1))
        cs = cs.copy()
        cs["dp"] = (cs["tariff_plan_code_to"].map(price) - cs["tariff_plan_code_from"].map(price)) / median_price
        fit = cs[cs["n"] >= 5]
        X = np.c_[np.ones(len(fit)), fit["dp"]]
        sw = np.sqrt(fit["n"].values)
        beta = np.linalg.lstsq(X * sw[:, None], fit["mean"].values * sw, rcond=None)[0]
        resid = fit["mean"].values - X @ beta
        tau2 = max(np.average(resid ** 2, weights=fit["n"]) - np.average(sigma2 / fit["n"], weights=fit["n"]), 0.0025)
        mu = beta[0] + beta[1] * cs["dp"]
        v = sigma2 / cs["n"]
        post_var = 1 / (1 / v + 1 / tau2)
        cs["shrunk"] = post_var * (cs["mean"] / v + mu / tau2)
        cs["post_sd"] = np.sqrt(post_var)
        cs["sigma2"] = sigma2
        out.append(cs)
    return pd.concat(out)


def main():
    parser = argparse.ArgumentParser(description="Split-half проверка априора")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="папка пакета участника")
    parser.add_argument("--splits", type=int, default=30)
    args = parser.parse_args()
    out_dir = Path(args.out)
    hist = pd.read_csv(out_dir / "clean" / "change_tariff_clean.csv")
    hist = hist[hist["usable_for_prior"]].copy()
    price = pd.read_csv(Path(args.data_dir) / "data" / "dict_tariff.csv").set_index("tariff_plan_code")["price_tariff"]
    median_price = float(price.median())

    rng = np.random.default_rng(2026)
    ids = hist["ID_NUMBER"].unique()
    rows = []
    for k in range(args.splits):
        a_ids = set(rng.choice(ids, size=len(ids) // 2, replace=False))
        A, B = hist[hist["ID_NUMBER"].isin(a_ids)], hist[~hist["ID_NUMBER"].isin(a_ids)]
        ca = shrink(A, cell_stats(A), price, median_price)
        cb = cell_stats(B)
        m = ca.merge(cb, on=KEY, suffixes=("_a", "_b"))
        m = m[m["n_b"] >= 5]
        w = m["n_b"]
        se_b = np.sqrt(m["sigma2"] / m["n_b"])
        z = (m["mean_b"] - m["shrunk"]) / np.sqrt(m["post_sd"] ** 2 + se_b ** 2)
        base_a_raw, base_a_sh, base_b = m["mean_a"] * m["conv_a"], m["shrunk"] * m["conv_a"], m["mean_b"] * m["conv_b"]
        big = m["n_b"] >= 20
        rows.append({
            "rmse_pct_raw": math.sqrt(np.average((m["mean_a"] - m["mean_b"]) ** 2, weights=w)),
            "rmse_pct_shrunk": math.sqrt(np.average((m["shrunk"] - m["mean_b"]) ** 2, weights=w)),
            "corr_base_raw": float(np.corrcoef(base_a_raw, base_b)[0, 1]),
            "corr_base_shrunk": float(np.corrcoef(base_a_sh, base_b)[0, 1]),
            "sign_agree_pct_n20": float((np.sign(m.loc[big, "shrunk"]) == np.sign(m.loc[big, "mean_b"])).mean()),
            "coverage_95": float((z.abs() < 1.96).mean()),
            "cells_compared": int(len(m)),
        })
    res = pd.DataFrame(rows)
    summary = {k: {"mean": float(res[k].mean()), "p10": float(res[k].quantile(0.1)), "p90": float(res[k].quantile(0.9))}
               for k in res.columns}
    summary["splits"] = args.splits
    summary["note"] = ("Проверяет только выборочный шум внутри истории; сдвиг «история → реальная аудитория» "
                       "не измеряется и закрывается пилотами.")
    write_json(summary, out_dir / "prior" / "validation.json")
    print(f"split-half, {args.splits} разбиений, ячеек в сравнении ~{summary['cells_compared']['mean']:.0f}")
    print(f"RMSE pct: сырое среднее {summary['rmse_pct_raw']['mean']:.3f} → со сжатием {summary['rmse_pct_shrunk']['mean']:.3f}")
    print(f"корреляция base (A vs B): сырое {summary['corr_base_raw']['mean']:.2f}, "
          f"со сжатием {summary['corr_base_shrunk']['mean']:.2f}")
    print(f"совпадение знака pct (ячейки n>=20): {summary['sign_agree_pct_n20']['mean']:.0%}; "
          f"покрытие 95%-интервалов: {summary['coverage_95']['mean']:.0%}")


if __name__ == "__main__":
    main()
