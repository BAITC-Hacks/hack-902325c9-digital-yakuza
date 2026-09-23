"""
Загрузка и чистка истории смен тарифов (data/change_tariff.csv) — общие правила для
prior/build_prior.py и prior/evaluate_estimators.py. Правила и их обоснование — prior/README.md.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARPU_BINS = [-np.inf, 1000, 5000, np.inf]      # как в среде (mock_environment / профиль)
ARPU_LABELS = ["LOW", "MID", "HIGH"]
MIN_PREV_ARPU = 100          # ниже относительное изменение не определено (0) или взрывается
PCT_CLIP = (-1.0, 3.0)       # как в среде: эффект считается по клипнутому Δ%
KEYS = ["tariff_from", "arpu_segment", "tariff_to"]


def load_history(root: Path = ROOT):
    """Возвращает (очищенные строки, отчёт о том, что и почему убрано)."""
    return clean_history(pd.read_csv(root / "data" / "change_tariff.csv"))


def clean_history(raw: pd.DataFrame):
    """Правила чистки (обоснование — prior/README.md). Принимает строки в формате change_tariff.csv."""
    report = {"rows_in_file": int(len(raw))}

    df = raw.drop_duplicates()
    report["removed_full_duplicates"] = int(len(raw) - len(df))          # 6 строк-повторов (тот же ID и те же цифры)

    prev, nxt = df["AVG_ARPU_PREV_3M"], df["AVG_ARPU_NEXT_3M"]
    report["prev_negative"] = int((prev < 0).sum())
    report["prev_zero"] = int((prev == 0).sum())
    report["prev_0_100"] = int(((prev > 0) & (prev < MIN_PREV_ARPU)).sum())
    report["next_zero_kept"] = int(((nxt == 0) & (prev >= MIN_PREV_ARPU)).sum())
    keep = prev >= MIN_PREV_ARPU
    report["removed_prev_below_100"] = int((~keep).sum())
    df = df[keep].copy()

    df["arpu_segment"] = pd.cut(df["AVG_ARPU_PREV_3M"], bins=ARPU_BINS, labels=ARPU_LABELS).astype(str)
    df["pct_raw"] = (df["AVG_ARPU_NEXT_3M"] - df["AVG_ARPU_PREV_3M"]) / df["AVG_ARPU_PREV_3M"]
    df["pct"] = df["pct_raw"].clip(*PCT_CLIP)
    report["pct_clipped_at_+300%"] = int((df["pct_raw"] > PCT_CLIP[1]).sum())
    report["pct_equal_-100%_kept"] = int((df["pct"] <= -0.999).sum())
    report["prev_above_p999_kept"] = int((df["AVG_ARPU_PREV_3M"] > raw["AVG_ARPU_PREV_3M"].quantile(0.999)).sum())
    report["rows_used"] = int(len(df))
    df = df.rename(columns={"tariff_plan_code_from": "tariff_from", "tariff_plan_code_to": "tariff_to"})
    return df, report


def trimmed_mean(x: np.ndarray, share: float) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    k = int(np.floor(share * len(x)))
    return float(x[k:len(x) - k].mean()) if len(x) - 2 * k > 0 else float(np.median(x))


def pooled_sigma2(df: pd.DataFrame, value: str = "pct") -> dict:
    """Внутригрупповая дисперсия Δ% по сегменту (пул по всем связкам сегмента)."""
    out = {}
    for seg, d in df.groupby("arpu_segment"):
        g = d.groupby(["tariff_from", "tariff_to"])[value]
        dof = len(d) - g.ngroups
        out[seg] = float((g.var(ddof=1).fillna(0) * (g.size() - 1)).sum() / max(dof, 1))
    return out


def eb_shrink(cells: pd.DataFrame, sigma2: dict, price: pd.Series, median_price: float,
              est_col: str = "est", n_col: str = "n", min_n_fit: int = 5, tau2_floor: float = 0.0025):
    """
    Empirical Bayes: оценка связки подтягивается к регрессии Δ% ~ a + b·Δцены внутри сегмента.
    Чем меньше наблюдений, тем сильнее сжатие. Возвращает cells с колонками
    shrunk, post_sd (неопределённость оценки), prior_mu, и параметры по сегментам.
    """
    parts, params = [], {}
    for seg, cs in cells.groupby("arpu_segment"):
        cs = cs.copy()
        s2 = sigma2[seg]
        cs["dprice"] = (cs["tariff_to"].map(price) - cs["tariff_from"].map(price)) / median_price
        fit = cs[cs[n_col] >= min_n_fit]
        X = np.c_[np.ones(len(fit)), fit["dprice"]]
        w = np.sqrt(fit[n_col].values)
        a, b = np.linalg.lstsq(X * w[:, None], fit[est_col].values * w, rcond=None)[0]
        resid = fit[est_col].values - (a + b * fit["dprice"].values)
        tau2 = max(np.average(resid ** 2, weights=fit[n_col]) - np.average(s2 / fit[n_col], weights=fit[n_col]),
                   tau2_floor)
        cs["prior_mu"] = a + b * cs["dprice"]
        v = s2 / cs[n_col].clip(lower=1)
        post_var = 1.0 / (1.0 / v + 1.0 / tau2)
        cs["shrunk"] = post_var * (cs[est_col] / v + cs["prior_mu"] / tau2)
        cs["post_sd"] = np.sqrt(post_var)
        params[seg] = {"a": float(a), "b": float(b), "tau": float(np.sqrt(tau2)), "sigma": float(np.sqrt(s2))}
        parts.append(cs)
    return pd.concat(parts), params
