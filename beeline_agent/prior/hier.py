"""
Иерархическая оценка Δ% смены тарифа по истории: «уровень страты» + «контраст цели».

Почему так. Плацебо-проверка (апрель–июнь против июля–сентября, без всяких смен тарифа) даёт
тот же рисунок, что и смены: LOW «растёт» (+0.88 в Δ%), HIGH «падает» (−0.04). Это регрессия к
среднему: общий сдвиг ARPU внутри страты «тариф_откуда × сегмент» одинаков для любых целей.
Поэтому Δ% раскладываем на две части с разной переносимостью:

    pct(from, seg, to) = L(from, seg)      — уровень страты: в основном дрейф (регрессия к среднему)
                       + C(from, seg, to)  — контраст: чем эта цель лучше/хуже средней цели страты

Контраст оцениваем иерархическим empirical Bayes по трём уровням: цель → сегмент × цель → ячейка.
На каждом уровне оценка сжимается к нулю тем сильнее, чем меньше наблюдений; сила сжатия
(межгрупповая дисперсия τ²) оценивается по данным методом моментов. Надёжность контрастов
(split-half, Спирмен–Браун): цель 0.96, сегмент×цель 0.92, ячейка 0.71 — сигнал реальный.
Цели без истории: пробуем перенос контраста по сходству атрибутов тарифа (цена, пакет данных,
минуты) и проверяем его leave-one-out на 9 известных целях. Если перенос не лучше «нуля»
(контраст как у средней цели страты), берём ноль — на этих данных так и выходит
(LOO RMSE переноса 0.15–0.30 против 0.13–0.24 у нуля), и ранжировать 12 тарифов по истории нельзя.

Важно: и уровень, и контраст описывают ARPU тех, кто сменил тариф сам. Среда считает эффект на
каждый контакт (× доля согласившихся) — это делает conversion в build_prior.py.
"""
import numpy as np
import pandas as pd

KEY = ["tariff_from", "arpu_segment", "tariff_to"]
SEGMENTS = ("LOW", "MID", "HIGH")
MIN_N_TAU = 5            # группы для оценки межгрупповой дисперсии
TAU2_FLOOR = 1e-4
KERNEL_BANDWIDTH = 1.0   # ширина ядра по стандартизованным атрибутам тарифа


def _mom_tau2(estimate: np.ndarray, se2: np.ndarray, w: np.ndarray) -> float:
    """Метод моментов: дисперсия оценок между группами минус средняя дисперсия ошибки."""
    if len(estimate) < 3:
        return TAU2_FLOOR
    m = np.average(estimate, weights=w)
    return float(max(np.average((estimate - m) ** 2, weights=w) - np.average(se2, weights=w), TAU2_FLOOR))


def _shrink(g: pd.DataFrame, target: pd.Series, se2: pd.Series, tau2) -> tuple[pd.Series, pd.Series]:
    """Нормальная модель: posterior = (est/se2 + target/tau2) / (1/se2 + 1/tau2)."""
    prec = 1.0 / se2 + 1.0 / tau2
    return (g["mean"] / se2 + target / tau2) / prec, np.sqrt(1.0 / prec)


def tariff_features(tariffs: pd.DataFrame) -> pd.DataFrame:
    t = tariffs.set_index("tariff_plan_code")
    f = pd.DataFrame({"price": t["price_tariff"].astype(float),
                      "ldata": np.log2(1.0 + t["Data_in_PKG"].astype(float)),
                      "minutes": (t["Min_another_operator_in_PKG"] + t["Min_another_operator_and_city_in_PKG"]).astype(float)})
    return (f - f.mean()) / f.std()


def _kernel_predict(values: pd.Series, feats: pd.DataFrame, target: str, exclude=()) -> float:
    src = [t for t in values.index if t != target and t not in exclude]
    d2 = ((feats.loc[src] - feats.loc[target]) ** 2).sum(axis=1)
    w = np.exp(-d2 / (2 * KERNEL_BANDWIDTH ** 2))
    return float((w * values.loc[src]).sum() / w.sum()) if w.sum() > 1e-12 else 0.0


def fit(hist: pd.DataFrame, tariffs: pd.DataFrame) -> dict:
    """hist: строки tariff_from, arpu_segment, tariff_to, pct (после чистки prior/history.py)."""
    d = hist[KEY + ["pct"]].copy()

    # --- 1. уровень страты (from, seg): сжатие к среднему сегмента ------------------------
    strata = d.groupby(["tariff_from", "arpu_segment"])["pct"].agg(n="size", mean="mean").reset_index()
    d = d.merge(strata[["tariff_from", "arpu_segment", "mean"]].rename(columns={"mean": "stratum_mean"}),
                on=["tariff_from", "arpu_segment"])
    d["dev2"] = (d["pct"] - d["stratum_mean"]) ** 2
    by_segment = d.groupby("arpu_segment")
    within_var = by_segment["dev2"].sum() / (by_segment.size() - by_segment["tariff_from"].nunique()).clip(lower=1)
    seg_mean = d.groupby("arpu_segment")["pct"].mean()
    level_rows, level_params = [], {}
    for seg, g in strata.groupby("arpu_segment"):
        se2 = within_var[seg] / g["n"]
        big = g["n"] >= MIN_N_TAU
        tau2 = _mom_tau2(g.loc[big, "mean"].values, se2[big].values, g.loc[big, "n"].values)
        estimate, sd = _shrink(g, seg_mean[seg], se2, tau2)
        level_rows.append(g.assign(level=estimate.values, level_sd=sd.values))
        level_params[seg] = {"mean": float(seg_mean[seg]), "tau": float(np.sqrt(tau2)), "sigma": float(np.sqrt(within_var[seg]))}
    level = pd.concat(level_rows).set_index(["tariff_from", "arpu_segment"])
    for seg in SEGMENTS:                       # сегмента нет в данных — общий уровень с широкой неопределённостью
        if seg not in level_params:
            level_params[seg] = {"mean": float(d["pct"].mean()), "tau": float(d["pct"].std()),
                                 "sigma": float(d["pct"].std())}

    # --- 2. контраст: цель → сегмент×цель → ячейка -----------------------------------------
    d["contrast"] = d["pct"] - d["stratum_mean"]
    # уровень «цель»
    by_target = d.groupby("tariff_to")["contrast"].agg(n="size", mean="mean", var="var")
    se2 = by_target["var"].fillna(by_target["var"].median()) / by_target["n"]
    tau2_target = _mom_tau2(by_target["mean"].values, se2.values, by_target["n"].values)
    target_effect, target_effect_sd = _shrink(by_target, 0.0, se2, tau2_target)
    d["resid_after_target"] = d["contrast"] - d["tariff_to"].map(target_effect)
    # уровень «сегмент × цель»
    by_seg_target = d.groupby(["arpu_segment", "tariff_to"])["resid_after_target"].agg(n="size", mean="mean", var="var")
    se2 = by_seg_target["var"].fillna(by_seg_target["var"].median()) / by_seg_target["n"]
    big = by_seg_target["n"] >= MIN_N_TAU
    tau2_seg_target = _mom_tau2(by_seg_target.loc[big, "mean"].values, se2[big].values,
                                by_seg_target.loc[big, "n"].values)
    seg_target_effect, seg_target_effect_sd = _shrink(by_seg_target, 0.0, se2, tau2_seg_target)
    seg_target_key = pd.Series(list(zip(d["arpu_segment"], d["tariff_to"])))
    d["resid_after_seg_target"] = d["resid_after_target"] - seg_target_key.map(seg_target_effect).values
    # уровень «ячейка»
    by_cell = d.groupby(KEY)["resid_after_seg_target"].agg(n="size", mean="mean")
    d["resid_dev2"] = (d["resid_after_seg_target"] - d.groupby(KEY)["resid_after_seg_target"].transform("mean")) ** 2
    n_cells = d.groupby("arpu_segment")[["tariff_from", "tariff_to"]].apply(lambda x: len(x.drop_duplicates()))
    resid_var = d.groupby("arpu_segment")["resid_dev2"].sum() / (d.groupby("arpu_segment").size() - n_cells).clip(lower=1)
    se2 = pd.Series([resid_var.get(s, resid_var.mean()) for s in by_cell.index.get_level_values("arpu_segment")],
                    index=by_cell.index) / by_cell["n"]
    big = by_cell["n"] >= MIN_N_TAU
    tau2_cell = _mom_tau2(by_cell.loc[big, "mean"].values, se2[big].values, by_cell.loc[big, "n"].values)
    cell_effect, cell_effect_sd = _shrink(by_cell, 0.0, se2, tau2_cell)

    # --- 3. перенос контраста на цели без истории по атрибутам тарифа -----------------------
    feats = tariff_features(tariffs)
    seen = target_effect.index.tolist()
    loo = np.array([_kernel_predict(target_effect, feats, t) - target_effect[t] for t in seen])
    kernel_rmse = float(np.sqrt(np.mean(loo ** 2)))
    zero_rmse = float(np.sqrt(np.mean(target_effect.values ** 2)))
    use_kernel = kernel_rmse < zero_rmse
    transfer_rmse = kernel_rmse if use_kernel else zero_rmse
    unseen = [t for t in feats.index if t not in seen]
    target_effect_unseen = {t: (_kernel_predict(target_effect, feats, t) if use_kernel else 0.0) for t in unseen}

    return {"level": level, "level_params": level_params,
            "target_effect": target_effect, "target_effect_sd": target_effect_sd,
            "seg_target_effect": seg_target_effect, "seg_target_effect_sd": seg_target_effect_sd,
            "cell_effect": cell_effect, "cell_effect_sd": cell_effect_sd,
            "tau": {"target": float(np.sqrt(tau2_target)), "seg_target": float(np.sqrt(tau2_seg_target)),
                    "cell": float(np.sqrt(tau2_cell))},
            "target_effect_unseen": target_effect_unseen, "transfer_rmse": transfer_rmse,
            "seen_targets": seen, "unseen_targets": unseen,
            "transfer": {"kernel_loo_rmse": kernel_rmse, "zero_rmse": zero_rmse,
                         "method": "kernel" if use_kernel else "zero"},
            "loo_errors": dict(zip(seen, loo.round(4)))}


def predict(model: dict, tariff_from: str, segment: str, tariff_to: str) -> dict:
    """Оценка Δ% тройки: уровень страты + контраст цели, с неопределённостью и источником каждой части."""
    seg_level = model["level_params"][segment]
    if (tariff_from, segment) in model["level"].index:
        row = model["level"].loc[(tariff_from, segment)]
        level, level_sd = float(row["level"]), float(row["level_sd"])
    else:
        level, level_sd = seg_level["mean"], seg_level["tau"]
    tau = model["tau"]
    if tariff_to in model["target_effect"].index:
        contrast = float(model["target_effect"][tariff_to])
        contrast_var, source = float(model["target_effect_sd"][tariff_to]) ** 2, "history"
    else:
        contrast = model["target_effect_unseen"][tariff_to]
        contrast_var, source = model["transfer_rmse"] ** 2, "unseen_target"
    if (segment, tariff_to) in model["seg_target_effect"].index:
        contrast += float(model["seg_target_effect"][(segment, tariff_to)])
        contrast_var += float(model["seg_target_effect_sd"][(segment, tariff_to)]) ** 2
    else:
        contrast_var += tau["seg_target"] ** 2
    cell = (tariff_from, segment, tariff_to)
    if cell in model["cell_effect"].index:
        contrast += float(model["cell_effect"][cell])
        contrast_var += float(model["cell_effect_sd"][cell]) ** 2
    else:
        contrast_var += tau["cell"] ** 2
        if source == "history":
            source = "no_history_for_cell"
    return {"level": level, "level_sd": level_sd, "contrast": contrast, "contrast_sd": float(np.sqrt(contrast_var)),
            "pct": level + contrast, "pct_sd": float(np.sqrt(level_sd ** 2 + contrast_var)), "source": source}
