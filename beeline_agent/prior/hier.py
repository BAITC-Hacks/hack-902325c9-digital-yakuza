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


def _mom_tau2(est: np.ndarray, se2: np.ndarray, w: np.ndarray) -> float:
    """Метод моментов: дисперсия оценок между группами минус средняя дисперсия ошибки."""
    if len(est) < 3:
        return TAU2_FLOOR
    m = np.average(est, weights=w)
    return float(max(np.average((est - m) ** 2, weights=w) - np.average(se2, weights=w), TAU2_FLOOR))


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
    gseg = d.groupby("arpu_segment")
    within = gseg["dev2"].sum() / (gseg.size() - gseg["tariff_from"].nunique()).clip(lower=1)
    seg_mean = d.groupby("arpu_segment")["pct"].mean()
    level_rows, level_params = [], {}
    for seg, g in strata.groupby("arpu_segment"):
        se2 = within[seg] / g["n"]
        big = g["n"] >= MIN_N_TAU
        tau2 = _mom_tau2(g.loc[big, "mean"].values, se2[big].values, g.loc[big, "n"].values)
        est, sd = _shrink(g, seg_mean[seg], se2, tau2)
        level_rows.append(g.assign(level=est.values, level_sd=sd.values))
        level_params[seg] = {"mean": float(seg_mean[seg]), "tau": float(np.sqrt(tau2)), "sigma": float(np.sqrt(within[seg]))}
    level = pd.concat(level_rows).set_index(["tariff_from", "arpu_segment"])
    for seg in SEGMENTS:                       # сегмента нет в данных — общий уровень с широкой неопределённостью
        if seg not in level_params:
            level_params[seg] = {"mean": float(d["pct"].mean()), "tau": float(d["pct"].std()),
                                 "sigma": float(d["pct"].std())}

    # --- 2. контраст: цель → сегмент×цель → ячейка -----------------------------------------
    d["c"] = d["pct"] - d["stratum_mean"]
    # уровень «цель»
    T = d.groupby("tariff_to")["c"].agg(n="size", mean="mean", var="var")
    se2 = T["var"].fillna(T["var"].median()) / T["n"]
    tau2_t = _mom_tau2(T["mean"].values, se2.values, T["n"].values)
    ct, ct_sd = _shrink(T, 0.0, se2, tau2_t)
    d["r1"] = d["c"] - d["tariff_to"].map(ct)
    # уровень «сегмент × цель»
    ST = d.groupby(["arpu_segment", "tariff_to"])["r1"].agg(n="size", mean="mean", var="var")
    se2 = ST["var"].fillna(ST["var"].median()) / ST["n"]
    big = ST["n"] >= MIN_N_TAU
    tau2_st = _mom_tau2(ST.loc[big, "mean"].values, se2[big].values, ST.loc[big, "n"].values)
    dst, dst_sd = _shrink(ST, 0.0, se2, tau2_st)
    d["r2"] = d["r1"] - pd.Series(list(zip(d["arpu_segment"], d["tariff_to"]))).map(dst).values
    # уровень «ячейка»
    F = d.groupby(KEY)["r2"].agg(n="size", mean="mean")
    d["r2dev2"] = (d["r2"] - d.groupby(KEY)["r2"].transform("mean")) ** 2
    n_cells = d.groupby("arpu_segment")[["tariff_from", "tariff_to"]].apply(lambda x: len(x.drop_duplicates()))
    resid_var = d.groupby("arpu_segment")["r2dev2"].sum() / (d.groupby("arpu_segment").size() - n_cells).clip(lower=1)
    se2 = pd.Series([resid_var.get(s, resid_var.mean()) for s in F.index.get_level_values("arpu_segment")],
                    index=F.index) / F["n"]
    big = F["n"] >= MIN_N_TAU
    tau2_f = _mom_tau2(F.loc[big, "mean"].values, se2[big].values, F.loc[big, "n"].values)
    ef, ef_sd = _shrink(F, 0.0, se2, tau2_f)

    # --- 3. перенос контраста на цели без истории по атрибутам тарифа -----------------------
    feats = tariff_features(tariffs)
    seen = ct.index.tolist()
    loo = np.array([_kernel_predict(ct, feats, t) - ct[t] for t in seen])
    kernel_rmse = float(np.sqrt(np.mean(loo ** 2)))
    zero_rmse = float(np.sqrt(np.mean(ct.values ** 2)))
    use_kernel = kernel_rmse < zero_rmse
    transfer_rmse = kernel_rmse if use_kernel else zero_rmse
    unseen = [t for t in feats.index if t not in seen]
    ct_unseen = {t: (_kernel_predict(ct, feats, t) if use_kernel else 0.0) for t in unseen}

    return {"level": level, "level_params": level_params, "ct": ct, "ct_sd": ct_sd, "dst": dst, "dst_sd": dst_sd,
            "ef": ef, "ef_sd": ef_sd, "tau": {"target": float(np.sqrt(tau2_t)), "seg_target": float(np.sqrt(tau2_st)),
                                              "cell": float(np.sqrt(tau2_f))},
            "ct_unseen": ct_unseen, "transfer_rmse": transfer_rmse, "seen_targets": seen, "unseen_targets": unseen,
            "transfer": {"kernel_loo_rmse": kernel_rmse, "zero_rmse": zero_rmse,
                         "method": "kernel" if use_kernel else "zero"},
            "loo_errors": dict(zip(seen, loo.round(4)))}


def predict(model: dict, f: str, s: str, t: str) -> dict:
    """Оценка Δ% тройки: уровень + контраст, с неопределённостью и источником каждой части."""
    lp = model["level_params"][s]
    if (f, s) in model["level"].index:
        row = model["level"].loc[(f, s)]
        level, level_sd = float(row["level"]), float(row["level_sd"])
    else:
        level, level_sd = lp["mean"], lp["tau"]
    tau = model["tau"]
    if t in model["ct"].index:
        c, var, src = float(model["ct"][t]), float(model["ct_sd"][t]) ** 2, "history"
    else:
        c, var, src = model["ct_unseen"][t], model["transfer_rmse"] ** 2, "unseen_target"
    if (s, t) in model["dst"].index:
        c += float(model["dst"][(s, t)])
        var += float(model["dst_sd"][(s, t)]) ** 2
    else:
        var += tau["seg_target"] ** 2
    if (f, s, t) in model["ef"].index:
        c += float(model["ef"][(f, s, t)])
        var += float(model["ef_sd"][(f, s, t)]) ** 2
    else:
        var += tau["cell"] ** 2
        if src == "history":
            src = "no_history_for_cell"
    return {"level": level, "level_sd": level_sd, "contrast": c, "contrast_sd": float(np.sqrt(var)),
            "pct": level + c, "pct_sd": float(np.sqrt(level_sd ** 2 + var)), "source": src}
