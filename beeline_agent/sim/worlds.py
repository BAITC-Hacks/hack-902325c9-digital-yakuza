"""
Генератор «миров» — альтернативных истинных моделей эффектов для настройки агента.

Скрытая модель организаторов отличается от мок-среды (мок = средние по истории), поэтому
стратегию проверяем на сотнях сгенерированных миров. Мир — таблица эффектов в формате
`_mock_impact_model()` по ВСЕМ тройкам (from, seg, to), from != to (21 × 3 × 20 = 1 260 строк),
плюс fallback-правило. Среда создаётся прогонщиком:

    from sim.worlds import make_world
    w = make_world(seed=17, scenario="random")
    env, internals = environment.make_environment(profile, w.impact_model, dict_tariff, CHANNELS,
                                                  TOTAL_BUDGET, MAX_TOTAL_CONTACTS, w.fallback_predict, seed)

Как в среде считается эффект (не меняем, подстраиваемся):
    lift_ratio = arpu_change_pct × min(conversion_rate × множитель_канала, 1);  эффект = lift_ratio × predicted_arpu

Основа мира (кроме mock и resample) — «истина», разложенная так же, как приор (prior/hier.py):
    Δ%(from, seg, to) = уровень страты L(from, seg) + контраст цели C(from, seg, to)
  * L — в основном регрессия к среднему (плацебо без смен тарифа даёт тот же рисунок), поэтому
    в искажённых мирах он сдвигается по сегментам и стратам: дрейф реальной аудитории другой;
  * C = ρ·C_истории + √(1−ρ²)·шум с иерархией τ (цель, сегмент×цель, ячейка); ρ — насколько
    переносится контраст (в истории split-half 0.92–0.96, в чужой среде — неизвестно);
  * тройки без истории получают C из той же иерархии (для 12 тарифов — только шум: перенос по
    атрибутам тарифа на истории не лучше нуля);
  * conversion — доля переходов (from, seg) → to со сглаживанием: сумма по целям ячейки = 1.
resample — как мок, но по бутстрэпу абонентов истории (так организаторы могли построить реальную
модель по другой выборке): тройки без наблюдений — мок-fallback, поэтому сумма conversion > 1, как у мока.
Затем сценарий искажает основу (см. SCENARIOS). Ограничения: Δ% в [−1; 3], conversion в (0; 0.9],
сумма conversion по целям одной (from, seg) не больше 1.

Проверка: cd beeline_agent && python -m sim.worlds --check
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mock_environment import _mock_fallback, _mock_impact_model  # noqa: E402
from scoring_core import CHANNELS, MAX_CAMPAIGNS, MAX_TOTAL_CONTACTS, TOTAL_BUDGET  # noqa: E402
from prior import hier  # noqa: E402
from prior.history import clean_history  # noqa: E402

SEGMENTS = ("LOW", "MID", "HIGH")
COLUMNS = ["tariff_plan_code_from", "tariff_plan_code_to", "arpu_segment", "arpu_change_pct", "conversion_rate"]
CONV_ALPHA = 10.0            # сглаживание доли переходов к структуре сегмента
CONV_UNIFORM_SHARE = 0.10    # доля «равномерной» массы, чтобы у целей без истории conversion > 0
CONV_MAX = 0.9
PCT_CLIP = (-1.0, 3.0)
RANDOM_STRENGTH = (0.5, 2.0)  # сила сдвига в сценарии random; шире/выше — история полезна реже (калибровка «~15×»)

SCENARIOS = {
    "mock": "ровно мок-среда: _mock_impact_model по истории + _mock_fallback (контроль)",
    "random": "основной: дрейф уровня по сегментам/стратам, ослабленный контраст (ρ) + смесь искажений ниже",
    "noise": "Δ% каждой тройки × логнормальный множитель вокруг 1",
    "flip": "у 20–30% троек знак эффекта меняется",
    "shift": "лучшие цели перемешаны: в 50–100% ячеек эффекты переставлены между целевыми тарифами",
    "stingy": "все эффекты × 0.2–0.4 — зарабатывать почти нечего",
    "unknown_rich": "лучшие эффекты — у тарифов без истории (2, 3, 5, 6, 7, 14–20)",
    "high_rich": "HIGH (74% денег) на деле прибыльнее, чем по истории: Δ% в HIGH сдвинут вверх на 0.15–0.4",
    "resample": "как мок, но по бутстрэпу абонентов истории (другая выборка — та же конструкция)",
    # добавлены в конце — миры прежних сценариев с теми же seed не меняются
    "random_soft": "как random, но сила сдвига 0.25–1.0 (история почти верна)",
    "random_hard": "как random, но сила сдвига 1.2–2.5 (история почти бесполезна; «~15×» по прокси оракула)",
    "causal": "эффект без регрессии к среднему: из уровня страты вычтен дрейф сегмента по плацебо "
              "(апрель–июнь → июль–сентябрь без смен тарифа); LOW перестаёт быть «золотом»",
}
STRENGTH_BY_SCENARIO = {"random": None, "random_soft": (0.25, 1.0), "random_hard": (1.2, 2.5)}
_SCENARIO_ID = {name: i for i, name in enumerate(SCENARIOS)}


@dataclass
class World:
    name: str                      # например "random-17" или "flip-3"
    seed: int
    impact_model: pd.DataFrame     # tariff_plan_code_from, tariff_plan_code_to, arpu_segment,
                                   # arpu_change_pct, conversion_rate
    fallback_predict: Callable     # (current_tariff, target_tariff, arpu_segment, dict_tariff, fallback_conversion)
                                   #   -> (arpu_change_pct, conversion_rate)
    params: dict = field(default_factory=dict)


# --------------------------------------------------------------------- fallback
class MockFallback:
    """_mock_fallback с фиксированной медианной конверсией мока (чтобы mock-мир совпадал с мок-средой)."""

    def __init__(self, conversion: float):
        self.conversion = float(conversion)

    def __call__(self, current_tariff, target_tariff, arpu_segment, dict_tariff, fallback_conversion):
        return _mock_fallback(current_tariff, target_tariff, arpu_segment, dict_tariff, self.conversion)


class PriceFallback:
    """Слабое правило по цене: половина мок-правила. Вызывается редко — таблица покрывает все тройки
    (нужно только абонентам без тарифа или сегмента)."""

    def __init__(self, scale: float = 0.5):
        self.scale = float(scale)

    def __call__(self, current_tariff, target_tariff, arpu_segment, dict_tariff, fallback_conversion):
        pct, conv = _mock_fallback(current_tariff, target_tariff, arpu_segment, dict_tariff, fallback_conversion)
        return self.scale * pct, conv


# ------------------------------------------------------------------ база (кэш)
_BASE = None


def _base() -> dict:
    """Всё, что не зависит от seed: сетка троек, оценки истории, мок-модель. Считается один раз."""
    global _BASE
    if _BASE is not None:
        return _BASE
    raw = pd.read_csv(ROOT / "data" / "change_tariff.csv")
    tariffs = pd.read_csv(ROOT / "data" / "dict_tariff.csv")
    price = tariffs.set_index("tariff_plan_code")["price_tariff"].astype(float)
    median_price = float(price.median())
    codes = sorted(price.index, key=lambda t: int(t.split("_")[1]))

    grid = pd.DataFrame([(f, t, s) for f in codes for s in SEGMENTS for t in codes if f != t],
                        columns=["tariff_plan_code_from", "tariff_plan_code_to", "arpu_segment"])
    grid["dprice"] = ((grid["tariff_plan_code_to"].map(price) - grid["tariff_plan_code_from"].map(price))
                      / median_price).values
    key = ["tariff_plan_code_from", "tariff_plan_code_to", "arpu_segment"]

    # мок-модель и её значения на всей сетке (там, где тройки нет, — мок-fallback)
    mock_im = _mock_impact_model(raw)
    mock_conv_median = float(mock_im["conversion_rate"].median())
    m = grid.merge(mock_im[key + ["arpu_change_pct", "conversion_rate"]], on=key, how="left")
    fb = [_mock_fallback(f, t, s, tariffs, mock_conv_median)
          for f, t, s in zip(m["tariff_plan_code_from"], m["tariff_plan_code_to"], m["arpu_segment"])]
    in_mock = m["arpu_change_pct"].notna().values
    pct_mock = np.where(in_mock, m["arpu_change_pct"].values, [x[0] for x in fb])
    conv_mock = np.where(in_mock, m["conversion_rate"].values, [x[1] for x in fb])

    # история после чистки: уровень страты + контраст цели (иерархический EB, prior/hier.py)
    hist, _ = clean_history(raw)
    model = hier.fit(hist, tariffs)
    preds = pd.DataFrame([hier.predict(model, f, s, t) for f, t, s in
                          zip(grid["tariff_plan_code_from"], grid["tariff_plan_code_to"], grid["arpu_segment"])])
    n_tab = hist.groupby(["tariff_from", "arpu_segment", "tariff_to"]).size()
    n_obs = np.array([n_tab.get((f, s, t), 0) for f, t, s in
                      zip(grid["tariff_plan_code_from"], grid["tariff_plan_code_to"], grid["arpu_segment"])], dtype=float)
    observed = n_obs > 0
    seg = grid["arpu_segment"].values
    tau = model["tau"]

    # conversion: сглаженная доля переходов; сумма по 20 целям каждой ячейки = 1
    cell_id = pd.factorize(grid["tariff_plan_code_from"] + "|" + grid["arpu_segment"])[0]
    seg_share = hist.groupby(["arpu_segment", "tariff_to"]).size() / hist.groupby("arpu_segment").size()
    p_seg = np.array([seg_share.get((s, t), 0.0) for s, t in zip(seg, grid["tariff_plan_code_to"])])
    p_seg = (1 - CONV_UNIFORM_SHARE) * p_seg + CONV_UNIFORM_SHARE / 20.0
    p_seg = p_seg / np.bincount(cell_id, weights=p_seg)[cell_id]          # нормировка внутри ячейки
    N_cell = np.bincount(cell_id, weights=n_obs)[cell_id]
    conv_base = (n_obs + CONV_ALPHA * p_seg) / (N_cell + CONV_ALPHA)

    unseen = sorted(set(codes) - set(hist["tariff_to"]), key=lambda t: int(t.split("_")[1]))
    pct_hist = (preds["level"] + preds["contrast"]).values
    # плацебо: дрейф Δ% по сегментам без смены тарифа (апрель–июнь → июль–сентябрь) — регрессия к среднему
    am = pd.read_csv(ROOT / "data" / "arpu_monthly.csv").drop_duplicates()
    piv = am.groupby(["ID_NUMBER", "TIME_KEY"])["ARPU_1M"].mean().unstack()
    p1 = piv[["2026-04-01", "2026-05-01", "2026-06-01"]].mean(axis=1)
    p2 = piv[["2026-07-01", "2026-08-01", "2026-09-01"]].mean(axis=1)
    ok = p1.notna() & p2.notna() & (p1 >= 100)
    placebo = (((p2[ok] - p1[ok]) / p1[ok]).clip(-1, 3)
               .groupby(pd.cut(p1[ok], [-np.inf, 1000, 5000, np.inf], labels=list(SEGMENTS)), observed=True).mean())
    placebo_drift = np.array([float(placebo.get(s, 0.0)) for s in seg])
    pos = {s: np.sort(pct_hist[observed & (seg == s)]) for s in SEGMENTS}
    target_id = pd.factorize(grid["tariff_plan_code_to"])[0]
    segtarget_id = pd.factorize(grid["arpu_segment"] + "|" + grid["tariff_plan_code_to"])[0]
    _BASE = {
        "grid": grid[key], "tariffs": tariffs, "price": price, "median_price": median_price,
        "mock_im": mock_im, "mock_conv_median": mock_conv_median, "in_mock": in_mock,
        "pct_mock": pct_mock.astype(float), "conv_mock": conv_mock.astype(float),
        "observed": observed, "n_obs": n_obs, "pct_hist": pct_hist.astype(float),
        "level": preds["level"].values, "level_sd": preds["level_sd"].values,
        "contrast": preds["contrast"].values, "contrast_sd": preds["contrast_sd"].values,
        "tau": tau, "target_id": target_id, "segtarget_id": segtarget_id,
        "raw": raw, "ids": raw["ID_NUMBER"].unique(),
        "seg": seg, "cell_id": cell_id, "n_cells": int(cell_id.max() + 1),
        "conv_base": conv_base, "unseen_targets": unseen, "placebo_drift": placebo_drift,
        "placebo_by_segment": {k: round(float(v), 3) for k, v in placebo.items()},
        "is_unseen": grid["tariff_plan_code_to"].isin(unseen).values,
        "seg_pct_quantiles": {s: (float(np.quantile(v, 0.80)), float(np.quantile(v, 0.95))) for s, v in pos.items()},
    }
    return _BASE


# ------------------------------------------------------------- искажения
def _plausible_truth(base: dict, rng, contrast_keep: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Истина из разложения истории: L + C с их неопределённостью; при contrast_keep < 1 контраст истории
    переносится лишь частично, остальное — новый шум с той же иерархией (цель, сегмент×цель, ячейка)."""
    n = len(base["pct_hist"])
    z_level = rng.standard_normal(base["n_cells"])[base["cell_id"]]            # один сдвиг на страту
    level = base["level"] + base["level_sd"] * z_level
    contrast_hist = base["contrast"] + base["contrast_sd"] * rng.standard_normal(n)
    tau = base["tau"]
    fresh = (tau["target"] * rng.standard_normal(base["target_id"].max() + 1)[base["target_id"]]
             + tau["seg_target"] * rng.standard_normal(base["segtarget_id"].max() + 1)[base["segtarget_id"]]
             + tau["cell"] * rng.standard_normal(n))
    contrast = contrast_keep * contrast_hist + np.sqrt(max(1.0 - contrast_keep ** 2, 0.0)) * fresh
    return level + contrast, base["conv_base"].copy()


def _resample(base: dict, rng) -> pd.DataFrame:
    """Мок-конструкция по бутстрэпу абонентов истории + мок-fallback на тройках без наблюдений."""
    ids = rng.choice(base["ids"], size=len(base["ids"]), replace=True)
    boot = base["raw"].set_index("ID_NUMBER").loc[ids].reset_index()
    im = _mock_impact_model(boot)
    key = ["tariff_plan_code_from", "tariff_plan_code_to", "arpu_segment"]
    m = base["grid"].merge(im[key + ["arpu_change_pct", "conversion_rate"]], on=key, how="left")
    conv_med = float(im["conversion_rate"].median())
    miss = m["arpu_change_pct"].isna().values
    fb = [_mock_fallback(f, t, s, base["tariffs"], conv_med) for f, t, s in
          zip(m.loc[miss, "tariff_plan_code_from"], m.loc[miss, "tariff_plan_code_to"], m.loc[miss, "arpu_segment"])]
    m.loc[miss, "arpu_change_pct"] = [x[0] for x in fb]
    m.loc[miss, "conversion_rate"] = [x[1] for x in fb]
    return m[COLUMNS].reset_index(drop=True), conv_med


def _noise(pct, rng, sigma):
    return pct * np.exp(sigma * rng.standard_normal(len(pct)))


def _flip(pct, rng, share):
    flip = rng.random(len(pct)) < share
    return np.where(flip, -pct, pct), int(flip.sum())


def _shift(pct, conv, base, rng, share_cells):
    """В выбранных ячейках пары (Δ%, conversion) переставляются между целевыми тарифами."""
    pct, conv = pct.copy(), conv.copy()
    shifted = 0
    for c in range(base["n_cells"]):
        if rng.random() >= share_cells:
            continue
        idx = np.flatnonzero(base["cell_id"] == c)
        perm = rng.permutation(idx)
        pct[idx], conv[idx] = pct[perm], conv[perm]
        shifted += 1
    return pct, conv, shifted


def _unknown_rich(pct, conv, base, rng, n_rich_range=(2, 4), damp_range=(0.3, 0.7)):
    """Лучшие эффекты — у тарифов без истории: в каждой ячейке 2–4 такие цели получают сильный Δ%
    (80–95-й перцентиль истории сегмента) и заметную conversion; известные цели ослаблены."""
    pct, conv = pct.copy(), conv.copy()
    damp = rng.uniform(*damp_range)
    known = ~base["is_unseen"]
    pct[known & (pct > 0)] *= damp
    for c in range(base["n_cells"]):
        idx = np.flatnonzero((base["cell_id"] == c) & base["is_unseen"])
        n_rich = min(len(idx), int(rng.integers(n_rich_range[0], n_rich_range[1] + 1)))
        rich = rng.choice(idx, size=n_rich, replace=False)
        lo, hi = base["seg_pct_quantiles"][base["seg"][rich[0]]]
        pct[rich] = np.abs(rng.uniform(lo, hi, size=n_rich)) + 0.05
        conv[rich] = rng.uniform(0.10, 0.30, size=n_rich)
    return pct, conv, damp


def _level_drift(pct, conv, base, rng, strength):
    """Дрейф уровня силы strength: реальная аудитория «регрессирует к среднему» иначе, чем история
    (сдвиг по сегментам и стратам), и отклик частично перераспределён между целями ячейки."""
    seg_idx = pd.factorize(base["seg"])[0]
    d_seg = rng.normal(0.0, 0.25 * strength, 3)[seg_idx]
    d_stratum = rng.normal(0.0, 0.15 * strength, base["n_cells"])[base["cell_id"]]
    pct = pct + d_seg + d_stratum
    conv_mix = float(np.clip(0.3 * strength, 0.0, 0.9))
    random_share = rng.gamma(0.3, 1.0, len(conv))
    random_share = random_share / np.bincount(base["cell_id"], weights=random_share)[base["cell_id"]]
    conv = (1 - conv_mix) * conv + conv_mix * random_share
    first = np.unique(seg_idx, return_index=True)[1]
    return pct, conv, {"seg_shift": {str(base["seg"][i]): round(float(d_seg[i]), 3) for i in first},
                       "conv_mix": round(conv_mix, 3)}


def _finalize(pct, conv, base):
    pct = np.clip(pct, *PCT_CLIP)
    conv = np.clip(conv, 1e-4, CONV_MAX)
    total = np.bincount(base["cell_id"], weights=conv)[base["cell_id"]]
    conv = np.where(total > 1.0, conv / total, conv)          # сумма по целям ячейки ≤ 1
    return pct, conv


def _frame(base, pct, conv) -> pd.DataFrame:
    df = base["grid"].copy()
    df["arpu_change_pct"] = pct.astype(float)
    df["conversion_rate"] = conv.astype(float)
    return df[COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------- интерфейс
def list_scenarios() -> list[str]:
    return list(SCENARIOS)


def make_world(seed: int, scenario: str = "random") -> World:
    if scenario not in SCENARIOS:
        raise ValueError(f"Неизвестный сценарий {scenario!r}. Доступны: {list_scenarios()}")
    base = _base()
    name = f"{scenario}-{seed}"
    if scenario == "mock":
        return World(name, seed, _frame(base, base["pct_mock"], base["conv_mock"]),
                     MockFallback(base["mock_conv_median"]), {"scenario": "mock"})

    rng = np.random.default_rng([int(seed), _SCENARIO_ID[scenario]])
    params: dict = {"scenario": scenario}
    if scenario == "resample":
        table, conv_med = _resample(base, rng)
        return World(name, int(seed), table, MockFallback(conv_med), params)
    if scenario in STRENGTH_BY_SCENARIO:
        params["strength"] = float(rng.uniform(*(STRENGTH_BY_SCENARIO[scenario] or RANDOM_STRENGTH)))
        params["contrast_rho"] = float(np.clip(1.0 - 0.4 * params["strength"], 0.0, 1.0))
    pct, conv = _plausible_truth(base, rng, contrast_keep=params.get("contrast_rho", 1.0))

    if scenario == "noise":
        params["noise_sigma"] = float(rng.uniform(0.3, 0.8))
        pct = _noise(pct, rng, params["noise_sigma"])
    elif scenario == "flip":
        params["flip_share"] = float(rng.uniform(0.20, 0.30))
        pct, params["flipped"] = _flip(pct, rng, params["flip_share"])
    elif scenario == "shift":
        params["shift_share_cells"] = float(rng.uniform(0.5, 1.0))
        pct, conv, params["shifted_cells"] = _shift(pct, conv, base, rng, params["shift_share_cells"])
    elif scenario == "stingy":
        params["scale"] = float(rng.uniform(0.2, 0.4))
        pct = pct * params["scale"]
    elif scenario == "unknown_rich":
        pct, conv, params["known_damp"] = _unknown_rich(pct, conv, base, rng)
    elif scenario == "high_rich":
        params["high_shift"] = float(rng.uniform(0.15, 0.40))
        is_high = base["seg"] == "HIGH"
        cell_noise = rng.normal(0.0, 0.1, base["n_cells"])[base["cell_id"]]
        pct = np.where(is_high, pct + params["high_shift"] + cell_noise, pct)
    elif scenario == "causal":
        # реальная модель посчитана «причинно»: дрейф регрессии к среднему убран (± погрешность плацебо)
        params["placebo_drift"] = base["placebo_by_segment"]
        pct = pct - base["placebo_drift"] * float(rng.uniform(0.8, 1.2))
    elif scenario in STRENGTH_BY_SCENARIO:
        # сила сдвига strength: 0.5 — история почти верна (ρ = 0.8), 2 — контраст почти не переносится (ρ = 0.2)
        pct, conv, info = _level_drift(pct, conv, base, rng, params["strength"])
        params.update(info)
        # плюс каждое из «именных» искажений с вероятностью 0.3
        if rng.random() < 0.3:
            params["noise_sigma"] = float(rng.uniform(0.2, 0.8))
            pct = _noise(pct, rng, params["noise_sigma"])
        if rng.random() < 0.3:
            params["flip_share"] = float(rng.uniform(0.05, 0.30))
            pct, params["flipped"] = _flip(pct, rng, params["flip_share"])
        if rng.random() < 0.3:
            params["shift_share_cells"] = float(rng.uniform(0.2, 0.8))
            pct, conv, params["shifted_cells"] = _shift(pct, conv, base, rng, params["shift_share_cells"])
        if rng.random() < 0.3:
            pct, conv, params["known_damp"] = _unknown_rich(pct, conv, base, rng)
        params["scale"] = float(np.exp(rng.normal(-0.2, 0.4)))       # от «скупого» до «щедрого» мира
        pct = pct * params["scale"]
        params["conv_scale"] = float(np.exp(rng.normal(0.0, 0.3)))  # другой уровень отклика
        conv = conv * params["conv_scale"]

    pct, conv = _finalize(pct, conv, base)
    return World(name, int(seed), _frame(base, pct, conv), PriceFallback(0.5), params)


def make_suite(n_per_scenario: int = 50, scenarios=None, start_seed: int = 0) -> list[World]:
    """Фиксированный набор миров — «локальная тестовая выборка» для сравнения гипотез на одних и тех же мирах."""
    scenarios = scenarios or [s for s in list_scenarios() if s != "mock"]
    return [make_world(start_seed + i, s) for s in scenarios for i in range(n_per_scenario)]


# ---------------------------------------------------- прокси «оракул / без разведки»
_AUD = None
_MULT = np.array([c["conversion_multiplier"] for c in CHANNELS.values()])   # push, sms, digital_ads, call
_COST = np.array([float(c["cost_per_contact"]) for c in CHANNELS.values()])


def _audience(base: dict) -> dict:
    """Ячейки аудитории, привязанные к строкам сетки: размер (≤ 5 000 на кампанию) и сумма predicted_arpu."""
    global _AUD
    if _AUD is None:
        prof = pd.read_csv(ROOT / "customer_profile.csv")
        cells = (prof.dropna(subset=["current_tariff", "arpu_segment"]).sort_values("ID_NUMBER")
                 .groupby(["current_tariff", "arpu_segment"])["predicted_arpu"]
                 .agg(lambda x: (min(len(x), 5000), float(x.head(5000).sum()))))
        g = base["grid"]
        key = list(zip(g["tariff_plan_code_from"], g["arpu_segment"]))
        n = np.array([cells.get(k, (0, 0.0))[0] for k in key], dtype=float)
        a = np.array([cells.get(k, (0, 0.0))[1] for k in key], dtype=float)
        _AUD = {"n": n, "arpu_sum": a}
    return _AUD


def _plan_value(pct_choose, conv_choose, pct_true, conv_true, base, contacts=MAX_TOTAL_CONTACTS,
                budget=float(TOTAL_BUDGET), max_campaigns=MAX_CAMPAIGNS):
    """Жадный план без пилотов: в каждой ячейке аудитории — лучшая (цель, канал) по «своим» эффектам,
    ячейки по убыванию ожидаемой ценности, лимиты охвата/бюджета/10 кампаний. Возвращает ценность по истине."""
    aud = _audience(base)
    n, arpu = aud["n"], aud["arpu_sum"]
    r_ch = pct_choose[:, None] * np.minimum(conv_choose[:, None] * _MULT[None, :], 1.0)
    v_ch = r_ch * arpu[:, None] - _COST[None, :] * n[:, None]
    r_true = pct_true[:, None] * np.minimum(conv_true[:, None] * _MULT[None, :], 1.0)
    v_true = r_true * arpu[:, None] - _COST[None, :] * n[:, None]
    best = pd.DataFrame({"cell": base["cell_id"], "row": np.arange(len(n)), "v": v_ch.max(axis=1)})
    best = best[n > 0].sort_values("v", ascending=False).drop_duplicates("cell")
    total, used_c, used_b, n_campaigns = 0.0, 0, 0.0, 0
    for row, v in zip(best["row"], best["v"]):
        if v <= 0 or n_campaigns >= max_campaigns:
            break
        for ch in np.argsort(-v_ch[row]):                     # самый выгодный канал, который влезает в бюджет
            cost = _COST[ch] * n[row]
            if used_c + n[row] <= contacts and used_b + cost <= budget and v_ch[row, ch] > 0:
                total += v_true[row, ch]
                used_c += n[row]
                used_b += cost
                n_campaigns += 1
                break
    return total


def oracle_gap(world: World) -> dict:
    """Во сколько раз план, знающий истинные эффекты, лучше плана «по истории без разведки» (ТЗ: ~15)."""
    base = _base()
    pct, conv = world.impact_model["arpu_change_pct"].values, world.impact_model["conversion_rate"].values
    oracle = _plan_value(pct, conv, pct, conv, base)
    blind = _plan_value(base["pct_mock"], base["conv_mock"], pct, conv, base)
    return {"oracle": oracle, "no_exploration": blind,
            "ratio": oracle / blind if blind > 0 else float("inf")}


# ------------------------------------------------------------------ проверка
def _stats(world: World, base: dict) -> dict:
    im = world.impact_model
    pct, conv = im["arpu_change_pct"].values, im["conversion_rate"].values
    q_world = pct * conv
    q_hist = base["pct_mock"] * base["conv_mock"]
    obs = base["in_mock"]
    sums = np.bincount(base["cell_id"], weights=conv)
    best_sms = pd.Series(pct * np.minimum(conv * 0.65, 1.0)).groupby(base["cell_id"]).max()
    gap = oracle_gap(world)
    return {"rows": len(im), "pos": float((pct > 0).mean()), "abs_pct": float(np.abs(pct).mean()),
            "conv": float(conv.mean()), "conv_sum_max": float(sums.max()),
            "corr_hist": float(np.corrcoef(q_world[obs], q_hist[obs])[0, 1]),
            "best_sms_median": float(best_sms.median()), "gap": min(gap["ratio"], 999.0), "oracle": gap["oracle"],
            "blind_nonpos": float(gap["no_exploration"] <= 0)}


def _check() -> bool:
    ok = True
    t0 = time.perf_counter()
    base = _base()
    print(f"База (история, сетка, мок): {time.perf_counter() - t0:.2f} с, троек в сетке {len(base['grid'])}")
    print(f"{'сценарий':13s} {'троек':>6s} {'pct>0':>6s} {'|pct|':>6s} {'conv':>6s} {'Σconv':>6s} "
          f"{'corr':>5s} {'лучш.SMS':>8s} {'оракул,млн':>10s} {'оракул/слепой':>13s} {'слепой≤0':>8s} {'время,с':>7s}")
    for scenario in list_scenarios():
        times, stats = [], []
        for seed in range(20):
            t = time.perf_counter()
            w = make_world(seed, scenario)
            times.append(time.perf_counter() - t)
            stats.append(_stats(w, base))
        st = {k: float(np.median([s[k] for s in stats])) for k in stats[0]}
        rows_ok = all(s["rows"] == 1260 for s in stats)
        ok &= rows_ok and max(times) < 1.0
        print(f"{scenario:13s} {int(st['rows']):6d} {st['pos']:6.2f} {st['abs_pct']:6.2f} {st['conv']:6.3f} "
              f"{st['conv_sum_max']:6.2f} {st['corr_hist']:5.2f} {st['best_sms_median']:8.3f} "
              f"{st['oracle'] / 1e6:10.2f} {st['gap']:13.1f} {np.mean([x['blind_nonpos'] for x in stats]):8.0%} "
              f"{max(times):7.3f}")
        if not rows_ok:
            print(f"  [!] {scenario}: число троек не 1260")

    # mock совпадает с _mock_impact_model: те же значения на тройках истории, на остальных — мок-fallback
    w = make_world(0, "mock")
    key = ["tariff_plan_code_from", "tariff_plan_code_to", "arpu_segment"]
    m = base["mock_im"].merge(w.impact_model, on=key, suffixes=("_mock", "_world"))
    same = (len(m) == len(base["mock_im"])
            and np.allclose(m["arpu_change_pct_mock"], m["arpu_change_pct_world"])
            and np.allclose(m["conversion_rate_mock"], m["conversion_rate_world"]))
    rest = w.impact_model.merge(base["mock_im"][key], on=key, how="left", indicator=True)
    rest = rest[rest["_merge"] == "left_only"]
    fb_ok = all(np.allclose(w.fallback_predict(r.tariff_plan_code_from, r.tariff_plan_code_to, r.arpu_segment,
                                               base["tariffs"], 0.0),
                            (r.arpu_change_pct, r.conversion_rate))
                for r in rest.head(200).itertuples())
    print(f"mock: совпадает с _mock_impact_model: {'да' if same and fb_ok else 'НЕТ'} "
          f"({len(m)} троек истории, остальные {len(rest)} = мок-fallback)")
    ok &= same and fb_ok

    # детерминизм
    a, b = make_world(7), make_world(7)
    det = a.impact_model.equals(b.impact_model) and a.params == b.params
    diff = not make_world(7).impact_model.equals(make_world(8).impact_model)
    print(f"make_world(7) дважды — одинаковые таблицы: {'да' if det else 'НЕТ'}; seed 7 и 8 различаются: "
          f"{'да' if diff else 'НЕТ'}")
    ok &= det and diff
    print("ИТОГ:", "OK" if ok else "ЕСТЬ ОШИБКИ")
    return ok


def _smoke(n: int) -> None:
    """Быстрый прогон текущего agent.py (без изменений) на n мирах каждого сценария — не замена прогонщику."""
    from environment import make_environment
    from scoring_core import sanitize_campaigns, score_campaigns
    import agent as agent_mod

    profile = pd.read_csv(ROOT / "customer_profile.csv")
    tariffs = pd.read_csv(ROOT / "data" / "dict_tariff.csv")
    cols = ["filter_arpu_segment", "filter_data_segment", "filter_call_segment", "filter_current_tariff", "explicit_ids"]
    for scenario in list_scenarios():
        nets = []
        for seed in range(n):
            w = make_world(seed, scenario)
            env, internals = make_environment(profile, w.impact_model, tariffs, CHANNELS, TOTAL_BUDGET,
                                              MAX_TOTAL_CONTACTS, w.fallback_predict, seed=seed)
            final = sanitize_campaigns(agent_mod.Agent(verbose=False).act(env), env.tariffs)[:MAX_CAMPAIGNS]
            camps = pd.DataFrame(internals.executed_pilot_campaigns() + final)
            for c in cols:
                if c not in camps.columns:
                    camps[c] = None
            res = score_campaigns(camps, env.customer_profile, w.impact_model, env.tariffs,
                                  env.customer_profile["predicted_arpu"].sum(), w.fallback_predict)
            nets.append(res["net_arpu_gain"])
        print(f"{scenario:13s} медиана {np.median(nets):>12,.0f}   мин {min(nets):>12,.0f}   в плюс {sum(x > 0 for x in nets)}/{n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Генератор миров")
    parser.add_argument("--check", action="store_true", help="проверка готовности")
    parser.add_argument("--show", metavar="SCENARIO", help="показать параметры нескольких миров сценария")
    parser.add_argument("--smoke", type=int, metavar="N", help="прогнать текущий agent.py на N мирах каждого сценария")
    args = parser.parse_args()
    if args.smoke:
        _smoke(args.smoke)
    elif args.show:
        for s in range(5):
            w = make_world(s, args.show)
            print(w.name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in w.params.items()})
    else:
        sys.exit(0 if _check() else 1)
