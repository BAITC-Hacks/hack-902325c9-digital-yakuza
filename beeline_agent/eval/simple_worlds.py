"""
Простые миры для тренажёра — временная замена генератора Тимура (sim/worlds.py).

Мир = скрытая модель эффектов, которую агент не видит. Интерфейс World тот же,
что у генератора Тимура, поэтому источник миров меняется без переделки тренажёра.

Сценарии:
  mock    — ровно мок-среда организаторов (эффекты = средние по истории)
  noise   — эффект каждой связки умножен на случайный множитель
  flip    — у части связок знак эффекта перевёрнут (downsell вместо upsell)
  shift   — внутри каждой ячейки «тариф × сегмент» эффекты перемешаны между целями
  stingy  — все эффекты сильно слабее
  random  — случайная смесь всех искажений со случайной силой (основной)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from mock_environment import _mock_fallback, _mock_impact_model

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ("mock", "noise", "flip", "shift", "stingy", "random")


@dataclass
class World:
    name: str
    seed: int
    impact_model: pd.DataFrame
    fallback_predict: Callable
    params: dict = field(default_factory=dict)


_HISTORY_MODEL = None


def _history_model() -> pd.DataFrame:
    global _HISTORY_MODEL
    if _HISTORY_MODEL is None:
        _HISTORY_MODEL = _mock_impact_model(pd.read_csv(ROOT / "data" / "change_tariff.csv"))
    return _HISTORY_MODEL.copy()


def _scaled_fallback(scale: float) -> Callable:
    def fallback(current_tariff, target_tariff, arpu_segment, dict_tariff, fallback_conversion):
        pct, conv = _mock_fallback(current_tariff, target_tariff, arpu_segment, dict_tariff, fallback_conversion)
        return float(np.clip(pct * scale, -1.0, 3.0)), conv
    return fallback


def _apply(model, rng, noise_sd=0.0, flip_share=0.0, shift=False, scale=1.0):
    m = model.copy()
    if noise_sd > 0:
        m["arpu_change_pct"] *= rng.lognormal(0.0, noise_sd, len(m))
    if flip_share > 0:
        flip = rng.random(len(m)) < flip_share
        m.loc[flip, "arpu_change_pct"] *= -1
    if shift:
        for _, idx in m.groupby(["tariff_plan_code_from", "arpu_segment"], observed=True).groups.items():
            idx = list(idx)
            m.loc[idx, "arpu_change_pct"] = rng.permutation(m.loc[idx, "arpu_change_pct"].to_numpy())
    m["arpu_change_pct"] = (m["arpu_change_pct"] * scale).clip(-1.0, 3.0)
    return m


def make_world(seed: int, scenario: str = "random") -> World:
    rng = np.random.default_rng(seed)
    base = _history_model()
    if scenario == "mock":
        return World(f"mock-{seed}", seed, base, _mock_fallback, {})
    if scenario == "noise":
        params = {"noise_sd": 0.6}
    elif scenario == "flip":
        params = {"flip_share": 0.25}
    elif scenario == "shift":
        params = {"shift": True}
    elif scenario == "stingy":
        params = {"scale": 0.3}
    elif scenario == "random":
        params = {"noise_sd": float(rng.uniform(0.2, 0.8)),
                  "flip_share": float(rng.uniform(0.0, 0.3)),
                  "shift": bool(rng.random() < 0.3),
                  "scale": float(rng.uniform(0.3, 1.2))}
    else:
        raise ValueError(f"Неизвестный сценарий {scenario!r}. Есть: {SCENARIOS}")
    fallback_scale = float(rng.uniform(-0.5, 1.5)) if scenario == "random" else 1.0
    params["fallback_scale"] = fallback_scale
    model = _apply(base, rng, **{k: v for k, v in params.items() if k != "fallback_scale"})
    return World(f"{scenario}-{seed}", seed, model, _scaled_fallback(fallback_scale), params)
