"""
Ядро тренажёра: запуск стратегии в мире и подсчёт результата тем же кодом, что у организаторов.

run_strategy повторяет local_eval.evaluate_agent, но для любого мира, а не только мок-среды.
upper_bound — потолок: сколько максимум можно заработать, зная истинные эффекты
(доказуемая верхняя граница, лимит 10 кампаний и стоимость пилотов не учитываются).
"""

import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from environment import make_environment
from mock_environment import CHANNELS, MAX_TOTAL_CONTACTS, TOTAL_BUDGET
from scoring_core import MAX_CAMPAIGNS, sanitize_campaigns, score_campaigns

ROOT = Path(__file__).resolve().parents[1]
FILTER_COLUMNS = ["filter_arpu_segment", "filter_data_segment", "filter_call_segment",
                  "filter_current_tariff", "explicit_ids"]


@lru_cache(maxsize=1)
def load_inputs():
    profile = pd.read_csv(ROOT / "customer_profile.csv")
    dict_tariff = pd.read_csv(ROOT / "data" / "dict_tariff.csv")
    return profile, dict_tariff


def run_strategy(world, make_strategy, env_seed):
    """Прогон одной стратегии в одном мире. Возвращает результат подсчёта + план и время."""
    profile, dict_tariff = load_inputs()
    env, internals = make_environment(
        customer_profile=profile, impact_model=world.impact_model, dict_tariff=dict_tariff,
        channels=CHANNELS, total_budget=TOTAL_BUDGET, max_total_contacts=MAX_TOTAL_CONTACTS,
        fallback_predict=world.fallback_predict, seed=env_seed)

    strategy = make_strategy()
    t0 = time.monotonic()
    error = None
    try:
        final = strategy.act(env)
    except Exception as exc:          # как у организаторов: падение = пустой план, пилоты в зачёте
        final, error = [], f"{type(exc).__name__}: {exc}"
    seconds = time.monotonic() - t0

    returned = len(final) if isinstance(final, list) else 0
    final = sanitize_campaigns(final, env.tariffs)[:MAX_CAMPAIGNS]
    pilots = internals.executed_pilot_campaigns()
    campaigns = pd.DataFrame(pilots + final)
    if campaigns.empty:
        result = {"net_arpu_gain": 0.0, "total_cost": 0.0, "risk_score_pct": 0.0}
    else:
        for col in FILTER_COLUMNS:
            if col not in campaigns.columns:
                campaigns[col] = None
        result = score_campaigns(campaigns, env.customer_profile, world.impact_model, env.tariffs,
                                 env.customer_profile["predicted_arpu"].sum(), world.fallback_predict)
    result.update({"plan": final, "n_pilots": len(pilots), "seconds": seconds, "error": error})
    result["violations"] = tz_violations(returned, final, result, len(pilots), seconds)
    return result


def tz_violations(returned, final, result, n_pilots, seconds):
    """Нарушения обязательных условий ТЗ в одном прогоне (пустой список = всё соблюдено)."""
    v = []
    if not 1 <= len(final) <= MAX_CAMPAIGNS:
        v.append(f"кампаний {len(final)} (нужно 1–10)")
    if returned > len(final):
        v.append(f"отброшено кампаний: {returned - len(final)}")
    if n_pilots == 0:
        v.append("пилотов 0")
    if result.get("total_cost", 0) > TOTAL_BUDGET:
        v.append("бюджет превышен")
    if result.get("total_contacts", 0) > MAX_TOTAL_CONTACTS:
        v.append("охват превышен")
    if seconds > 600:
        v.append("дольше 10 минут")
    if result.get("error"):
        v.append("агент упал")
    return v


def _ratio_table(world, profile, dict_tariff):
    """Истинный эффект lift_ratio для каждой ячейки (тариф × сегмент), цели и канала."""
    im = world.impact_model
    model = {(f, str(s), t): (pct, conv) for f, s, t, pct, conv in zip(
        im["tariff_plan_code_from"], im["arpu_segment"], im["tariff_plan_code_to"],
        im["arpu_change_pct"], im["conversion_rate"])}
    fallback_conversion = world.impact_model["conversion_rate"].median()
    cells = profile.dropna(subset=["current_tariff", "arpu_segment"])[["current_tariff", "arpu_segment"]]
    cells = cells.drop_duplicates().itertuples(index=False)
    targets = list(dict_tariff["tariff_plan_code"])
    table = {}
    for tariff, segment in cells:
        for target in targets:
            key = (tariff, segment, target)
            if key in model:
                pct, conv = model[key]
            else:
                pct, conv = world.fallback_predict(tariff, target, segment, dict_tariff, fallback_conversion)
            for channel, spec in CHANNELS.items():
                ratio = pct * min(conv * spec["conversion_multiplier"], 1.0)
                best = table.get((tariff, segment, channel))
                if best is None or ratio > best:
                    table[(tariff, segment, channel)] = ratio
    return table


def upper_bound(world):
    """
    Доказуемый потолок чистого результата (слабая двойственность):
      UB(λ) = Σ top-R max(0, max_канал(v_i,к − λ·цена_к)) + λ·B,  минимум по λ ≥ 0,
    где v_i,к — лучший эффект абонента i на канале к минус цена контакта,
    R — лимит контактов, B — бюджет. Любой допустимый план не может заработать больше.
    """
    profile, dict_tariff = load_inputs()
    table = _ratio_table(world, profile, dict_tariff)
    channels = list(CHANNELS)
    costs = np.array([CHANNELS[c]["cost_per_contact"] for c in channels], dtype=float)
    ratios = np.array([[table.get((t, s, c), 0.0) for c in channels]
                       for t, s in zip(profile["current_tariff"], profile["arpu_segment"])])
    values = ratios * profile["predicted_arpu"].to_numpy()[:, None] - costs[None, :]
    def bound(lam):
        s = np.maximum(0.0, (values - lam * costs[None, :]).max(axis=1))
        if len(s) > MAX_TOTAL_CONTACTS:
            s = np.partition(s, -MAX_TOTAL_CONTACTS)[-MAX_TOTAL_CONTACTS:]
        return s.sum() + lam * TOTAL_BUDGET

    # любая λ ≥ 0 даёт верную границу; ищем самую плотную: грубая сетка, затем уточнение
    coarse = min(np.arange(0.0, 100.0, 0.5), key=bound)
    fine = np.arange(max(0.0, coarse - 0.5), coarse + 0.5, 0.01)
    return float(min(bound(lam) for lam in fine))
