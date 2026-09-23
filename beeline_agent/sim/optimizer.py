"""
Лучший план при известных истинных эффектах — целочисленная оптимизация (MILP, scipy / HiGHS).

Нужен, чтобы понять, сколько денег даёт само устройство плана, если эффекты уже известны:
разрыв «планировщик агента со знанием эффектов» → «потолок» — это недостаток планировщика или
только запас в оценке потолка? Используется в sim/headroom.py (колонка known_milp).

Кампания — ровно то, что разрешает ТЗ: сегмент ARPU × (фильтр трафика) × (фильтр звонков) ×
набор текущих тарифов × целевой тариф × канал. Кандидаты: для каждого сочетания сегмент × фильтры ×
цель × канал берём тарифы с положительной чистой ценностью по убыванию ценности на абонента,
в трёх размерах — до 5 000 (лимит кампании), 2 000 и 700 абонентов (чтобы влезали дорогие каналы),
плюс «все выгодные тарифы» с обрезкой средой до первых 5 000 по ID (ценность считается ровно по ним).
Ограничения: не больше 10 кампаний, 15 000 контактов и 100 000 бюджета; каждая группа абонентов
(тариф × сегмент × трафик × звонки) — не больше чем в одной кампании (повторный контакт стоит денег,
а эффект засчитывается один раз). Пилоты не нужны — эффекты известны.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEGMENTS = ("LOW", "MID", "HIGH")
DATA_SEGMENTS = ("NON_USER", "LITE", "HEAVY")
CALL_SEGMENTS = ("LOW", "MEDIUM", "HIGH")
SIZE_CAPS = (5000, 2000, 700)
ROUNDS = 3                   # решение → добор кампаний из незанятых абонентов → решение ещё раз
ATOM = ["current_tariff", "arpu_segment", "data_segment", "call_segment"]


def lift_table(world, dict_tariff: pd.DataFrame, channels: dict) -> dict:
    """Истинный относительный эффект pct × min(conv × m, 1) для (тариф, сегмент, цель, канал)."""
    im = world.impact_model
    truth = {(f, str(s), t): (p, c) for f, s, t, p, c in zip(
        im["tariff_plan_code_from"], im["arpu_segment"], im["tariff_plan_code_to"],
        im["arpu_change_pct"], im["conversion_rate"])}
    conv_median = float(im["conversion_rate"].median())
    codes = list(dict_tariff["tariff_plan_code"])
    table = {}
    for f in codes:
        for s in SEGMENTS:
            for t in codes:
                if f == t:
                    continue
                p, c = truth.get((f, s, t)) or world.fallback_predict(f, t, s, dict_tariff, conv_median)
                for ch, spec in channels.items():
                    table[(f, s, t, ch)] = float(p) * min(float(c) * spec["conversion_multiplier"], 1.0)
    return table


def _candidates(atoms: pd.DataFrame, customers: pd.DataFrame, lift: dict, targets: list, channels: dict,
                exclude: frozenset = frozenset(), seen: set | None = None) -> list:
    """Кандидаты-кампании; exclude — группы абонентов, уже занятые выбранными кампаниями (добор остатков)."""
    cands, seen = [], (set() if seen is None else seen)
    codes = sorted(customers["current_tariff"].unique())
    code_idx = {c: i for i, c in enumerate(codes)}
    for s in SEGMENTS:
        a_s = atoms[atoms["arpu_segment"] == s]
        for d in (None,) + DATA_SEGMENTS:
            a_d = a_s if d is None else a_s[a_s["data_segment"] == d]
            for k in (None,) + CALL_SEGMENTS:
                sub = a_d if k is None else a_d[a_d["call_segment"] == k]
                if sub.empty:
                    continue
                by_tariff = sub.groupby("current_tariff").agg(n=("n", "sum"), arpu=("arpu", "sum"),
                                                              atoms=("atom_id", list))
                if exclude:     # фильтр кампании не может обойти занятых: тариф с занятой группой — целиком нельзя
                    free = [not (set(a) & exclude) for a in by_tariff["atoms"]]
                    by_tariff = by_tariff[free]
                    if by_tariff.empty:
                        continue
                # абоненты этой группы по возрастанию ID — так среда обрезает кампанию больше 5 000
                cust = customers[(customers["arpu_segment"] == s)
                                 & ((customers["data_segment"] == d) if d is not None else True)
                                 & ((customers["call_segment"] == k) if k is not None else True)]
                cust_tariff = cust["current_tariff"].map(code_idx).values
                cust_arpu = cust["predicted_arpu"].values
                for t in targets:
                    for ch, spec in channels.items():
                        cost = spec["cost_per_contact"]
                        l_arr = np.array([lift.get((f, s, t, ch), 0.0) for f in by_tariff.index])
                        value = l_arr * by_tariff["arpu"].values - cost * by_tariff["n"].values
                        good = np.flatnonzero(value > 0)
                        if len(good) == 0:
                            continue
                        good = good[np.argsort(-(value[good] / by_tariff["n"].values[good]))]
                        if int(by_tariff["n"].values[good].sum()) > SIZE_CAPS[0]:
                            # все выгодные тарифы, обрезка средой до первых 5 000 по ID: считаем ровно эту выборку
                            lift_vec = np.zeros(len(codes))
                            for j in good:
                                lift_vec[code_idx[by_tariff.index[j]]] = l_arr[j]
                            chosen = np.flatnonzero(lift_vec[cust_tariff] > 0)[:SIZE_CAPS[0]]
                            atom_ids = tuple(sorted(a for j in good for a in by_tariff["atoms"].values[j]))
                            key = (t, ch, atom_ids, "cut")
                            if key not in seen:
                                seen.add(key)
                                v = float((lift_vec[cust_tariff[chosen]] * cust_arpu[chosen]).sum() - cost * len(chosen))
                                cands.append({"segment": s, "data": d, "call": k, "target": t, "channel": ch,
                                              "tariffs": tuple(sorted(by_tariff.index[j] for j in good)),
                                              "n": len(chosen), "cost": cost * len(chosen), "value": v,
                                              "atoms": atom_ids})
                        for cap in SIZE_CAPS:
                            pick, size = [], 0
                            for j in good:                       # по убыванию ценности на абонента, что влезает
                                n_j = int(by_tariff["n"].values[j])
                                if size + n_j <= cap:
                                    pick.append(j)
                                    size += n_j
                            if not pick:
                                continue
                            atom_ids = tuple(sorted(a for j in pick for a in by_tariff["atoms"].values[j]))
                            key = (t, ch, atom_ids)
                            if key in seen:
                                continue
                            seen.add(key)
                            cands.append({"segment": s, "data": d, "call": k, "target": t, "channel": ch,
                                          "tariffs": tuple(sorted(by_tariff.index[j] for j in pick)),
                                          "n": size, "cost": cost * size, "value": float(value[pick].sum()),
                                          "atoms": atom_ids})
    return cands


def optimal_plan(world, profile: pd.DataFrame, dict_tariff: pd.DataFrame, channels: dict,
                 budget: float = 100_000, contacts: int = 15_000, max_campaigns: int = 10,
                 time_limit: float = 20.0) -> tuple[list, dict]:
    """(план в формате ТЗ, сводка решения) — лучший план при известных эффектах мира."""
    return plan_from_lift(lift_table(world, dict_tariff, channels), profile, dict_tariff, channels,
                          budget, contacts, max_campaigns, time_limit)


def plan_from_lift(lift: dict, profile: pd.DataFrame, dict_tariff: pd.DataFrame, channels: dict,
                   budget: float = 100_000, contacts: int = 15_000, max_campaigns: int = 10,
                   time_limit: float = 20.0) -> tuple[list, dict]:
    """
    Лучший план для заданной таблицы эффектов lift[(тариф, сегмент, цель, канал)] = pct · min(conv · m, 1).
    Агент может передать сюда свои оценки (например, осторожные: среднее − k·sd после пилотов) —
    оптимизатор сам выберет фильтры, наборы тарифов и каналы в пределах лимитов ТЗ.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csr_matrix

    prof = profile.dropna(subset=ATOM)
    atoms = prof.groupby(ATOM)["predicted_arpu"].agg(n="size", arpu="sum").reset_index()
    atoms["atom_id"] = np.arange(len(atoms))
    customers = prof.sort_values("ID_NUMBER").merge(atoms[ATOM + ["atom_id"]], on=ATOM, how="left")
    customers = customers[ATOM + ["predicted_arpu", "atom_id"]]
    targets, seen = list(dict_tariff["tariff_plan_code"]), set()
    cands = _candidates(atoms, customers, lift, targets, channels, seen=seen)
    if not cands:
        return [], {"candidates": 0}
    best_value, chosen = -np.inf, []
    for _ in range(ROUNDS):
        rows = [a for i, c in enumerate(cands) for a in c["atoms"]]
        cols = [i for i, c in enumerate(cands) for _ in c["atoms"]]
        cover = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(atoms), len(cands)))
        n = np.array([c["n"] for c in cands], dtype=float)
        cost = np.array([c["cost"] for c in cands], dtype=float)
        value = np.array([c["value"] for c in cands], dtype=float)
        limits = csr_matrix(np.vstack([n, cost, np.ones(len(cands))]))
        res = milp(-value, integrality=np.ones(len(cands)), bounds=Bounds(0, 1),
                   constraints=[LinearConstraint(cover, 0, 1),
                                LinearConstraint(limits, 0, [contacts, budget, max_campaigns])],
                   options={"time_limit": time_limit, "mip_rel_gap": 1e-3})
        if res.x is None:
            break
        pick = sorted(np.flatnonzero(res.x > 0.5), key=lambda i: -value[i])
        if value[pick].sum() <= best_value + 1.0:
            break
        best_value, chosen = float(value[pick].sum()), [cands[i] for i in pick]
        # добор: кампании из абонентов, не занятых выбранными (так находятся «остатки» вроде MID → tariff_10)
        used = frozenset(a for c in chosen for a in c["atoms"])
        extra = _candidates(atoms, customers, lift, targets, channels, exclude=used, seen=seen)
        if not extra:
            break
        cands = cands + extra
    if not chosen:
        return [], {"candidates": len(cands), "status": "нет решения"}
    plan = []
    for c in chosen:
        camp = {"campaign_name": f"opt_{c['segment']}_{c['target']}_{c['channel']}_{len(plan) + 1}",
                "filter_arpu_segment": c["segment"], "filter_current_tariff": ";".join(c["tariffs"]),
                "target_tariff": c["target"], "channel": c["channel"]}
        if c["data"] is not None:
            camp["filter_data_segment"] = c["data"]
        if c["call"] is not None:
            camp["filter_call_segment"] = c["call"]
        plan.append(camp)
    info = {"candidates": len(cands), "value": best_value, "contacts": float(sum(c["n"] for c in chosen)),
            "budget": float(sum(c["cost"] for c in chosen)),
            "channels": ",".join(sorted({c["channel"] for c in chosen}))}
    return plan, info


class KnownOptimal:
    """Стратегия для тренажёра: лучший план при известных эффектах (без пилотов)."""

    def __init__(self, world, time_limit: float = 20.0):
        self.world, self.time_limit, self.info = world, time_limit, {}

    def act(self, env):
        plan, self.info = optimal_plan(self.world, env.customer_profile, env.tariffs, env.channels,
                                       env.remaining_budget, env.remaining_contacts, time_limit=self.time_limit)
        return plan
