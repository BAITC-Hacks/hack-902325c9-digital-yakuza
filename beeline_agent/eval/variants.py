"""
Варианты агента для экспериментов «одно изменение за раз».

  agent_v0        — замороженный v0 (eval/baselines/agent_v0.py): старые единицы, q как в v0
  agent_units     — шаг 3а: код с новыми единицами, q ровно как в v0, q_se по договорённой формуле
  agent           — текущий agent.py как есть (q из таблицы Тимура в режиме PRIOR_MODE)
  agent_hier      — шаг 3б: код с новыми единицами, q иерархический (PRIOR_SCORE: уровень + контраст)

Подмена таблиц делается только здесь, в тренажёре; agent.py её не видит.
"""

import importlib.util
import math
from pathlib import Path

import pandas as pd

import agent as agent_module
from eval.prior_tables import read_tables

BASE = Path(__file__).resolve().parent / "baselines"
FORMAT = ("q", "q_se", "n_obs")
CONV_ALPHA = 10.0            # как в prior/build_prior.py


def _load_frozen_v0():
    spec = importlib.util.spec_from_file_location("agent_v0", BASE / "agent_v0.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Agent


def units_only_table():
    """q — как в v0 (вшитые значения), q_se = √((conv·pct_eb_sd)² + (pct_eb·conv_se)²) по таблице v0."""
    q_v0 = read_tables(BASE / "agent_v0.py")["PRIOR"]
    t = pd.read_csv(BASE / "prior_table_v0.csv")
    totals = t.groupby(["tariff_from", "arpu_segment"])["n_obs"].transform("sum")
    conv_se = ((t["conversion"] * (1 - t["conversion"])) / (totals + CONV_ALPHA + 1)).pow(0.5)
    q_se = ((t["conversion"] * t["pct_eb_sd"]) ** 2 + (t["pct_eb"] * conv_se) ** 2).pow(0.5)
    table = {}
    for f, s, to, se, n in zip(t["tariff_from"], t["arpu_segment"], t["tariff_to"], q_se, t["n_obs"]):
        key = (f, s, to)
        table[key] = (q_v0[key][0], round(float(se), 5), int(n))
    assert set(table) == set(q_v0), "набор ключей не совпал с v0"
    return table


def hier_table():
    """q = q_level + q_contrast, q_se из PRIOR_SCORE; ключи — те же, что у PRIOR."""
    score = agent_module.PRIOR_SCORE
    return {k: (round(score[k][0] + score[k][1], 5), score[k][2], n) for k, (_, _, n) in agent_module.PRIOR.items()}


def with_table(table):
    """Фабрика агента, который на время act() видит другую таблицу PRIOR (формат тот же)."""
    class TableAgent(agent_module.Agent):
        def act(self, env):
            saved = agent_module.PRIOR, agent_module.PRIOR_FORMAT
            agent_module.PRIOR, agent_module.PRIOR_FORMAT = table, FORMAT
            try:
                return super().act(env)
            finally:
                agent_module.PRIOR, agent_module.PRIOR_FORMAT = saved
    return lambda: TableAgent(verbose=False)


def build_variants():
    v0 = _load_frozen_v0()
    return {
        "agent_v0": lambda: v0(verbose=False),
        "agent_units": with_table(units_only_table()),
        "agent_hier": with_table(hier_table()),
    }


def check_units_table():
    """Проверка шага 3а: ключи и q совпадают с v0, q_se в разумных пределах."""
    table, v0 = units_only_table(), read_tables(BASE / "agent_v0.py")["PRIOR"]
    same_q = all(abs(table[k][0] - v0[k][0]) < 1e-12 for k in v0)
    se = sorted(r[1] for r in table.values())
    return same_q, se[len(se) // 2], max(se), all(math.isfinite(x) and x >= 0 for x in se)


if __name__ == "__main__":
    same_q, med, mx, ok = check_units_table()
    print(f"шаг 3а: q совпадает с v0: {same_q}; q_se медиана {med:.4f}, максимум {mx:.4f}, корректны: {ok}")
