"""
Проверка гипотез стратегии на фиксированном наборе миров — с учётом их сочетаний.

    cd beeline_agent
    python -m sim.hypotheses --worlds 40 --scenarios random,resample \\
        --grid "RISK_K=0,0.84" "PRIOR_SHRINK=0.5,0.8"

Каждая комбинация параметров агента прогоняется на ОДНИХ И ТЕХ ЖЕ мирах (make_world с теми же seed),
поэтому разница между вариантами — эффект гипотезы, а не удачи. Для двух двухуровневых параметров
печатается взаимодействие: даёт ли пара больше, чем сумма эффектов по отдельности.

Параметры — константы модуля agent.py (файл не меняется, значения подставляются в памяти):
RISK_K, PRIOR_SHRINK, PRIOR_MODEL_SD, MAX_PILOTS_PER_CANDIDATE, PILOT_CHANNEL, PILOT_N_LARGE,
PILOT_N_SMALL, SMALL_SEGMENT; плюс PRIOR_SET=hist|hist+unseen (подмешать PRIOR_UNSEEN в кандидаты).
Это вспомогательный инструмент; основной прогонщик — у Димаша.
"""
from __future__ import annotations

import argparse
import itertools
import time

import numpy as np
import pandas as pd

from sim.worlds import ROOT, make_world

COLS = ["filter_arpu_segment", "filter_data_segment", "filter_call_segment", "filter_current_tariff", "explicit_ids"]


def _parse_value(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def _apply(agent_mod, params: dict, base_prior: dict) -> None:
    for k, v in params.items():
        if k == "PRIOR_SET":
            prior = dict(base_prior)
            if v == "hist+unseen":
                prior = {**getattr(agent_mod, "PRIOR_UNSEEN", {}), **prior}
            agent_mod.PRIOR = prior
            continue
        setattr(agent_mod, k, v)
        if k == "RISK_K":                       # значение по умолчанию в lcb() связывается при импорте
            agent_mod.Candidate.lcb.__defaults__ = (v,)


def run(grid: dict, n_worlds: int, scenarios: list) -> pd.DataFrame:
    from environment import make_environment
    from mock_environment import CHANNELS, MAX_TOTAL_CONTACTS, TOTAL_BUDGET
    from scoring_core import MAX_CAMPAIGNS, sanitize_campaigns, score_campaigns
    import agent as agent_mod

    defaults = {k: getattr(agent_mod, k) for k in grid if hasattr(agent_mod, k)}
    base_prior = dict(agent_mod.PRIOR)
    profile = pd.read_csv(ROOT / "customer_profile.csv")
    tariffs = pd.read_csv(ROOT / "data" / "dict_tariff.csv")
    worlds = [make_world(s, sc) for sc in scenarios for s in range(n_worlds)]
    rows = []
    for combo in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), combo))
        _apply(agent_mod, params, base_prior)
        t0 = time.perf_counter()
        for w in worlds:
            env, internals = make_environment(profile, w.impact_model, tariffs, CHANNELS, TOTAL_BUDGET,
                                              MAX_TOTAL_CONTACTS, w.fallback_predict, seed=w.seed)
            final = sanitize_campaigns(agent_mod.Agent(verbose=False).act(env), env.tariffs)[:MAX_CAMPAIGNS]
            pilots = internals.executed_pilot_campaigns()
            camps = pd.DataFrame(pilots + final)
            for c in COLS:
                if c not in camps.columns:
                    camps[c] = None
            res = score_campaigns(camps, env.customer_profile, w.impact_model, env.tariffs,
                                  env.customer_profile["predicted_arpu"].sum(), w.fallback_predict)
            fin = res["campaigns_detail"][len(pilots):]
            rows.append({**params, "world": w.name, "net": res["net_arpu_gain"],
                         "loss_campaigns": sum(c["gross_lift"] - c["cost"] < 0 for c in fin), "campaigns": len(fin)})
        print(f"  {params}: {time.perf_counter() - t0:.0f} с", flush=True)
    _apply(agent_mod, {k: v for k, v in defaults.items()}, base_prior)
    agent_mod.PRIOR = base_prior
    return pd.DataFrame(rows)


def summarize(res: pd.DataFrame, keys: list) -> pd.DataFrame:
    g = res.groupby(keys)
    return pd.DataFrame({"mean": g["net"].mean(), "median": g["net"].median(),
                         "p10": g["net"].quantile(0.10), "P(net<0)": g["net"].apply(lambda x: (x < 0).mean()),
                         "loss_campaigns": g["loss_campaigns"].sum(), "campaigns": g["campaigns"].sum()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=30, help="миров на сценарий")
    parser.add_argument("--scenarios", default="random", help="через запятую")
    parser.add_argument("--grid", nargs="+", required=True, help='например "RISK_K=0,0.84" "PRIOR_SHRINK=0.5,0.8"')
    args = parser.parse_args()
    grid = {}
    for item in args.grid:
        k, vals = item.split("=", 1)
        grid[k.strip()] = [_parse_value(v.strip()) for v in vals.split(",")]
    res = run(grid, args.worlds, args.scenarios.split(","))
    out = ROOT / "sim" / "reports"
    out.mkdir(exist_ok=True)
    res.to_csv(out / "hypotheses_runs.csv", index=False)
    keys = list(grid)
    pd.set_option("display.width", 200)
    print(summarize(res, keys).round(3).to_string(float_format=lambda x: f"{x:,.3f}" if abs(x) < 10 else f"{x:,.0f}"))
    two = [k for k in keys if len(grid[k]) == 2]
    if len(two) >= 2:
        a, b = two[:2]
        m = res.groupby([a, b])["net"].mean()
        (a0, a1), (b0, b1) = grid[a], grid[b]
        inter = (m[(a1, b1)] - m[(a0, b1)]) - (m[(a1, b0)] - m[(a0, b0)])
        print(f"\nЭффект {a}: {m[(a1, b0)] - m[(a0, b0)]:+,.0f} (при {b}={b0}), {m[(a1, b1)] - m[(a0, b1)]:+,.0f} (при {b}={b1}); "
              f"взаимодействие {a}×{b}: {inter:+,.0f}")
    print(f"\nВсе прогоны: {out / 'hypotheses_runs.csv'}")


if __name__ == "__main__":
    main()
