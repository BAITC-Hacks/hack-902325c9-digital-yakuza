"""
Сколько ещё можно заработать и где: разрыв между агентом и потолком, разложенный по причинам.

    cd beeline_agent
    python -m sim.headroom --worlds 5 --scenarios all --jobs 4

В каждом мире считаются (подсчёт — eval.core.run_strategy, тем же кодом, что у организаторов):
  agent          — текущий агент, как на судействе;
  history_only   — план только по истории, без пилотов (eval.strategies);
  known          — планировщик агента (_build_plan: кампания = сегмент × цель, SMS или push, ≤ 10 кампаний),
                   но истинные эффекты известны и пилоты не нужны. Это предел для любых улучшений оценки
                   и разведки при нынешнем устройстве плана;
  known_channels — то же, но канал каждой кампании выбирается из всех четырёх по истинной ценности
                   с учётом бюджета: сколько теряем, пользуясь только SMS и push;
  ceiling        — eval.core.upper_bound, доказуемый потолок (без лимита 10 кампаний и стоимости пилотов).

Разрыв ceiling − agent = (known − agent)                  «незнание»: оценка + разведка (пилоты)
                       + (known_channels − known)         «каналы»
                       + (ceiling − known_channels)       «устройство плана и запас оценки потолка».
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from sim.worlds import ROOT, list_scenarios, make_world

STRATEGY_NAMES = ["agent", "history_only", "known", "known_channels"]


def _true_table(world, dict_tariff) -> dict:
    """Истинный q = pct × conv для каждой тройки мира в формате таблицы агента (q, q_se, n_obs)."""
    im = world.impact_model
    table = {(f, str(s), t): (float(p * c), 0.0, 1) for f, s, t, p, c in zip(
        im["tariff_plan_code_from"], im["arpu_segment"], im["tariff_plan_code_to"],
        im["arpu_change_pct"], im["conversion_rate"])}
    conv_median = float(im["conversion_rate"].median())
    codes = list(dict_tariff["tariff_plan_code"])
    for f in codes:
        for s in ("LOW", "MID", "HIGH"):
            for t in codes:
                if f != t and (f, s, t) not in table:
                    p, c = world.fallback_predict(f, t, s, dict_tariff, conv_median)
                    table[(f, s, t)] = (float(p * c), 0.0, 1)
    return table


def _pick_channels(plan: list, candidates: dict, env) -> list:
    """Каналы кампаний по истинной ценности: жадно повышаем канал там, где прирост на рубль бюджета больше."""
    from agent import Agent
    channels = env.channels
    info = []
    for camp in plan:
        c = candidates[(camp["filter_arpu_segment"], camp["target_tariff"])]
        size, arpu = Agent._audience(env.customer_profile, camp["filter_arpu_segment"],
                                     tuple(camp["filter_current_tariff"].split(";")))
        value = {ch: c.mu * spec["conversion_multiplier"] * arpu - spec["cost_per_contact"] * size
                 for ch, spec in channels.items()}
        cost = {ch: spec["cost_per_contact"] * size for ch, spec in channels.items()}
        info.append((value, cost))
    choice = ["push"] * len(plan)
    budget = env.remaining_budget
    while True:
        best = None
        for i, (value, cost) in enumerate(info):
            cur = choice[i]
            for ch in channels:
                dv, dc = value[ch] - value[cur], cost[ch] - cost[cur]
                if dv <= 0 or dc > budget:
                    continue
                ratio = dv / dc if dc > 0 else np.inf
                if best is None or ratio > best[0]:
                    best = (ratio, i, ch, dc)
        if best is None:
            break
        _, i, ch, dc = best
        budget -= dc
        choice[i] = ch
    return [{**camp, "channel": ch} for camp, ch in zip(plan, choice)]


class KnownEffects:
    """Планировщик агента с истинными эффектами вместо оценок, без пилотов."""

    def __init__(self, world, all_channels: bool = False):
        self.world, self.all_channels = world, all_channels

    def act(self, env):
        import agent as A
        saved = (A.PRIOR, A.PRIOR_SHRINK, A.TRANSFER_SD)
        try:
            A.PRIOR, A.PRIOR_SHRINK, A.TRANSFER_SD = _true_table(self.world, env.tariffs), 1.0, 0.0
            ag = A.Agent(verbose=False)
            ag.trace, ag._t0 = [], time.monotonic()
            candidates = ag._make_candidates(env.customer_profile, env)
            plan = ag._build_plan(candidates, env, require_pilot=False)
            if self.all_channels and plan:
                by_key = {(c.segment, c.target): c for c in candidates}
                plan = _pick_channels(plan, by_key, env)
            return plan
        finally:
            A.PRIOR, A.PRIOR_SHRINK, A.TRANSFER_SD = saved


def _one_world(task: tuple) -> dict:
    from eval.core import run_strategy, upper_bound
    from eval.strategies import STRATEGIES

    scenario, seed = task
    world = make_world(seed, scenario)
    makers = {"agent": STRATEGIES["agent"], "history_only": STRATEGIES["history_only"],
              "known": lambda: KnownEffects(world), "known_channels": lambda: KnownEffects(world, all_channels=True)}
    row = {"scenario": scenario, "seed": seed, "world": world.name, "ceiling": upper_bound(world)}
    for name, make in makers.items():
        res = run_strategy(world, make, env_seed=seed)
        row[name] = res["net_arpu_gain"]
        row[f"{name}_campaigns"] = len(res["plan"])
        if name == "known_channels":
            row["known_channels_mix"] = ",".join(sorted({c["channel"] for c in res["plan"]}))
    return row


def main():
    ap = argparse.ArgumentParser(description="Разрыв агента с потолком по причинам")
    ap.add_argument("--worlds", type=int, default=5)
    ap.add_argument("--scenarios", default="all")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "sim" / "reports" / "headroom.csv"))
    args = ap.parse_args()
    scenarios = list_scenarios() if args.scenarios == "all" else args.scenarios.split(",")
    tasks = [(sc, s) for sc in scenarios for s in range(args.worlds)]
    t0 = time.perf_counter()
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            rows = list(pool.map(_one_world, tasks))
    else:
        rows = [_one_world(t) for t in tasks]
    res = pd.DataFrame(rows)
    ROOT.joinpath("sim", "reports").mkdir(exist_ok=True)
    res.to_csv(args.out, index=False)

    cols = ["history_only", "agent", "known", "known_channels", "ceiling"]
    med = res.groupby("scenario", sort=False)[cols].median() / 1e6
    share = res[cols].div(res["ceiling"], axis=0).groupby(res["scenario"], sort=False).median() * 100
    pd.set_option("display.width", 200)
    print(f"{len(res)} миров, {time.perf_counter() - t0:.0f} с\n\nМедиана, млн:")
    print(med.round(2).to_string())
    print("\nМедиана доли потолка, %:")
    print(share.round(0).to_string())
    tot = res[cols].sum()
    gaps = {"незнание (known − agent)": tot["known"] - tot["agent"],
            "каналы (known_channels − known)": tot["known_channels"] - tot["known"],
            "план и запас потолка (ceiling − known_channels)": tot["ceiling"] - tot["known_channels"]}
    whole = tot["ceiling"] - tot["agent"]
    print(f"\nСумма по всем мирам: агент {tot['agent'] / 1e6:,.1f} млн из потолка {tot['ceiling'] / 1e6:,.1f} млн. "
          f"Разрыв {whole / 1e6:,.1f} млн:")
    for k, v in gaps.items():
        print(f"  {k}: {v / 1e6:,.1f} млн ({v / whole:.0%})")
    print(f"\nВсе прогоны: {args.out}")


if __name__ == "__main__":
    main()
