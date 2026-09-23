"""
Прогон стратегий по множеству миров и сводная таблица.

    cd beeline_agent
    python -m eval.run                         # 30 тренировочных миров, сценарий random
    python -m eval.run --worlds 100 --scenario all
    python -m eval.run --split control         # контрольные миры: смотреть один раз в конце

Миры делятся на тренировочные (seed 0…) и контрольные (seed 10 000…), чтобы не подогнать
агента под наш же симулятор. Результат каждого прогона пишется в eval/results/<метка>.csv.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from eval.core import run_strategy, upper_bound
from eval.simple_worlds import SCENARIOS, make_world
from eval.strategies import STRATEGIES

CONTROL_OFFSET = 10_000
RESULTS = Path(__file__).resolve().parent / "results"


def world_seeds(split, n):
    start = CONTROL_OFFSET if split == "control" else 0
    return range(start, start + n)


def run(worlds, scenarios, split, strategies, env_seed_base=0):
    rows = []
    for scenario in scenarios:
        for seed in world_seeds(split, worlds):
            world = make_world(seed, scenario)
            ceiling = upper_bound(world)
            for name in strategies:
                res = run_strategy(world, STRATEGIES[name], env_seed=env_seed_base + seed)
                rows.append({"scenario": scenario, "world": world.name, "strategy": name,
                             "net": res["net_arpu_gain"], "ceiling": ceiling,
                             "share_of_ceiling": res["net_arpu_gain"] / ceiling if ceiling > 0 else np.nan,
                             "cost": res["total_cost"], "risk_pct": res["risk_score_pct"],
                             "pilots": res["n_pilots"], "campaigns": len(res["plan"]),
                             "seconds": res["seconds"], "error": res["error"]})
    return pd.DataFrame(rows)


def summarize(df):
    g = df.groupby(["scenario", "strategy"], sort=False)
    return pd.DataFrame({
        "миров": g.size(),
        "медиана": g["net"].median(),
        "худший": g["net"].min(),
        "10%": g["net"].quantile(0.10),
        "в плюс, %": g["net"].apply(lambda s: 100 * (s > 0).mean()),
        "от потолка, %": g["share_of_ceiling"].median() * 100,
        "пилотов": g["pilots"].median(),
        "сек": g["seconds"].max(),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=30)
    ap.add_argument("--scenario", default="random", help=f"один из {SCENARIOS} или all")
    ap.add_argument("--split", choices=["train", "control"], default="train")
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    scenarios = SCENARIOS if args.scenario == "all" else (args.scenario,)
    t0 = time.monotonic()
    df = run(args.worlds, scenarios, args.split, args.strategies.split(","))
    RESULTS.mkdir(exist_ok=True)
    label = args.label or time.strftime("%H%M%S")
    df.to_csv(RESULTS / f"{label}.csv", index=False)

    pd.set_option("display.width", 200)
    table = summarize(df)
    fmt = {c: "{:,.0f}".format for c in ["медиана", "худший", "10%"]}
    fmt.update({c: "{:.0f}".format for c in ["в плюс, %", "от потолка, %", "пилотов"]})
    fmt["сек"] = "{:.1f}".format
    print(table.to_string(formatters=fmt))
    errors = df["error"].dropna()
    if len(errors):
        print(f"\n[!] падений стратегий: {len(errors)}; пример: {errors.iloc[0]}")
    print(f"\nсохранено: eval/results/{label}.csv   время: {time.monotonic() - t0:.0f} с")


if __name__ == "__main__":
    main()
