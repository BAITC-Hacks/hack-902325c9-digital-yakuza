"""
Парное сравнение двух стратегий на одних и тех же мирах и seed.

    python -m eval.compare step3 agent_v0 agent_units          # из eval/results/step3.csv
    python -m eval.compare step3 agent_v0 agent_units --by-scenario

Для каждого мира берётся разница «B − A»: так случайность мира не маскирует эффект изменения.
Решение по правилу: оставляем B, если медиана выросла И худший случай не стал хуже.
"""

import argparse
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"


def paired(df, a, b):
    wide = df[df["strategy"].isin([a, b])].pivot_table(index=["scenario", "world"], columns="strategy", values="net")
    wide = wide.dropna()
    wide["diff"] = wide[b] - wide[a]
    return wide


def summary(wide, a, b):
    return {
        "миров": len(wide),
        f"медиана {a}": wide[a].median(), f"медиана {b}": wide[b].median(),
        f"худший {a}": wide[a].min(), f"худший {b}": wide[b].min(),
        f"10% {a}": wide[a].quantile(0.1), f"10% {b}": wide[b].quantile(0.1),
        "медиана разницы": wide["diff"].median(),
        f"{b} лучше, %": 100 * (wide["diff"] > 0).mean(),
        f"в минусе {a}, %": 100 * (wide[a] < 0).mean(), f"в минусе {b}, %": 100 * (wide[b] < 0).mean(),
    }


def verdict(s, a, b):
    better_median = s[f"медиана {b}"] > s[f"медиана {a}"]
    not_worse_tail = s[f"худший {b}"] >= s[f"худший {a}"] and s[f"в минусе {b}, %"] <= s[f"в минусе {a}, %"]
    if better_median and not_worse_tail:
        return f"ОСТАВЛЯЕМ {b}: медиана выше, хвост не хуже"
    if better_median:
        return f"СПОРНО: медиана {b} выше, но хвост хуже — смотреть по сценариям"
    return f"НЕ ОСТАВЛЯЕМ {b}: медиана не выросла"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--by-scenario", action="store_true")
    args = ap.parse_args()
    df = pd.read_csv(RESULTS / f"{args.label}.csv")
    wide = paired(df, args.a, args.b)
    s = summary(wide, args.a, args.b)
    for k, v in s.items():
        print(f"  {k:24} {v:>14,.0f}" if abs(v) >= 100 else f"  {k:24} {v:>14.1f}")
    print(verdict(s, args.a, args.b))
    if args.by_scenario:
        rows = {sc: summary(g, args.a, args.b) for sc, g in wide.groupby(level="scenario")}
        cols = [f"медиана {args.a}", f"медиана {args.b}", "медиана разницы", f"{args.b} лучше, %",
                f"худший {args.a}", f"худший {args.b}"]
        table = pd.DataFrame(rows).T[cols]
        print(table.to_string(float_format=lambda v: f"{v:,.0f}"))


if __name__ == "__main__":
    main()
