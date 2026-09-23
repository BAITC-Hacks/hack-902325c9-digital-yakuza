"""
Проверка гипотез стратегии на фиксированном наборе миров — с учётом их сочетаний.

    cd beeline_agent
    python -m sim.hypotheses --worlds 15 --scenarios all \\
        --grid "TRANSFER_SD=0.10,0.05" "PRIOR_SET=hist,hist+unseen"

Прогон идёт через тренажёр Димаша: eval.core.run_strategy (подсчёт тем же кодом, что у организаторов)
и eval.core.upper_bound (потолок). Seed миров — как в eval.run: --split train → 0..N-1,
--split control → 10000..; seed среды = seed мира. Каждая комбинация параметров агента идёт на ОДНИХ
И ТЕХ ЖЕ мирах, поэтому разница между вариантами — эффект гипотезы, а не удачи.

Первая комбинация сетки — база: для остальных печатается парное сравнение (средняя разница ± ст. ошибка,
в какой доле миров лучше/хуже), с --by-scenario — ещё и по сценариям. Для двух двухуровневых параметров
печатается взаимодействие: даёт ли пара больше, чем сумма эффектов по отдельности.

Параметры — константы верхнего уровня agent.py (файл не меняется, значения подставляются в памяти):
RISK_K, TRANSFER_SD, PRIOR_SHRINK, MAX_PILOTS_PER_CANDIDATE, PILOT_CHANNEL, PILOT_N_LARGE, PILOT_N_SMALL,
SMALL_SEGMENT и т. д.; плюс PRIOR_SET=hist|hist+unseen (подмешать PRIOR_UNSEEN в кандидаты).
Основной прогонщик — eval.run; этот скрипт — для сеток и сочетаний гипотез.
"""
from __future__ import annotations

import argparse
import itertools
import time

import numpy as np
import pandas as pd

from sim.worlds import ROOT, list_scenarios, make_world

CONTROL_OFFSET = 10_000      # как в eval/run.py: контрольные миры не пересекаются с тренировочными


def world_seeds(split: str, n: int) -> range:
    start = CONTROL_OFFSET if split == "control" else 0
    return range(start, start + n)


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
            if v not in ("hist", "hist+unseen"):
                raise ValueError(f"PRIOR_SET={v!r}: ожидается hist или hist+unseen")
            prior = dict(base_prior)
            if v == "hist+unseen":
                prior = {**getattr(agent_mod, "PRIOR_UNSEEN", {}), **prior}
            agent_mod.PRIOR = prior
            continue
        if not hasattr(agent_mod, k):
            raise ValueError(f"в agent.py нет константы {k}")
        setattr(agent_mod, k, v)
        if k == "RISK_K":                       # значение по умолчанию в lcb() связывается при импорте
            agent_mod.Candidate.lcb.__defaults__ = (v,)


def run(grid: dict, n_worlds: int, scenarios: list, split: str = "train", ceiling: bool = True) -> pd.DataFrame:
    import agent as agent_mod
    from eval.core import run_strategy, upper_bound

    saved = {k: getattr(agent_mod, k) for k in grid if k != "PRIOR_SET"}
    saved_lcb = agent_mod.Candidate.lcb.__defaults__
    base_prior = dict(agent_mod.PRIOR)
    worlds = [make_world(s, sc) for sc in scenarios for s in world_seeds(split, n_worlds)]
    ceilings = {w.name: upper_bound(w) for w in worlds} if ceiling else {}
    rows = []
    try:
        for combo in itertools.product(*grid.values()):
            params = dict(zip(grid.keys(), combo))
            _apply(agent_mod, params, base_prior)
            t0 = time.perf_counter()
            for w in worlds:
                res = run_strategy(w, lambda: agent_mod.Agent(verbose=False), env_seed=w.seed)
                final = res.get("campaigns_detail", [])[res["n_pilots"]:]
                ub = ceilings.get(w.name, np.nan)
                rows.append({**params, "scenario": w.params["scenario"], "world": w.name,
                             "net": res["net_arpu_gain"],
                             "share_of_ceiling": res["net_arpu_gain"] / ub if ub > 0 else np.nan,
                             "loss_campaigns": sum(c["gross_lift"] - c["cost"] < 0 for c in final),
                             "campaigns": len(res["plan"]), "pilots": res["n_pilots"], "error": res["error"]})
            print(f"  {params}: {len(worlds)} миров, {time.perf_counter() - t0:.0f} с", flush=True)
    finally:
        for k, v in saved.items():
            setattr(agent_mod, k, v)
        agent_mod.Candidate.lcb.__defaults__ = saved_lcb
        agent_mod.PRIOR = base_prior
    return pd.DataFrame(rows)


def summarize(res: pd.DataFrame, keys: list) -> pd.DataFrame:
    g = res.groupby(keys, sort=False)
    return pd.DataFrame({"миров": g.size(), "среднее": g["net"].mean(), "медиана": g["net"].median(),
                         "10%": g["net"].quantile(0.10), "худший": g["net"].min(),
                         "в минусе, %": g["net"].apply(lambda x: 100 * (x < 0).mean()),
                         "от потолка, %": g["share_of_ceiling"].median() * 100,
                         "убыточных кампаний": g["loss_campaigns"].sum(), "кампаний": g["campaigns"].sum()})


def paired(res: pd.DataFrame, keys: list, by_scenario: bool = False) -> tuple[str, pd.DataFrame]:
    """Каждый вариант против первого (база) на тех же мирах."""
    res = res.assign(variant=res[keys].astype(str).agg(" ".join, axis=1))
    order = list(dict.fromkeys(res["variant"]))
    piv = res.pivot_table(index=["scenario", "world"], columns="variant", values="net")[order]
    base, rows = order[0], []
    for v in order[1:]:
        d = piv[v] - piv[base]
        groups = [("все", d)] + (list(d.groupby(level="scenario")) if by_scenario else [])
        for scen, x in groups:
            rows.append({"вариант": v, "сценарий": scen, "миров": len(x), "разница, среднее": x.mean(),
                         "± ст. ошибка": x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan,
                         "лучше, %": 100 * (x > 1).mean(), "хуже, %": 100 * (x < -1).mean()})
    return base, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Сетка гипотез агента на одних и тех же мирах (через eval.core)")
    ap.add_argument("--worlds", type=int, default=15, help="миров на сценарий (как в журнале eval: 15)")
    ap.add_argument("--scenarios", default="random", help="через запятую или all")
    ap.add_argument("--split", choices=["train", "control"], default="train")
    ap.add_argument("--grid", nargs="+", required=True, help='например "RISK_K=1.0,0.84" "TRANSFER_SD=0.10,0.05"')
    ap.add_argument("--by-scenario", action="store_true", help="парное сравнение ещё и по сценариям")
    ap.add_argument("--no-ceiling", action="store_true", help="не считать потолок (быстрее)")
    args = ap.parse_args()

    grid = {}
    for item in args.grid:
        k, vals = item.split("=", 1)
        grid[k.strip()] = [_parse_value(v.strip()) for v in vals.split(",")]
    scenarios = list_scenarios() if args.scenarios == "all" else args.scenarios.split(",")
    res = run(grid, args.worlds, scenarios, args.split, ceiling=not args.no_ceiling)

    out = ROOT / "sim" / "reports"
    out.mkdir(exist_ok=True)
    res.to_csv(out / "hypotheses_runs.csv", index=False)
    keys = list(grid)
    pd.set_option("display.width", 220)
    fmt = lambda x: f"{x:,.1f}" if abs(x) < 1000 else f"{x:,.0f}"  # noqa: E731
    print("\n" + summarize(res, keys).to_string(float_format=fmt))
    if len(res.groupby(keys)) > 1:
        base, pr = paired(res, keys, args.by_scenario)
        print(f"\nПарно против базы ({base}):\n" + pr.to_string(index=False, float_format=fmt))
    errors = res["error"].notna().sum()
    if errors:
        print(f"\nВНИМАНИЕ: агент упал в {errors} прогонах (см. столбец error)")
    two = [k for k in keys if len(grid[k]) == 2]
    if len(two) >= 2:
        a, b = two[:2]
        m = res.groupby([a, b])["net"].mean()
        (a0, a1), (b0, b1) = grid[a], grid[b]
        inter = (m[(a1, b1)] - m[(a0, b1)]) - (m[(a1, b0)] - m[(a0, b0)])
        print(f"\nЭффект {a}: {m[(a1, b0)] - m[(a0, b0)]:+,.0f} (при {b}={b0}), {m[(a1, b1)] - m[(a0, b1)]:+,.0f} "
              f"(при {b}={b1}); взаимодействие {a}×{b}: {inter:+,.0f}")
    print(f"\nВсе прогоны: {out / 'hypotheses_runs.csv'}")


if __name__ == "__main__":
    main()
