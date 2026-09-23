"""
Каналы по точке окупаемости поверх плана агента: SMS → реклама / звонок, если это окупается
по НИЖНЕЙ оценке эффекта (lcb) и хватает свободного бюджета. agent.py не меняется — план
агента дорабатывается снаружи, чтобы измерить выигрыш до переноса правила в агента.

    cd beeline_agent
    python -m sim.channel_rule --start 0 --worlds 15          # тренировочные seed
    python -m sim.channel_rule --start 20000 --worlds 15      # свежие seed для итоговой проверки

Правило: для кампании на SMS с n контактами и средним ARPU a переход на канал c выгоден, если
    lcb · (m_c − m_sms) · a − (цена_c − цена_sms) > 0   (выигрыш на контакт),
а доплата (цена_c − цена_sms) · n помещается в бюджет, оставшийся после пилотов и плана на SMS.
Повышаем жадно: сначала там, где выигрыш на рубль доплаты больше.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from sim.worlds import ROOT, list_scenarios, make_world

UPGRADES = ("digital_ads", "call")


class AgentChannels:
    def __init__(self, use: str = "lcb"):
        self.use = use

    def act(self, env):
        from agent import Agent
        ag = Agent(verbose=False)
        plan = ag.act(env)
        adds = [e for e in ag.trace if e["kind"] == "plan_add"]
        if len(adds) != len(plan):
            return plan
        prof, ch = env.customer_profile, env.channels
        free = env.remaining_budget - sum(a["contacts"] * ch[p["channel"]]["cost_per_contact"] for a, p in zip(adds, plan))
        opts = []
        for i, (a, p) in enumerate(zip(adds, plan)):
            if p["channel"] != "sms" or not a["candidate"].startswith(f"{p['filter_arpu_segment']}:{p['target_tariff']}:"):
                continue
            aud = prof[(prof["arpu_segment"] == p["filter_arpu_segment"])
                       & prof["current_tariff"].isin(p["filter_current_tariff"].split(";"))]
            aud = aud.sort_values("ID_NUMBER").head(a["contacts"])
            arpu = float(aud["predicted_arpu"].mean()) if len(aud) else 0.0
            q = a["lcb"] if self.use == "lcb" else a["mu"]
            for name in UPGRADES:
                dm = ch[name]["conversion_multiplier"] - ch["sms"]["conversion_multiplier"]
                dc = ch[name]["cost_per_contact"] - ch["sms"]["cost_per_contact"]
                gain = (q * dm * arpu - dc) * a["contacts"]
                if gain > 0:
                    opts.append((gain / (dc * a["contacts"]), i, name, dc * a["contacts"]))
        upgraded = set()
        for _, i, name, extra in sorted(opts, reverse=True):
            if i not in upgraded and extra <= free:
                plan[i] = {**plan[i], "channel": name}
                free -= extra
                upgraded.add(i)
        return plan


def _one(task):
    from eval.core import run_strategy
    from eval.strategies import STRATEGIES
    sc, s = task
    w = make_world(s, sc)
    row = {"scenario": sc, "seed": s}
    for name, make in [("agent", STRATEGIES["agent"]), ("lcb", lambda: AgentChannels("lcb")),
                       ("mu", lambda: AgentChannels("mu"))]:
        res = run_strategy(w, make, env_seed=s)
        row[name] = res["net_arpu_gain"]
        row[f"{name}_violations"] = len(res.get("violations", []))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--worlds", type=int, default=15)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "sim" / "reports" / "channel_rule_runs.csv"))
    args = ap.parse_args()
    tasks = [(sc, s) for sc in list_scenarios() for s in range(args.start, args.start + args.worlds)]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        d = pd.DataFrame(list(pool.map(_one, tasks)))
    d.to_csv(args.out, index=False)
    for v in ["lcb", "mu"]:
        diff = d[v] - d["agent"]
        print(f"{v}: {d[v].sum() / 1e6:.1f} против {d['agent'].sum() / 1e6:.1f} млн; разница {diff.mean() / 1e3:+.0f} ± "
              f"{diff.std(ddof=1) / np.sqrt(len(d)) / 1e3:.0f} тыс. на мир; лучше {(diff > 1).mean():.0%}, хуже {(diff < -1).mean():.0%}; "
              f"медиана {d[v].median() / 1e6:.3f} против {d['agent'].median() / 1e6:.3f}; худший {d[v].min() / 1e6:.3f} против "
              f"{d['agent'].min() / 1e6:.3f}; нарушений ТЗ {int(d[f'{v}_violations'].gt(0).sum())}")


if __name__ == "__main__":
    main()
