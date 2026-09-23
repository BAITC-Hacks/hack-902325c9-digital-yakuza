"""
Помогает ли новый приор агенту там, где истина — не наша история? Проверка на «чужих мирах».

    python prior/ab_worlds.py [--worlds 10]

Мир = историю случайно делим по абонентам пополам. По половине B строим «истинные» эффекты
среды (тем же способом, что мок), по половине A — приор агента. Так моделируется главное
условие кейса: история описывает другую выборку, чем та, на которой считается результат.
Второй тип мира — «сдвиг»: истина строится по половине истории, больше похожей на аудиторию
(по сочетанию ARPU/data/call-сегментов), приор — по остальной.

В каждом мире агент (agent.py как есть) запускается дважды: с сырым приором (среднее Δ% ×
сырая доля перехода, как мок) и с иерархическим (уровень страты + контраст, prior/hier.py).
Выход: prior/reports/ab_worlds.csv + сводка.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import agent as agent_mod  # noqa: E402
from build_prior import build  # noqa: E402
from environment import make_environment  # noqa: E402
from history import clean_history  # noqa: E402
from mock_environment import CHANNELS, MAX_TOTAL_CONTACTS, TOTAL_BUDGET, _mock_fallback, _mock_impact_model  # noqa: E402
from scoring_core import MAX_CAMPAIGNS, sanitize_campaigns, score_campaigns  # noqa: E402

FILTER_COLS = ["filter_arpu_segment", "filter_data_segment", "filter_call_segment", "filter_current_tariff"]


def old_prior(raw: pd.DataFrame) -> dict:
    """Приор в том виде, как он был до изменений (для сравнения)."""
    df = raw[raw["AVG_ARPU_PREV_3M"] >= 100].copy()
    df["arpu_segment"] = pd.cut(df["AVG_ARPU_PREV_3M"], bins=[-np.inf, 1000, 5000, np.inf],
                                labels=["LOW", "MID", "HIGH"]).astype(str)
    df["pct"] = ((df["AVG_ARPU_NEXT_3M"] - df["AVG_ARPU_PREV_3M"]) / df["AVG_ARPU_PREV_3M"]).clip(-1, 3)
    keys = ["tariff_plan_code_from", "arpu_segment", "tariff_plan_code_to"]
    g = df.groupby(keys).agg(pct_mean=("pct", "mean"), pct_std=("pct", "std"), n_obs=("pct", "size")).reset_index()
    g["conversion"] = g["n_obs"] / g.groupby(["tariff_plan_code_from", "arpu_segment"])["n_obs"].transform("sum")
    g["q"] = g["pct_mean"] * g["conversion"]
    g["pct_std"] = g["pct_std"].fillna(g["pct_std"].median())
    return {(r.tariff_plan_code_from, r.arpu_segment, r.tariff_plan_code_to): (r.q, int(r.n_obs), r.pct_std)
            for r in g.itertuples()}


def new_prior(raw: pd.DataFrame, tariffs: pd.DataFrame, profile: pd.DataFrame) -> dict:
    hist, _ = clean_history(raw)
    seen, _, _, _ = build(hist, tariffs, profile)
    return {(r.tariff_from, r.arpu_segment, r.tariff_to): (r.q, int(r.n_obs), r.pct_std) for r in seen.itertuples()}


def run_world(prior: dict, truth: pd.DataFrame, profile, tariffs, seed: int) -> dict:
    env, internals = make_environment(profile, truth, tariffs, CHANNELS, TOTAL_BUDGET, MAX_TOTAL_CONTACTS,
                                      _mock_fallback, seed=seed)
    agent_mod.PRIOR = prior
    try:
        final = agent_mod.Agent(verbose=False).act(env)
    except Exception as exc:  # noqa: BLE001
        print(f"  агент упал: {exc}")
        final = []
    final = sanitize_campaigns(final, env.tariffs)[:MAX_CAMPAIGNS]
    camps = pd.DataFrame(internals.executed_pilot_campaigns() + final)
    for col in FILTER_COLS + ["explicit_ids"]:
        if col not in camps.columns:
            camps[col] = None
    res = score_campaigns(camps, env.customer_profile, truth, env.tariffs,
                          env.customer_profile["predicted_arpu"].sum(), _mock_fallback)
    return {"net": res["net_arpu_gain"], "n_final": len(final), "risk": res["risk_score_pct"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=10)
    args = parser.parse_args()

    raw = pd.read_csv(ROOT / "data" / "change_tariff.csv")
    profile = pd.read_csv(ROOT / "customer_profile.csv")
    tariffs = pd.read_csv(ROOT / "data" / "dict_tariff.csv")

    # вес «похожести на аудиторию» для миров со сдвигом — грубо по ARPU-сегменту до смены
    seg = pd.cut(raw["AVG_ARPU_PREV_3M"], [-np.inf, 1000, 5000, np.inf], labels=["LOW", "MID", "HIGH"]).astype(str)
    w_seg = (profile["arpu_segment"].value_counts(normalize=True) / seg.value_counts(normalize=True)).to_dict()
    raw_w = seg.map(w_seg).fillna(0.0)

    rng = np.random.default_rng(2026)
    ids = raw["ID_NUMBER"].unique()
    rows = []
    for k in range(args.worlds):
        for kind in ["random"]:
            if kind == "random":
                a_ids = set(rng.choice(ids, size=len(ids) // 2, replace=False))
                in_a = raw["ID_NUMBER"].isin(a_ids)
            else:
                score = raw_w.values + rng.random(len(raw)) * 1e-3   # похожие на аудиторию → в «истину»
                in_a = pd.Series(score < np.median(score), index=raw.index)
            A, B = raw[in_a], raw[~in_a]
            truth = _mock_impact_model(B)
            priors = {"old": old_prior(A), "new": new_prior(A, tariffs, profile)}
            for name, pr in priors.items():
                r = run_world(pr, truth, profile, tariffs, seed=k)
                rows.append({"world": k, "kind": kind, "prior": name, **r})
            last = rows[-2:]
            print(f"мир {k:2d} [{kind:6s}] сырой {last[0]['net']:>12,.0f}   иерархия {last[1]['net']:>12,.0f}", flush=True)
    res = pd.DataFrame(rows)
    out = HERE / "reports"
    out.mkdir(exist_ok=True)
    res.to_csv(out / "ab_worlds.csv", index=False)
    piv0 = res.pivot_table(index=["world", "kind"], columns="prior", values="net")
    d = piv0["new"] - piv0["old"]
    print(f"\nПарная разница иерархия − сырой: {d.mean():+,.0f} ± {d.std(ddof=1) / np.sqrt(len(d)):,.0f} (ст. ошибка), "
          f"иерархия лучше в {(d > 0).mean():.0%} миров")
    piv = res.pivot_table(index=["world", "kind"], columns="prior", values="net").reset_index()
    for kind, d in piv.groupby("kind"):
        diff = d["new"] - d["old"]
        print(f"\n[{kind}] медиана: старый {d['old'].median():,.0f}  новый {d['new'].median():,.0f} | "
              f"минимум: старый {d['old'].min():,.0f}  новый {d['new'].min():,.0f} | "
              f"новый лучше в {(diff > 0).sum()} из {len(d)} миров, средний выигрыш {diff.mean():+,.0f}")


if __name__ == "__main__":
    main()
