"""
Априор из истории смен тарифов → таблица, вшитая в agent.py.

    python prior/build_prior.py            # пересчитать prior/prior_table.csv и вшить в agent.py

Владелец логики чистки и расчёта — роль «данные» (Тимур). Агент читает только
вшитую константу PRIOR, файлы data/ во время работы агента не открываются.

Формат строки: (tariff_from, arpu_segment, tariff_to) -> (q, n_obs, pct_std)
  q       = средний Δ% ARPU × доля переходов (from, seg) → to   — база эффекта без канала
  n_obs   = число наблюдений в истории
  pct_std = разброс Δ% внутри группы (для неопределённости)
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARPU_BINS = [-np.inf, 1000, 5000, np.inf]
ARPU_LABELS = ["LOW", "MID", "HIGH"]
MIN_PREV_ARPU = 100          # ниже — деление на почти ноль, Δ% бессмысленен
PCT_CLIP = (-1.0, 3.0)
START, END = "# === PRIOR START (генерируется prior/build_prior.py) ===", "# === PRIOR END ==="


def build_prior(change_tariff: pd.DataFrame) -> pd.DataFrame:
    df = change_tariff.copy()
    df = df[df["AVG_ARPU_PREV_3M"] >= MIN_PREV_ARPU]
    df["arpu_segment"] = pd.cut(df["AVG_ARPU_PREV_3M"], bins=ARPU_BINS, labels=ARPU_LABELS).astype(str)
    df["pct"] = ((df["AVG_ARPU_NEXT_3M"] - df["AVG_ARPU_PREV_3M"]) / df["AVG_ARPU_PREV_3M"]).clip(*PCT_CLIP)

    keys = ["tariff_plan_code_from", "arpu_segment", "tariff_plan_code_to"]
    g = df.groupby(keys).agg(pct_mean=("pct", "mean"), pct_std=("pct", "std"), n_obs=("pct", "size")).reset_index()
    totals = g.groupby(["tariff_plan_code_from", "arpu_segment"])["n_obs"].transform("sum")
    g["conversion"] = g["n_obs"] / totals
    g["q"] = g["pct_mean"] * g["conversion"]
    g["pct_std"] = g["pct_std"].fillna(g["pct_std"].median())
    return g.rename(columns={"tariff_plan_code_from": "tariff_from", "tariff_plan_code_to": "tariff_to"})


def embed(prior: pd.DataFrame, agent_path: Path) -> None:
    lines = [START, "PRIOR = {"]
    for r in prior.sort_values(["tariff_from", "arpu_segment", "tariff_to"]).itertuples():
        lines.append(f'    ("{r.tariff_from}", "{r.arpu_segment}", "{r.tariff_to}"): '
                     f"({r.q:.5f}, {int(r.n_obs)}, {r.pct_std:.4f}),")
    lines += ["}", END]
    text = agent_path.read_text(encoding="utf-8")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    agent_path.write_text(head + "\n".join(lines) + tail, encoding="utf-8")


def main() -> None:
    prior = build_prior(pd.read_csv(ROOT / "data" / "change_tariff.csv"))
    prior.to_csv(ROOT / "prior" / "prior_table.csv", index=False)
    embed(prior, ROOT / "agent.py")
    print(f"prior: {len(prior)} троек, положительных {(prior['q'] > 0).sum()}, вшито в agent.py")


if __name__ == "__main__":
    main()
