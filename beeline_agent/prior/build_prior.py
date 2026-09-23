"""
Априор из истории смен тарифов → таблица, вшитая в agent.py.

    python prior/build_prior.py            # пересчитать prior/prior_table.csv и вшить в agent.py

Владелец логики чистки и расчёта — роль «данные». Агент читает только вшитую константу
PRIOR, файлы data/ во время работы агента не открываются. Правила и обоснование — prior/README.md.

Формат строки: (tariff_from, arpu_segment, tariff_to) -> (q, n_obs, pct_std)
  q       = оценка Δ% ARPU × доля переходов (from, seg) → to   — база эффекта без канала
  n_obs   = число наблюдений в истории (после чистки)
  pct_std = разброс Δ% внутри группы (для неопределённости)

Как считается (выбрано проверкой split-half, см. prior/evaluate_estimators.py):
  * чистка — prior/history.py: дубли, ARPU до < 100, клип Δ% в [-1; 3] как в среде;
  * Δ% — среднее, сжатое к регрессии «Δ% ~ разница цен» внутри сегмента (empirical Bayes):
    связки с 1–5 наблюдениями почти целиком опираются на регрессию, с 50+ — на свои данные.
    Медиана и усечённое среднее проверены и хуже: среда считает именно среднее, а
    распределение Δ% скошено (много -100% и +300%), поэтому медиана систематически занижает;
  * доля перехода — сглажена к структуре переходов сегмента (α = 10), чтобы 1 из 2 не давало 50%;
  * pct_std — внутригрупповой разброс, для малых групп подтянут к разбросу сегмента.

Отдельно — PRIOR_UNSEEN (тот же формат, n_obs = 0) для 12 тарифов без истории:
осторожная оценка по правилу «подходит ли пакет потреблению ячейки» и по ближайшему тарифу-аналогу
(prior/unseen.py, таблица prior/prior_unseen.csv). Агент решает сам, пускать ли их в кандидаты.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from history import KEYS, ROOT, eb_shrink, load_history, pooled_sigma2  # noqa: E402
from unseen import build_unseen  # noqa: E402

CONV_ALPHA = 10.0            # сила сглаживания доли переходов
VAR_PRIOR_DOF = 5            # сколько «наблюдений» весит разброс сегмента при оценке pct_std
START, END = "# === PRIOR START (генерируется prior/build_prior.py) ===", "# === PRIOR END ==="


def build_prior(hist: pd.DataFrame, price: pd.Series) -> tuple[pd.DataFrame, dict]:
    median_price = float(price.median())
    g = hist.groupby(KEYS)["pct"]
    cells = g.agg(n_obs="size", pct_mean="mean", pct_std_raw="std").reset_index()

    # 1) Δ%: empirical Bayes — сжатие к регрессии по разнице цен внутри сегмента
    sigma2 = pooled_sigma2(hist)
    cells, seg_params = eb_shrink(cells.assign(est=cells["pct_mean"]), sigma2, price, median_price,
                                  est_col="est", n_col="n_obs")
    cells = cells.rename(columns={"shrunk": "pct_eb", "post_sd": "pct_eb_sd"}).drop(columns=["est"])

    # 2) доля переходов из (from, seg): сглаживание к структуре переходов сегмента
    totals = cells.groupby(["tariff_from", "arpu_segment"])["n_obs"].transform("sum")
    seg_share = hist.groupby(["arpu_segment", "tariff_to"]).size() / hist.groupby("arpu_segment").size()
    p0 = np.array([seg_share.get((s, t), 0.0) for s, t in zip(cells["arpu_segment"], cells["tariff_to"])])
    cells["conversion_raw"] = cells["n_obs"] / totals
    cells["conversion"] = (cells["n_obs"] + CONV_ALPHA * p0) / (totals + CONV_ALPHA)

    # 3) разброс Δ% внутри группы: для малых n подтягиваем к разбросу сегмента
    seg_var = cells["arpu_segment"].map(sigma2)
    dof = (cells["n_obs"] - 1).clip(lower=0)
    cells["pct_std"] = np.sqrt((dof * cells["pct_std_raw"].fillna(0) ** 2 + VAR_PRIOR_DOF * seg_var)
                               / (dof + VAR_PRIOR_DOF))

    cells["q_raw"] = cells["pct_mean"] * cells["conversion_raw"]      # как было (и как считает мок)
    cells["q"] = cells["pct_eb"] * cells["conversion"]
    cells["reliability"] = np.select([cells["n_obs"] >= 20, cells["n_obs"] >= 6], ["high", "medium"], "low")
    meta = {"segments": seg_params, "conv_alpha": CONV_ALPHA, "median_price": median_price}
    cols = ["tariff_from", "arpu_segment", "tariff_to", "pct_mean", "pct_eb", "pct_eb_sd", "pct_std", "n_obs",
            "conversion_raw", "conversion", "q_raw", "q", "dprice", "prior_mu", "reliability"]
    return cells[cols].sort_values(KEYS).reset_index(drop=True), meta


def _dict_lines(name: str, table: pd.DataFrame, comment: str) -> list:
    lines = [comment, f"{name} = {{"]
    for r in table.sort_values(["tariff_from", "arpu_segment", "tariff_to"]).itertuples():
        lines.append(f'    ("{r.tariff_from}", "{r.arpu_segment}", "{r.tariff_to}"): '
                     f"({r.q:.5f}, {int(r.n_obs)}, {r.pct_std:.4f}),")
    lines.append("}")
    return lines


def embed(prior: pd.DataFrame, agent_path: Path, unseen: pd.DataFrame = None) -> None:
    lines = [START] + _dict_lines("PRIOR", prior, "# (from, seg, to) -> (q, n_obs, pct_std): история смен тарифов")
    if unseen is not None and len(unseen):
        lines += _dict_lines("PRIOR_UNSEEN", unseen,
                             "# тарифы без истории: осторожная оценка по пакету и тарифу-аналогу, n_obs = 0")
    lines.append(END)
    text = agent_path.read_text(encoding="utf-8")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    agent_path.write_text(head + "\n".join(lines) + tail, encoding="utf-8")


def main() -> None:
    hist, clean_report = load_history()
    price = pd.read_csv(ROOT / "data" / "dict_tariff.csv").set_index("tariff_plan_code")["price_tariff"]
    prior, meta = build_prior(hist, price)
    prior.to_csv(ROOT / "prior" / "prior_table.csv", index=False)
    tariffs = pd.read_csv(ROOT / "data" / "dict_tariff.csv")
    profile = pd.read_csv(ROOT / "customer_profile.csv")
    unseen = build_unseen(prior, meta["segments"], meta["median_price"], profile, tariffs,
                          conv_median=float(prior["conversion_raw"].median()))
    unseen.to_csv(ROOT / "prior" / "prior_unseen.csv", index=False)
    embed(prior, ROOT / "agent.py", unseen)

    print("Чистка change_tariff:", ", ".join(f"{k}={v}" for k, v in clean_report.items()))
    for seg, p in meta["segments"].items():
        print(f"  {seg}: Δ% ≈ {p['a']:+.3f} {p['b']:+.3f}·Δцены/медиану, τ={p['tau']:.3f}, σ={p['sigma']:.3f}")
    moved = (np.sign(prior["q"]) != np.sign(prior["q_raw"])).sum()
    print(f"prior: {len(prior)} троек, положительных {(prior['q'] > 0).sum()} "
          f"(было {(prior['q_raw'] > 0).sum()}), знак сменился у {moved}; "
          f"надёжность: {prior['reliability'].value_counts().to_dict()}; вшито в agent.py")
    print(f"prior_unseen: {len(unseen)} троек по 12 тарифам без истории, положительных {(unseen['q'] > 0).sum()}; "
          f"пакет: {unseen['fit_label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
