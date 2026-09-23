"""
Шаг 3. Карта аудитории кампаний и сравнение с исторической выборкой.

    python analytics/audience_map.py [--data-dir beeline_case_participants] [--out analytics/output]

Выход (в --out/audience):
  cells.csv            — ячейки «текущий тариф × ARPU-сегмент»: размер, predicted_arpu, покрытие историей;
  subsegments.csv      — ячейка × data_segment × call_segment: сколько абонентов и какой у них predicted_arpu
                         (эффект кампании пропорционален predicted_arpu, поэтому фильтр по call/data позволяет
                         при ограниченном бюджете взять самых ценных абонентов ячейки);
  history_vs_audience.json — насколько история похожа на аудиторию (сегменты, тарифы, ARPU, поведение).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common import parse_args, write_json

MAX_CUSTOMERS_PER_CAMPAIGN = 5000


def shares(s: pd.Series) -> dict:
    return s.fillna("NA").value_counts(normalize=True).round(4).to_dict()


def main():
    args = parse_args("Карта аудитории и сравнение с историей")
    data_dir, out_dir = Path(args.data_dir), Path(args.out)
    aud_dir = out_dir / "audience"
    aud_dir.mkdir(parents=True, exist_ok=True)
    clean = out_dir / "clean"
    if not (clean / "history_features.csv").exists():
        raise SystemExit("Сначала запустите analytics/clean_data.py")

    profile = pd.read_csv(data_dir / "customer_profile.csv")
    hist = pd.read_csv(clean / "history_features.csv")
    total_arpu = profile["predicted_arpu"].sum()

    # --- ячейки --------------------------------------------------------------
    hist_from = (hist.groupby(["tariff_plan_code_from", "arpu_segment"]).size()
                 .rename("history_switches").reset_index()
                 .rename(columns={"tariff_plan_code_from": "current_tariff"}))
    cells = (profile.groupby(["current_tariff", "arpu_segment"], dropna=False)
             .agg(n_customers=("ID_NUMBER", "size"), arpu_sum=("predicted_arpu", "sum"),
                  arpu_mean=("predicted_arpu", "mean"), arpu_p90=("predicted_arpu", lambda x: x.quantile(0.9)),
                  zero_arpu=("predicted_arpu", lambda x: int((x <= 0).sum())),
                  heavy_data_share=("data_segment", lambda x: (x == "HEAVY").mean()),
                  high_call_share=("call_segment", lambda x: (x == "HIGH").mean()))
             .reset_index().merge(hist_from, on=["current_tariff", "arpu_segment"], how="left")
             .fillna({"history_switches": 0})
             .sort_values("arpu_sum", ascending=False))
    cells["share_customers"] = cells["n_customers"] / len(profile)
    cells["share_arpu"] = cells["arpu_sum"] / total_arpu
    cells["cum_share_arpu"] = cells["share_arpu"].cumsum()
    cells["addressable_by_filters"] = cells["current_tariff"].notna() & cells["arpu_segment"].notna()
    cells["fits_one_campaign"] = cells["n_customers"] <= MAX_CUSTOMERS_PER_CAMPAIGN
    cells.to_csv(aud_dir / "cells.csv", index=False)

    # --- подсегменты внутри ячеек --------------------------------------------
    sub = (profile.dropna(subset=["current_tariff", "arpu_segment"])
           .groupby(["current_tariff", "arpu_segment", "data_segment", "call_segment"], dropna=False)
           .agg(n_customers=("ID_NUMBER", "size"), arpu_sum=("predicted_arpu", "sum"),
                arpu_mean=("predicted_arpu", "mean"))
           .reset_index())
    cell_mean = sub.groupby(["current_tariff", "arpu_segment"])["arpu_sum"].transform("sum") \
        / sub.groupby(["current_tariff", "arpu_segment"])["n_customers"].transform("sum")
    sub["arpu_vs_cell_mean"] = sub["arpu_mean"] / cell_mean
    sub = sub.sort_values(["current_tariff", "arpu_segment", "arpu_mean"], ascending=[True, True, False])
    sub.to_csv(aud_dir / "subsegments.csv", index=False)
    by_call = (profile.dropna(subset=["current_tariff", "arpu_segment"])
               .assign(rel=lambda d: d["predicted_arpu"] / d.groupby(["current_tariff", "arpu_segment"])
                       ["predicted_arpu"].transform("mean"))
               .groupby("call_segment")["rel"].mean().round(3).to_dict())

    # --- история vs аудитория --------------------------------------------------
    pre = hist  # фичи истории за 3 мес. до смены
    q = [0.1, 0.25, 0.5, 0.75, 0.9]
    arpu_q = {}
    for seg in ["LOW", "MID", "HIGH"]:
        a = profile.loc[profile["arpu_segment"] == seg, "ARPU_3m_avg"]
        h = pre.loc[pre["arpu_segment"] == seg, "AVG_ARPU_PREV_3M"]
        arpu_q[seg] = {"audience": a.quantile(q).round(1).to_dict(), "history": h.quantile(q).round(1).to_dict()}
    behaviour = {}
    for col in ["DATA_VOLUME", "OUT_LOC_ONNET_MIN", "OUT_LOC_OFFNET_MIN", "COUNT_CONTACT", "COUNT_BASE_STATION"]:
        behaviour[col] = {"audience_median": float(profile[col].median()), "history_median": float(pre[col].median())}
    tariff_mix = pd.DataFrame({
        "audience": profile["current_tariff"].value_counts(normalize=True),
        "history_from": pre["tariff_plan_code_from"].value_counts(normalize=True),
    }).fillna(0).round(4)
    tariff_mix["diff_pp"] = (100 * (tariff_mix["audience"] - tariff_mix["history_from"])).round(1)
    report = {
        "customers": {"audience": int(len(profile)), "history": int(len(pre))},
        "arpu_segment_share": {"audience": shares(profile["arpu_segment"]), "history": shares(pre["arpu_segment"])},
        "data_segment_share": {"audience": shares(profile["data_segment"]), "history": shares(pre["data_segment"])},
        "call_segment_share": {"audience": shares(profile["call_segment"]), "history": shares(pre["call_segment"])},
        "arpu_quantiles_within_segment": arpu_q,
        "behaviour_medians": behaviour,
        "tariff_mix": tariff_mix.sort_values("audience", ascending=False).to_dict(orient="index"),
        "predicted_arpu_by_call_segment_vs_cell_mean": by_call,
        "top_cells_cover": {f"top{k}_share_arpu": float(cells["share_arpu"].head(k).sum()) for k in (5, 10, 15)},
        "not_addressable_by_tariff_filter": int(profile["current_tariff"].isna().sum()),
    }
    write_json(report, aud_dir / "history_vs_audience.json")

    a, h = report["arpu_segment_share"]["audience"], report["arpu_segment_share"]["history"]
    print(f"Ячеек: {len(cells)}; топ-10 по ARPU покрывают {report['top_cells_cover']['top10_share_arpu']:.0%} ARPU")
    print(f"Доля HIGH: аудитория {a.get('HIGH', 0):.0%} vs история {h.get('HIGH', 0):.0%}; "
          f"LOW: {a.get('LOW', 0):.0%} vs {h.get('LOW', 0):.0%}")
    print(f"Медиана ARPU внутри HIGH: аудитория {arpu_q['HIGH']['audience'][0.5]:,.0f} vs история "
          f"{arpu_q['HIGH']['history'][0.5]:,.0f}")
    print(f"predicted_arpu по call_segment относительно среднего ячейки: {by_call}")
    print(f"Сохранено в {aud_dir}")


if __name__ == "__main__":
    main()
