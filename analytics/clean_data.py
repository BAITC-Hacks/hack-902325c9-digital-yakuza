"""
Шаг 1. Очистка данных и отчёт о качестве.

    python analytics/clean_data.py [--data-dir beeline_case_participants] [--out analytics/output]

Что делает:
  * customer_profile — строки НЕ удаляются (среда работает с сырым профилем), добавляются флаги качества
    и проверка, что сегменты воспроизводятся по правилам ТЗ;
  * change_tariff — убираются полные дубли, считается относительный эффект pct как в среде
    (ARPU до >= 100, клип [-1; 3]), ARPU-сегмент по ARPU до смены, направление UPSELL/DOWNSELL/FLAT,
    сверка с arpu_monthly и traffic;
  * arpu_monthly — полные дубли удаляются, конфликты (ID, месяц) суммируются с флагом;
  * traffic — флаги пропусков/противоречий; фичи истории за 3 месяца до смены в формате профиля.

Выход (в --out):
  clean/customer_profile_clean.csv, clean/change_tariff_clean.csv, clean/arpu_monthly_clean.csv,
  clean/history_features.csv, data_quality.json
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common import (DIRECTION_THRESHOLD, HISTORY_PRE_MONTHS, PCT_CLIP, PREV_ARPU_MIN, TRAFFIC_COLS,
                    arpu_segment, call_segment, data_segment, parse_args, write_json)


def clean_profile(profile: pd.DataFrame, report: dict) -> pd.DataFrame:
    df = profile.copy()
    n = len(df)
    df["flag_no_tariff"] = df["current_tariff"].isna()
    df["flag_no_arpu"] = df["ARPU_3m_avg"].isna()
    df["flag_no_traffic"] = df["DATA_VOLUME"].isna()
    df["flag_zero_pred_arpu"] = df["predicted_arpu"].fillna(0) <= 0
    p999 = df["predicted_arpu"].quantile(0.999)
    df["flag_pred_arpu_outlier"] = df["predicted_arpu"] > p999
    df["flag_lte_gt_total"] = df["LTE_DATA_VOLUME"] > df["DATA_VOLUME"]
    df["cell"] = df["current_tariff"].fillna("NA") + "|" + df["arpu_segment"].fillna("NA")

    seg_check = {
        "arpu_segment_mismatch": int((arpu_segment(df["ARPU_3m_avg"]).fillna("NA")
                                      != df["arpu_segment"].fillna("NA")).sum()),
        "data_segment_mismatch": int((data_segment(df["DATA_VOLUME"]).fillna("NA")
                                      != df["data_segment"].fillna("NA")).sum()),
        "call_segment_mismatch": int((call_segment(df["OUT_LOC_ONNET_MIN"], df["OUT_LOC_OFFNET_MIN"])
                                      != df["call_segment"]).sum()),
    }
    report["customer_profile"] = {
        "rows": n,
        "unique_ids": int(df["ID_NUMBER"].nunique()),
        "baseline_predicted_arpu_sum": float(df["predicted_arpu"].sum()),
        "no_current_tariff": int(df["flag_no_tariff"].sum()),
        "no_arpu_and_arpu_segment": int(df["flag_no_arpu"].sum()),
        "no_traffic_and_data_segment": int(df["flag_no_traffic"].sum()),
        "zero_predicted_arpu": int(df["flag_zero_pred_arpu"].sum()),
        "predicted_arpu_p999": float(p999),
        "predicted_arpu_max": float(df["predicted_arpu"].max()),
        "predicted_arpu_max_id": int(df.loc[df["predicted_arpu"].idxmax(), "ID_NUMBER"]),
        "lte_greater_than_total_data": int(df["flag_lte_gt_total"].sum()),
        "segment_rules_check": seg_check,
        "segment_rules": {
            "arpu_segment": "ARPU_3m_avg: <=1000 LOW, <=5000 MID, >5000 HIGH",
            "data_segment": "DATA_VOLUME: 0 NON_USER, <=2000 LITE, >2000 HEAVY (LTE_DATA_VOLUME не участвует)",
            "call_segment": "OUT_LOC_ONNET_MIN + OUT_LOC_OFFNET_MIN: <100 LOW, <=400 MEDIUM, >400 HIGH; пропуск -> LOW",
        },
        "segments": {c: df[c].fillna("NA").value_counts().to_dict()
                     for c in ["arpu_segment", "data_segment", "call_segment", "ARPU_trend"]},
        "n_cells_tariff_x_arpu_segment": int(df.dropna(subset=["current_tariff", "arpu_segment"])
                                             .groupby(["current_tariff", "arpu_segment"]).ngroups),
    }
    return df


def clean_arpu_monthly(am: pd.DataFrame, report: dict) -> pd.DataFrame:
    n0 = len(am)
    am1 = am.drop_duplicates()
    conflicts = am1.duplicated(["ID_NUMBER", "TIME_KEY"], keep=False)
    conflict_keys = am1.loc[conflicts, ["ID_NUMBER", "TIME_KEY"]].drop_duplicates()
    # две разные записи за один месяц — считаем их начислениями одного абонента и суммируем
    out = (am1.groupby(["ID_NUMBER", "TIME_KEY"], as_index=False)
              .agg(ARPU_1M=("ARPU_1M", "sum"), n_records=("ARPU_1M", "size")))
    out["flag_conflict_summed"] = out["n_records"] > 1
    report["arpu_monthly"] = {
        "rows": n0,
        "full_duplicates_removed": int(n0 - len(am1)),
        "conflicting_id_month_groups": int(len(conflict_keys)),
        "conflicting_rows": int(conflicts.sum()),
        "negative_arpu_rows": int((am1["ARPU_1M"] < 0).sum()),
        "zero_arpu_rows": int((am1["ARPU_1M"] == 0).sum()),
        "months": sorted(am1["TIME_KEY"].unique().tolist()),
        "rows_after_clean": int(len(out)),
    }
    return out


def clean_change_tariff(ct: pd.DataFrame, am_clean: pd.DataFrame, traffic: pd.DataFrame,
                        report: dict) -> pd.DataFrame:
    n0 = len(ct)
    df = ct.drop_duplicates().copy()
    df["TIME_KEY"] = pd.to_datetime(df["TIME_KEY"]).dt.strftime("%Y-%m-%d")
    prev, nxt = df["AVG_ARPU_PREV_3M"], df["AVG_ARPU_NEXT_3M"]
    df["flag_prev_lt_100"] = prev < PREV_ARPU_MIN
    df["flag_prev_le_0"] = prev <= 0
    df["flag_next_le_0"] = nxt <= 0
    df["arpu_segment"] = arpu_segment(prev)
    raw = (nxt - prev) / prev
    df["pct_raw"] = raw.where(prev >= PREV_ARPU_MIN)
    df["pct"] = df["pct_raw"].clip(*PCT_CLIP)
    df["flag_pct_clipped"] = df["pct_raw"].notna() & ((df["pct_raw"] < PCT_CLIP[0]) | (df["pct_raw"] > PCT_CLIP[1]))
    df["direction"] = np.select([raw > DIRECTION_THRESHOLD, raw < -DIRECTION_THRESHOLD],
                                ["UPSELL", "DOWNSELL"], default="FLAT")
    df["usable_for_prior"] = df["pct"].notna()

    # сверка ARPU до смены с помесячной выручкой (3 мес. до смены)
    pre = (am_clean[am_clean["TIME_KEY"].isin(HISTORY_PRE_MONTHS)]
           .groupby("ID_NUMBER")["ARPU_1M"].mean().rename("arpu_monthly_pre3m_mean"))
    df = df.merge(pre, left_on="ID_NUMBER", right_index=True, how="left")
    df["check_prev_vs_monthly_absdiff"] = (df["arpu_monthly_pre3m_mean"] - prev).abs()

    # сверка тарифа "откуда" с тарифом в traffic за последний месяц до смены
    last = (traffic[traffic["time_key"] == HISTORY_PRE_MONTHS[-1]]
            .drop_duplicates("ID_NUMBER").set_index("ID_NUMBER")["tariff_plan_code"].rename("traffic_tariff_last_month"))
    df = df.merge(last, left_on="ID_NUMBER", right_index=True, how="left")

    usable = df[df["usable_for_prior"]]
    report["change_tariff"] = {
        "rows": n0,
        "full_duplicates_removed": int(n0 - len(df)),
        "change_months": sorted(df["TIME_KEY"].unique().tolist()),
        "from_equals_to": int((df["tariff_plan_code_from"] == df["tariff_plan_code_to"]).sum()),
        "prev_arpu_lt_100_excluded": int(df["flag_prev_lt_100"].sum()),
        "prev_arpu_le_0": int(df["flag_prev_le_0"].sum()),
        "next_arpu_le_0": int(df["flag_next_le_0"].sum()),
        "usable_for_prior": int(len(usable)),
        "pct_clipped_share": float(df["flag_pct_clipped"].sum() / max(len(usable), 1)),
        "pct_raw_quantiles": {str(q): float(v) for q, v in
                              usable["pct_raw"].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).items()},
        "direction_share": df.loc[df["usable_for_prior"], "direction"].value_counts(normalize=True).round(4).to_dict(),
        "from_tariffs": int(df["tariff_plan_code_from"].nunique()),
        "to_tariffs": sorted(df["tariff_plan_code_to"].unique().tolist(), key=lambda t: int(t.split("_")[1])),
        "prev_matches_arpu_monthly_share": float((df["check_prev_vs_monthly_absdiff"] < 1).mean()),
        "from_matches_traffic_tariff_share": float((df["traffic_tariff_last_month"]
                                                     == df["tariff_plan_code_from"]).mean()),
        "by_arpu_segment": (usable.groupby("arpu_segment")["pct"]
                            .agg(n="size", mean="mean", median="median",
                                 upsell_share=lambda s: (s > DIRECTION_THRESHOLD).mean(),
                                 downsell_share=lambda s: (s < -DIRECTION_THRESHOLD).mean())
                            .round(4).to_dict(orient="index")),
    }
    return df


def build_history_features(traffic: pd.DataFrame, ct_clean: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Фичи истории в формате customer_profile (среднее за 3 месяца до смены) — для сравнения популяций."""
    tr = traffic[traffic["time_key"].isin(HISTORY_PRE_MONTHS)]
    feats = tr.groupby("ID_NUMBER")[TRAFFIC_COLS].mean()
    feats["months_observed"] = tr.groupby("ID_NUMBER").size()
    hist = (ct_clean.drop_duplicates("ID_NUMBER")
            .set_index("ID_NUMBER")[["tariff_plan_code_from", "tariff_plan_code_to", "AVG_ARPU_PREV_3M",
                                     "AVG_ARPU_NEXT_3M", "arpu_segment", "pct", "usable_for_prior"]])
    out = hist.join(feats, how="left")
    out["data_segment"] = data_segment(out["DATA_VOLUME"])
    out["call_segment"] = call_segment(out["OUT_LOC_ONNET_MIN"], out["OUT_LOC_OFFNET_MIN"])
    out["flag_lte_gt_total"] = out["LTE_DATA_VOLUME"] > out["DATA_VOLUME"]
    report["traffic"] = {
        "rows": int(len(traffic)),
        "months": sorted(traffic["time_key"].unique().tolist()),
        "missing_tariff_plan_code_rows": int(traffic["tariff_plan_code"].isna().sum()),
        "missing_device_id_rows": int(traffic["DEVICE_ID"].isna().sum()),
        "missing_contact_metrics_rows": int(traffic["COUNT_CONTACT"].isna().sum()),
        "lte_greater_than_total_rows": int((traffic["LTE_DATA_VOLUME"] > traffic["DATA_VOLUME"]).sum()),
        "ids_with_tariff_change_inside_period": int((traffic.groupby("ID_NUMBER")["tariff_plan_code"].nunique() > 1).sum()),
        "history_customers_with_pre_period_traffic": int(out["DATA_VOLUME"].notna().sum()),
    }
    return out.reset_index()


def main():
    args = parse_args("Очистка данных кейса Beeline")
    data_dir, out_dir = Path(args.data_dir), Path(args.out)
    (out_dir / "clean").mkdir(parents=True, exist_ok=True)

    profile = pd.read_csv(data_dir / "customer_profile.csv")
    ct = pd.read_csv(data_dir / "data" / "change_tariff.csv")
    am = pd.read_csv(data_dir / "data" / "arpu_monthly.csv")
    traffic = pd.read_csv(data_dir / "data" / "traffic.csv")

    report: dict = {}
    profile_c = clean_profile(profile, report)
    am_c = clean_arpu_monthly(am, report)
    ct_c = clean_change_tariff(ct, am_c, traffic, report)
    hist_f = build_history_features(traffic, ct_c, report)

    ids = {"customer_profile": set(profile["ID_NUMBER"]), "change_tariff": set(ct["ID_NUMBER"]),
           "arpu_monthly": set(am["ID_NUMBER"]), "traffic": set(traffic["ID_NUMBER"])}
    report["id_overlap"] = {f"{a}&{b}": len(ids[a] & ids[b])
                            for i, a in enumerate(ids) for b in list(ids)[i + 1:]}

    profile_c.to_csv(out_dir / "clean" / "customer_profile_clean.csv", index=False)
    ct_c.to_csv(out_dir / "clean" / "change_tariff_clean.csv", index=False)
    am_c.to_csv(out_dir / "clean" / "arpu_monthly_clean.csv", index=False)
    hist_f.to_csv(out_dir / "clean" / "history_features.csv", index=False)
    write_json(report, out_dir / "data_quality.json")

    cp, ctr = report["customer_profile"], report["change_tariff"]
    print(f"customer_profile: {cp['rows']:,} абонентов, без тарифа {cp['no_current_tariff']}, "
          f"без ARPU {cp['no_arpu_and_arpu_segment']}, predicted_arpu=0 у {cp['zero_predicted_arpu']}")
    print(f"change_tariff: {ctr['rows']:,} строк, дублей {ctr['full_duplicates_removed']}, "
          f"пригодно для приора {ctr['usable_for_prior']:,}, целевых тарифов {len(ctr['to_tariffs'])}")
    print(f"пересечение ID аудитории и истории: {report['id_overlap']['customer_profile&change_tariff']}")
    print(f"Сохранено в {out_dir}")


if __name__ == "__main__":
    main()
