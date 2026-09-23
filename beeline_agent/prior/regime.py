"""
Проверка режима по первым пилотам: верна ли история для нашей аудитории.

    cd beeline_agent
    python prior/regime.py --worlds 15        # калибровка на мирах sim/worlds.py, ~1 мин

Если история верна, пилот (сегмент s, текущие тарифы T, цель t, канал c, n абонентов) должен показать
    ожидание = Σ_f w_f · pct(f,s,t) · min(conv(f,s,t) · m_c, 1)     — w_f: доля тарифа f в аудитории фильтра;
а разброс вокруг ожидания складывается из трёх частей:
    шум пилота            0.804 / √n                                  (задан средой);
    состав выборки        √(Σ w_f (lift_f − ожидание)² / n · (1 − n/N)) (пилот берёт случайных n из N);
    ошибка истории        √(Σ (w_f · m_c · q_se_f)²)                   (q_se из таблицы PRIOR).
z = (наблюдение − ожидание) / √(сумма квадратов). По одному пилоту: |z| > 2 — история для этой связки
не подтвердилась. По нескольким: Q = Σ z² сравниваем с χ²-порогом для k степеней свободы (95%) —
история в целом не подходит; знак средней z показывает, завышает (−) или занижает (+) она эффект.

Функции expected_pilot / regime_check не зависят от этого репозитория (нужны только numpy и pandas) —
их можно перенести в agent.py как есть; таблицы берутся из вшитых PRIOR (q_se) и PRIOR_PARTS (pct, conv).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PILOT_NOISE_SD = 0.804
# 95%-квантили χ² для k = 1..10 степеней свободы
CHI2_95 = (3.841, 5.991, 7.815, 9.488, 11.070, 12.592, 14.067, 15.507, 16.919, 18.307)


def expected_pilot(profile: pd.DataFrame, segment: str, tariffs, target: str, multiplier: float, n: int,
                   parts: dict, prior: dict) -> dict | None:
    """Что покажет пилот, если история верна: ожидание и разброс (все три части). None — нет истории."""
    aud = profile[(profile["arpu_segment"] == segment) & profile["current_tariff"].isin(list(tariffs))]
    counts = aud["current_tariff"].value_counts()
    N = int(counts.sum())
    if N == 0:
        return None
    w, lift, hist_var = [], [], 0.0
    for f, cnt in counts.items():
        key = (f, segment, target)
        if key not in parts or key not in prior:
            return None                         # у связки нет истории — проверять нечем
        pct, conv = parts[key]
        w.append(cnt / N)
        lift.append(pct * min(conv * multiplier, 1.0))
        hist_var += (cnt / N * multiplier * prior[key][1]) ** 2
    w, lift = np.array(w), np.array(lift)
    mean = float((w * lift).sum())
    comp_var = float((w * (lift - mean) ** 2).sum()) / n * max(1.0 - n / N, 0.0)
    noise_var = PILOT_NOISE_SD ** 2 / n
    sd = math.sqrt(noise_var + comp_var + hist_var)
    return {"mean": mean, "sd": sd, "noise_sd": math.sqrt(noise_var), "composition_sd": math.sqrt(comp_var),
            "history_sd": math.sqrt(hist_var)}


def regime_check(z_values: list) -> dict:
    """Сводка по нескольким пилотам: Q = Σ z², порог χ²(95%), средняя z, вывод."""
    k = len(z_values)
    if k == 0:
        return {"k": 0, "verdict": "нет пилотов с историей"}
    q = float(sum(z * z for z in z_values))
    threshold = CHI2_95[min(k, len(CHI2_95)) - 1]
    mean_z = float(np.mean(z_values))
    return {"k": k, "Q": q, "threshold": threshold, "mean_z": mean_z,
            "history_ok": q <= threshold,
            "verdict": "история подтверждается" if q <= threshold else
                       ("история завышает эффект" if mean_z < 0 else "история занижает эффект")}


def _load_tables():
    table = pd.read_csv(ROOT / "prior" / "prior_table.csv")
    keys = list(zip(table["tariff_from"], table["arpu_segment"], table["tariff_to"]))
    parts = dict(zip(keys, zip(table["pct_mean"], table["conversion_raw"])))
    prior = dict(zip(keys, zip(table["prior_q"], table["prior_se"], table["n_obs"])))
    return parts, prior


def _first_pilots(profile, parts, prior, k: int) -> list:
    """Первые пилоты, как их выбрал бы агент: самые ценные по истории связки сегмент × цель."""
    cells = profile.dropna(subset=["current_tariff", "arpu_segment"])
    arpu = cells.groupby(["current_tariff", "arpu_segment"])["predicted_arpu"].sum()
    rows = []
    for (s, t), grp in pd.DataFrame([(f, s, t, v[0]) for (f, s, t), v in prior.items()],
                                    columns=["f", "s", "t", "q"]).groupby(["s", "t"]):
        good = grp[(grp["q"] > 0) & grp["f"].map(lambda f: (f, s) in arpu.index)]
        if good.empty:
            continue
        value = float(sum(q * arpu[(f, s)] for f, q in zip(good["f"], good["q"])))
        rows.append((value, s, t, tuple(sorted(good["f"]))))
    rows.sort(reverse=True)
    return rows[:k]


def calibrate(n_worlds: int, scenarios: list, k: int = 5, n: int = 200, channel: str = "sms") -> pd.DataFrame:
    sys.path.insert(0, str(ROOT))
    from environment import make_environment
    from mock_environment import CHANNELS, MAX_TOTAL_CONTACTS, TOTAL_BUDGET
    from sim.worlds import make_world

    profile = pd.read_csv(ROOT / "customer_profile.csv")
    tariffs = pd.read_csv(ROOT / "data" / "dict_tariff.csv")
    parts, prior = _load_tables()
    pilots = _first_pilots(profile, parts, prior, k)
    m = CHANNELS[channel]["conversion_multiplier"]
    rows = []
    for sc in scenarios:
        for seed in range(n_worlds):
            w = make_world(seed, sc)
            env, _ = make_environment(profile, w.impact_model, tariffs, CHANNELS, TOTAL_BUDGET, MAX_TOTAL_CONTACTS,
                                      w.fallback_predict, seed=seed)
            zs = []
            for _, s, t, tar in pilots:
                exp = expected_pilot(env.customer_profile, s, tar, t, m, n, parts, prior)
                if exp is None:
                    continue
                res = env.run_pilot(target_tariff=t, channel=channel, n_customers=n, filter_arpu_segment=s,
                                    filter_current_tariff=";".join(tar))
                zs.append((res["observed_lift_ratio"] - exp["mean"]) / exp["sd"])
            row = {"scenario": sc, "seed": seed}
            for j in range(1, len(zs) + 1):
                chk = regime_check(zs[:j])
                row[f"flag_{j}"] = not chk["history_ok"]
                row[f"mean_z_{j}"] = chk["mean_z"]
            row["single_flag_any"] = any(abs(z) > 2 for z in zs)
            rows.append(row)
    return pd.DataFrame(rows), pilots


def main():
    ap = argparse.ArgumentParser(description="Калибровка проверки режима по первым пилотам на мирах")
    ap.add_argument("--worlds", type=int, default=15)
    ap.add_argument("--scenarios", default="mock,resample,noise,stingy,causal,high_rich,random_soft,random,"
                                           "random_hard,flip,shift,unknown_rich")
    ap.add_argument("--pilots", type=int, default=5)
    args = ap.parse_args()
    res, pilots = calibrate(args.worlds, args.scenarios.split(","), args.pilots)
    print("Первые пилоты (сегмент → цель, тарифы):")
    for v, s, t, tar in pilots:
        print(f"  {s} → {t}: {len(tar)} тарифов, ценность по истории {v / 1e6:.2f} млн")
    flags = [c for c in res.columns if c.startswith("flag_")]
    table = res.groupby("scenario", sort=False)[flags].mean().mul(100).round(0)
    table.columns = [f"после {c.split('_')[1]} пил., %" for c in flags]
    print("\nДоля миров, где проверка говорит «история не подходит» (χ², 95%):")
    print(table.to_string())
    out = ROOT / "prior" / "reports" / "regime_calibration.csv"
    res.to_csv(out, index=False)
    print(f"\nВсе миры: {out}")


if __name__ == "__main__":
    main()
