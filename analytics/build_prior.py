"""
Шаг 2. Априорная таблица эффектов смены тарифа (для agent.py).

    python analytics/build_prior.py [--data-dir beeline_case_participants] [--out analytics/output]

Среда считает эффект абонента так (scoring_core.score_campaign):
    lift = pct(from, to, arpu_segment) * min(conv(from, to, arpu_segment) * m_channel, 1) * predicted_arpu
где pct — среднее относительное изменение ARPU после смены, conv — доля переходов from→to среди всех
смен из (from, сегмент). Историческая выборка — ДРУГИЕ абоненты, поэтому это только априор.

Что делаем поверх «сырого» среднего (которое и есть мок-модель):
  1. Поправка на сдвиг популяции. Внутри ARPU-сегмента pct сильно зависит от уровня ARPU
     (регрессия к среднему), а аудитория кампаний богаче истории в HIGH и беднее в LOW.
     Историю перевзвешиваем под распределение ARPU аудитории (веса по квантильным корзинам
     внутри сегмента).
  2. Сжатие (empirical Bayes). Среднее по ячейке с малым n подтягивается к регрессии
     pct ~ a_s + b_s * Δprice (по сегменту); τ² — межъячеечная дисперсия, σ² — внутриячеечная.
  3. Переходы без истории: pct из регрессии по цене, conv — медианная (так делает fallback среды),
     неопределённость расширена на расхождение с мок-fallback.
  4. Неопределённость: pct_sd (выборочная) и pct_sd_total = sqrt(pct_sd² + SHIFT_SD²) — с запасом
     на то, что на судействе эффекты другие. Базовый эффект base = pct·conv·κ (эффект до множителя
     канала), где κ — логнормальный множитель с log-sd CONV_SCALE_SD: доля переходов в истории —
     не то же самое, что отклик на кампанию, её масштаб в реальной среде неизвестен. Знак base
     задаёт pct, поэтому заведомый downsell остаётся downsell'ом. Для каналов — ratio и EV на контакт,
     и сколько «пилотных абонентов» стоит априор (prior_equiv_pilot_n_*).

Выход (в --out/prior): prior_effects.csv, pilot_candidates.csv, audience_cells.csv,
prior_meta.json, prior_table.py (словарь для вставки в agent.py).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from common import CHANNELS, PCT_CLIP, PILOT_NOISE_STD, parse_args, write_json

SHIFT_SD = 0.15          # запас на отличие реальной аудитории от истории (в единицах pct); настраивается
CONV_SCALE_SD = 0.5      # log-sd неизвестного множителя к conv: масштаб конверсии в реальной среде не известен
CONV_SCALE_SD_FALLBACK = 0.8  # то же для переходов без истории (conv = медиана, как fallback среды)
N_REWEIGHT_BINS = 5      # квантильные корзины ARPU внутри сегмента для перевзвешивания
WEIGHT_CLIP = (0.2, 5.0)
CONV_ALPHA = 10.0        # сила сглаживания долей переходов (Дирихле)
MIN_CELL_N_FOR_TAU = 5   # ячейки для оценки межъячеечной дисперсии
SEGMENTS = ["LOW", "MID", "HIGH"]


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def mock_fallback_pct(price_from: float, price_to: float, median_price: float) -> float:
    """Как _mock_fallback в mock_environment.py (для справки: реальное правило среды скрыто)."""
    return float(np.clip((price_to - price_from) / max(median_price, 1.0) * 0.4, *PCT_CLIP))


def reweight_history(hist: pd.DataFrame, profile: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Вес исторического абонента = доля его ARPU-корзины в аудитории / доля в истории (внутри сегмента)."""
    w = pd.Series(1.0, index=hist.index)
    info = {}
    for seg in SEGMENTS:
        aud = profile.loc[profile["arpu_segment"] == seg, "ARPU_3m_avg"].dropna()
        mask = hist["arpu_segment"] == seg
        h = hist.loc[mask, "AVG_ARPU_PREV_3M"]
        if len(aud) < 50 or mask.sum() < 50:
            continue
        edges = np.unique(np.quantile(aud, np.linspace(0, 1, N_REWEIGHT_BINS + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        a_share = pd.cut(aud, edges).value_counts(normalize=True, sort=False)
        h_bins = pd.cut(h, edges)
        h_share = h_bins.value_counts(normalize=True, sort=False)
        ratio = (a_share / h_share.replace(0, np.nan)).fillna(1.0).clip(*WEIGHT_CLIP)
        w.loc[mask] = h_bins.map(ratio).astype(float).values
        info[seg] = {"audience_median_arpu": float(aud.median()), "history_median_arpu": float(h.median()),
                     "bin_weights": [round(float(x), 3) for x in ratio.values]}
    return w, info


def main():
    args = parse_args("Априорная таблица эффектов")
    data_dir, out_dir = Path(args.data_dir), Path(args.out)
    prior_dir = out_dir / "prior"
    prior_dir.mkdir(parents=True, exist_ok=True)

    clean = out_dir / "clean"
    if not (clean / "change_tariff_clean.csv").exists():
        raise SystemExit("Сначала запустите analytics/clean_data.py")
    hist = pd.read_csv(clean / "change_tariff_clean.csv")
    hist = hist[hist["usable_for_prior"]].copy()
    profile = pd.read_csv(data_dir / "customer_profile.csv")
    tariffs = pd.read_csv(data_dir / "data" / "dict_tariff.csv")
    price = tariffs.set_index("tariff_plan_code")["price_tariff"]
    median_price = float(price.median())
    all_targets = sorted(tariffs["tariff_plan_code"], key=lambda t: int(t.split("_")[1]))
    seen_targets = set(hist["tariff_plan_code_to"])

    # --- 1. перевзвешивание под аудиторию ----------------------------------
    hist["w"], reweight_info = reweight_history(hist, profile)
    hist["w_pct"] = hist["w"] * hist["pct"]
    hist["w_pct2"] = hist["w"] * hist["pct"] ** 2

    grp = hist.groupby(["tariff_plan_code_from", "tariff_plan_code_to", "arpu_segment"])
    cells = grp.agg(n=("pct", "size"), pct_hist=("pct", "mean"), sw=("w", "sum"),
                    sw2=("w", lambda x: float((x ** 2).sum())), swp=("w_pct", "sum")).reset_index()
    cells["pct_rw"] = cells["swp"] / cells["sw"]
    cells["n_eff"] = cells["sw"] ** 2 / cells["sw2"]
    totals = (hist.groupby(["tariff_plan_code_from", "arpu_segment"])
              .agg(N_from=("pct", "size"), SW_from=("w", "sum")).reset_index())
    cells = cells.merge(totals, on=["tariff_plan_code_from", "arpu_segment"])
    cells["conv_hist"] = cells["n"] / cells["N_from"]          # как в мок-модели
    cells["conv_rw"] = cells["sw"] / cells["SW_from"]
    cells["dp"] = (cells["tariff_plan_code_to"].map(price) - cells["tariff_plan_code_from"].map(price)) / median_price

    # --- 2. σ² внутри ячеек, регрессия по цене и τ² по сегментам ----------
    seg_params = {}
    for seg in SEGMENTS:
        hs = hist[hist["arpu_segment"] == seg]
        g = hs.groupby(["tariff_plan_code_from", "tariff_plan_code_to"])["pct"]
        dof = len(hs) - g.ngroups
        sigma2 = float((g.var(ddof=1).fillna(0) * (g.size() - 1)).sum() / max(dof, 1))
        cs = cells[(cells["arpu_segment"] == seg) & (cells["n"] >= MIN_CELL_N_FOR_TAU)]
        X = np.c_[np.ones(len(cs)), cs["dp"].values]
        sw = np.sqrt(cs["n_eff"].values)
        beta = np.linalg.lstsq(X * sw[:, None], cs["pct_rw"].values * sw, rcond=None)[0]
        resid = cs["pct_rw"].values - X @ beta
        v = sigma2 / cs["n_eff"].values
        tau2 = float(max(np.average(resid ** 2, weights=cs["n_eff"]) - np.average(v, weights=cs["n_eff"]), 0.0025))
        seg_params[seg] = {"sigma_within": math.sqrt(sigma2), "reg_intercept": float(beta[0]),
                           "reg_slope_dprice": float(beta[1]), "tau_between": math.sqrt(tau2),
                           "cells_used": int(len(cs))}

    # мок-модель (ровно как в mock_environment) — для справки/сверки
    conv_median_hist = float(cells["conv_hist"].median())

    # --- 3. аудитория: ячейки тариф × сегмент ------------------------------
    aud = (profile.dropna(subset=["current_tariff", "arpu_segment"])
           .groupby(["current_tariff", "arpu_segment"])
           .agg(n_customers=("ID_NUMBER", "size"), arpu_sum=("predicted_arpu", "sum"),
                arpu_mean=("predicted_arpu", "mean"), arpu_median=("predicted_arpu", "median"),
                zero_arpu=("predicted_arpu", lambda x: int((x <= 0).sum())))
           .reset_index().sort_values("n_customers", ascending=False))
    aud["share_customers"] = aud["n_customers"] / len(profile)
    aud["share_arpu"] = aud["arpu_sum"] / profile["predicted_arpu"].sum()
    hist_from = totals.rename(columns={"tariff_plan_code_from": "current_tariff", "N_from": "history_switches"})
    aud = aud.merge(hist_from[["current_tariff", "arpu_segment", "history_switches"]],
                    on=["current_tariff", "arpu_segment"], how="left").fillna({"history_switches": 0})

    # --- 4. априор для каждой ячейки аудитории × целевой тариф ------------
    cells_idx = cells.set_index(["tariff_plan_code_from", "tariff_plan_code_to", "arpu_segment"])
    totals_idx = totals.set_index(["tariff_plan_code_from", "arpu_segment"])
    seg_target_share = (hist.groupby(["arpu_segment", "tariff_plan_code_to"])["w"].sum()
                        / hist.groupby("arpu_segment")["w"].sum())
    rows = []
    for _, c in aud.iterrows():
        f, s = c["current_tariff"], c["arpu_segment"]
        sp = seg_params[s]
        sigma2, tau2 = sp["sigma_within"] ** 2, sp["tau_between"] ** 2
        N_from = float(totals_idx.loc[(f, s), "SW_from"]) if (f, s) in totals_idx.index else 0.0
        # доли целевых тарифов в сегменте без самого f — база для сглаживания conv
        base_share = seg_target_share.loc[s].drop(f, errors="ignore")
        base_share = base_share / base_share.sum()
        for t in all_targets:
            if t == f:
                continue
            dp = (price[t] - price[f]) / median_price
            mu = sp["reg_intercept"] + sp["reg_slope_dprice"] * dp
            fb = mock_fallback_pct(price[f], price[t], median_price)
            key = (f, t, s)
            if key in cells_idx.index:
                cr = cells_idx.loc[key]
                v = sigma2 / float(cr["n_eff"])
                post_var = 1.0 / (1.0 / v + 1.0 / tau2)
                pct_prior = post_var * (float(cr["pct_rw"]) / v + mu / tau2)
                pct_sd = math.sqrt(post_var)
                n, pct_hist, conv_hist = int(cr["n"]), float(cr["pct_hist"]), float(cr["conv_hist"])
                sw = float(cr["sw"])
                src = "history" if n >= 20 else "history_sparse"
            else:
                n, pct_hist, conv_hist, sw = 0, np.nan, np.nan, 0.0
                if t in seen_targets:
                    pct_prior, pct_sd, src = mu, math.sqrt(tau2), "no_history_for_cell"
                else:
                    # цель ни разу не встречалась в истории: среда возьмёт своё fallback-правило
                    pct_prior = mu
                    pct_sd = math.sqrt(tau2 + (mu - fb) ** 2)
                    src = "unseen_target"
            pct_prior = float(np.clip(pct_prior, *PCT_CLIP))

            if src.startswith("history"):
                p0 = float(base_share.get(t, 0.0))
                conv = (sw + CONV_ALPHA * p0) / (N_from + CONV_ALPHA)
                conv_sd = math.sqrt(conv * (1 - conv) / (N_from + CONV_ALPHA + 1))
                k_sd = CONV_SCALE_SD
            else:
                # перехода нет в таблице эффектов → среда берёт медианную конверсию (fallback)
                conv, conv_sd, k_sd = conv_median_hist, 0.0, CONV_SCALE_SD_FALLBACK

            pct_sd_total = math.sqrt(pct_sd ** 2 + SHIFT_SD ** 2)
            # base = pct * conv * κ, κ ~ LogNormal(0, k_sd); conv с выборочной ошибкой conv_sd
            e_k, e_k2 = math.exp(k_sd ** 2 / 2), math.exp(2 * k_sd ** 2)
            e_conv, e_conv2 = conv * e_k, (conv ** 2 + conv_sd ** 2) * e_k2
            base = pct_prior * e_conv
            base_var = (pct_prior ** 2 + pct_sd_total ** 2) * e_conv2 - base ** 2
            base_sd = math.sqrt(max(base_var, 1e-12))
            row = {"current_tariff": f, "arpu_segment": s, "target_tariff": t, "source": src,
                   "n_history": n, "pct_hist_raw": pct_hist, "conv_hist_raw": conv_hist,
                   "pct_prior": pct_prior, "pct_sd": pct_sd, "pct_sd_total": pct_sd_total,
                   "conv_prior": conv, "conv_sd": conv_sd,
                   "conv_expected": e_conv, "base_prior": base, "base_sd": base_sd,
                   "dprice": dp, "pct_mock_fallback": fb,
                   "p_positive": norm_cdf(pct_prior / pct_sd_total),
                   "n_customers": int(c["n_customers"]), "arpu_mean": float(c["arpu_mean"])}
            for ch, prm in CHANNELS.items():
                m = prm["conversion_multiplier"]
                ratio = pct_prior * min(e_conv * m, 1.0)
                ratio_sd = base_sd * m
                row[f"ratio_{ch}"] = ratio
                row[f"ratio_sd_{ch}"] = ratio_sd
                row[f"ev_contact_{ch}"] = ratio * c["arpu_mean"] - prm["cost_per_contact"]
                # сколько абонентов в пилоте на этом канале дают столько же информации, сколько априор
                row[f"prior_equiv_pilot_n_{ch}"] = (PILOT_NOISE_STD / ratio_sd) ** 2
            best_ch = max(CHANNELS, key=lambda ch: row[f"ev_contact_{ch}"])
            row["best_channel_prior"] = best_ch
            row["best_ev_contact_prior"] = row[f"ev_contact_{best_ch}"]
            rows.append(row)

    prior = pd.DataFrame(rows)

    # --- 5. короткий список гипотез для пилотов ---------------------------
    # ожидаемая польза проверки (expected improvement над нулём) для SMS × размер ячейки:
    # EI = σ·(zΦ(z) + φ(z)), z = μ/σ, где μ, σ — априорная ценность контакта. Большая неопределённость
    # и крупная дорогая ячейка → стоит пилотировать раньше.
    cand = prior[prior["n_customers"] >= 100].copy()
    mu = cand["ratio_sms"] * cand["arpu_mean"] - CHANNELS["sms"]["cost_per_contact"]
    sd = cand["ratio_sd_sms"] * cand["arpu_mean"]
    z = mu / sd
    pdf = np.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
    cdf = z.map(norm_cdf)
    cand["ei_contact_sms"] = sd * (z * cdf + pdf)
    cand["explore_score"] = cand["ei_contact_sms"] * cand["n_customers"].clip(upper=5000)
    cand["prior_value_sms"] = cand["ev_contact_sms"] * cand["n_customers"].clip(upper=5000)
    cand = cand.sort_values("explore_score", ascending=False)
    cand["rank_in_cell"] = cand.groupby(["current_tariff", "arpu_segment"]).cumcount() + 1
    shortlist = cand[cand["rank_in_cell"] <= 3]
    keep = ["current_tariff", "arpu_segment", "target_tariff", "rank_in_cell", "source", "n_history",
            "n_customers", "arpu_mean", "pct_prior", "pct_sd_total", "conv_prior", "p_positive",
            "base_prior", "base_sd", "ratio_sms", "ratio_sd_sms", "ev_contact_sms",
            "best_channel_prior", "best_ev_contact_prior", "prior_value_sms", "ei_contact_sms", "explore_score"]
    shortlist = shortlist[keep]

    # --- 6. сохранение -----------------------------------------------------
    prior.sort_values(["n_customers", "current_tariff", "arpu_segment", "best_ev_contact_prior"],
                      ascending=[False, True, True, False]).to_csv(prior_dir / "prior_effects.csv", index=False)
    shortlist.to_csv(prior_dir / "pilot_candidates.csv", index=False)
    aud.to_csv(prior_dir / "audience_cells.csv", index=False)
    meta = {
        "description": "Априор эффектов смены тарифа из истории (другая выборка абонентов). "
                       "pct_sd — выборочная неопределённость, pct_sd_total — с запасом SHIFT_SD на сдвиг.",
        "shift_sd": SHIFT_SD, "conv_scale_sd": CONV_SCALE_SD, "conv_scale_sd_fallback": CONV_SCALE_SD_FALLBACK,
        "conv_alpha": CONV_ALPHA, "pilot_noise_std": PILOT_NOISE_STD,
        "median_price": median_price, "conv_median_history": conv_median_hist,
        "segment_params": seg_params, "reweighting": reweight_info,
        "targets_seen_in_history": sorted(seen_targets, key=lambda t: int(t.split("_")[1])),
        "targets_unseen_in_history": [t for t in all_targets if t not in seen_targets],
        "rows": int(len(prior)), "audience_cells": int(len(aud)),
        "source_counts": prior["source"].value_counts().to_dict(),
    }
    write_json(meta, prior_dir / "prior_meta.json")
    write_prior_module(prior, aud, meta, prior_dir / "prior_table.py")

    print(f"Ячеек аудитории: {len(aud)}, строк априора: {len(prior):,}")
    print("Источники:", meta["source_counts"])
    for s, p in seg_params.items():
        print(f"  {s}: σ={p['sigma_within']:.3f} τ={p['tau_between']:.3f} "
              f"pct ≈ {p['reg_intercept']:+.3f} {p['reg_slope_dprice']:+.3f}·Δprice")
    print(f"Сохранено в {prior_dir}")


def write_prior_module(prior: pd.DataFrame, aud: pd.DataFrame, meta: dict, path: Path) -> None:
    """Самодостаточный python-модуль: можно вставить в agent.py целиком (сдаётся только agent.py)."""
    def r(x, k=5):
        return None if (x is None or (isinstance(x, float) and not np.isfinite(x))) else round(float(x), k)

    lines = [
        '"""Автосгенерировано analytics/build_prior.py — не редактировать руками.',
        "",
        "PRIOR[(current_tariff, arpu_segment, target_tariff)] =",
        "    (pct, pct_sd_total, conv, conv_sd, base, base_sd, n_history, source)",
        "base — ожидаемый эффект до множителя канала (pct * conv * κ), base_sd — его неопределённость",
        "с запасом на сдвиг pct (SHIFT_SD) и неизвестный масштаб конверсии (κ, CONV_SCALE_SD).",
        "CELLS[(current_tariff, arpu_segment)] = (n_customers, arpu_mean, arpu_sum, history_switches)",
        "",
        "Эффект кампании в среде: pct * min(conv * m_channel, 1) * predicted_arpu.",
        "Это ИСТОРИЯ другой выборки: использовать как стартовую оценку и уточнять пилотами.",
        '"""',
        "",
        f"SHIFT_SD = {meta['shift_sd']}",
        f"CONV_SCALE_SD = {meta['conv_scale_sd']}",
        f"PILOT_NOISE_STD = {meta['pilot_noise_std']}",
        "CHANNEL_MULT = {" + ", ".join(f'"{k}": {v["conversion_multiplier"]}' for k, v in CHANNELS.items()) + "}",
        "CHANNEL_COST = {" + ", ".join(f'"{k}": {v["cost_per_contact"]}' for k, v in CHANNELS.items()) + "}",
        "",
        "PRIOR = {",
    ]
    for _, p in prior.iterrows():
        lines.append(f"    ({p.current_tariff!r}, {p.arpu_segment!r}, {p.target_tariff!r}): "
                     f"({r(p.pct_prior)}, {r(p.pct_sd_total)}, {r(p.conv_prior)}, {r(p.conv_sd)}, "
                     f"{r(p.base_prior)}, {r(p.base_sd)}, {int(p.n_history)}, {p.source!r}),")
    lines.append("}")
    lines.append("")
    lines.append("CELLS = {")
    for _, c in aud.iterrows():
        lines.append(f"    ({c.current_tariff!r}, {c.arpu_segment!r}): ({int(c.n_customers)}, "
                     f"{r(c.arpu_mean, 2)}, {r(c.arpu_sum, 0)}, {int(c.history_switches)}),")
    lines.append("}")
    lines += [
        "",
        "",
        "def prior_ratio(current_tariff, arpu_segment, target_tariff, channel):",
        '    """Априорный относительный эффект кампании (mean, sd) для канала; None, если комбинации нет."""',
        "    rec = PRIOR.get((current_tariff, arpu_segment, target_tariff))",
        "    if rec is None:",
        "        return None",
        "    _, _, _, _, base, base_sd, _, _ = rec",
        "    m = CHANNEL_MULT[channel]",
        "    return base * m, base_sd * m",
        "",
        "",
        "def posterior_base(current_tariff, arpu_segment, target_tariff, pilots, max_equiv_n=None):",
        '    """Нормальное обновление base по пилотам: pilots = [(channel, n, observed_lift_ratio), ...].',
        "    Наблюдение пилота: observed = m_channel * base + N(0, PILOT_NOISE_STD / sqrt(n)).",
        "    max_equiv_n — ограничить вес априора пилотом на столько абонентов (в единицах call-канала m=1).",
        "    Пилот по смеси ячеек (без filter_current_tariff) сюда подавать нельзя — он усредняет разные эффекты.\"\"\"",
        "    rec = PRIOR.get((current_tariff, arpu_segment, target_tariff))",
        "    mean, sd = (rec[4], rec[5]) if rec else (0.0, 0.25)",
        "    if max_equiv_n:",
        "        sd = max(sd, PILOT_NOISE_STD / max_equiv_n ** 0.5)",
        "    prec, num = 1.0 / sd ** 2, mean / sd ** 2",
        "    for channel, n, observed in pilots:",
        "        m = CHANNEL_MULT[channel]",
        "        w = n * m * m / PILOT_NOISE_STD ** 2",
        "        prec += w",
        "        num += w * observed / m",
        "    return num / prec, (1.0 / prec) ** 0.5",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
