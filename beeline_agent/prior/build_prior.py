"""
Априор из истории смен тарифов → таблицы, вшитые в agent.py.

    python prior/build_prior.py            # пересчитать prior/*.csv и вшить словари в agent.py

Агент читает только вшитые константы, файлы data/ во время работы не открываются.
Правила и обоснование — prior/README.md.

Вшивается (между маркерами PRIOR START / PRIOR END), формат согласован с агентом:
  PRIOR_FORMAT = ("q", "q_se", "n_obs")
  PRIOR[(from, seg, to)]        = (q, q_se, n_obs)  — тройки с историей; что именно q — PRIOR_MODE ниже
  PRIOR_UNSEEN[(from, seg, to)] = (q, q_se, 0)      — тройки ячеек аудитории без истории
                                                      (12 тарифов без истории + пробелы по ячейкам)
      q     — база эффекта: Δ% × доля перехода, до множителя канала (доля от ARPU абонента);
      q_se  — стандартная ошибка оценки q по истории, в единицах q; размер выборки уже учтён —
              агент берёт как есть и на √n не делит; перенос «история → аудитория» — отдельно,
              TRANSFER_SD агента: sd = √(q_se² + TRANSFER_SD²);
      n_obs — число наблюдений, только для справки.
  PRIOR_PARTS[(from, seg, to)]  = (pct, conv)       — из чего сложен q: чтобы разброс переноса масштабировать
                                                      конверсией (TRANSFER_SD_PCT × conv): редкий переход не может
                                                      дать большой эффект
  PRIOR_SCORE[(from, seg, to)]  = (q_level, q_contrast, q_se) — разложение q для «структурного» постериора
      q_level    — часть эффекта от уровня страты (в основном регрессия к среднему, переносится хуже);
      q_contrast — чем эта цель лучше средней цели страты (надёжная часть, split-half 0.92–0.96);
      q = q_level + q_contrast, q_se — стандартная ошибка q по истории.
  q = оценка Δ% × доля перехода — база эффекта без множителя канала; в пилоте на канале c
  среда вернёт примерно m_c · q + шум 0.804/√n. Это не деньги, а ранжирование и стартовая оценка.

Как считается:
  * чистка — prior/history.py (дубли, ARPU до < 100, клип Δ% в [-1; 3] как в среде);
  * Δ% = уровень страты (from, seg) + контраст цели (иерархический EB: цель → сегмент×цель → ячейка),
    prior/hier.py; на split-half ошибка ниже, чем у простого среднего и у сжатия к ценовой регрессии;
  * доля перехода: у троек с историей — сглаженная к структуре переходов сегмента (α = 10);
    у троек без истории — медианная (так делает fallback среды) × (0.5 + 0.5·fit пакета);
  * pct_std — внутригрупповой разброс Δ% (для малых групп подтянут к разбросу сегмента);
  * пороги каналов (prior/breakeven.csv) — при каком q канал окупается сам и против канала дешевле.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hier  # noqa: E402
from history import KEYS, ROOT, load_history, pooled_sigma2  # noqa: E402
from unseen import package_fit  # noqa: E402

PRIOR_MODE = "raw"           # что кладём в PRIOR (его читает текущий агент):
                             #   "raw"  — среднее Δ% × доля перехода, как в моке (лучше в связке с текущим агентом,
                             #            см. prior/README.md, «A/B на мирах»);
                             #   "hier" — уровень страты + контраст цели (точнее по истории, для структурного постериора)
PRIOR_TUPLE = "q_se"         # "q_se" — (q, q_se, n_obs), договорённость с агентом; "legacy" — старый (q, n_obs, pct_std)
CONV_ALPHA = 10.0            # сила сглаживания доли переходов
UNSEEN_CONV_UNC = 0.5        # у троек без истории доля перехода — допущение (fallback × пакет): ±50%
VAR_PRIOR_DOF = 5            # сколько «наблюдений» весит разброс сегмента при оценке pct_std
BUDGET_VALUE = 10.0          # у.е. прироста, которые приносит 1 у.е. бюджета в финальном плане (цена бюджета)
CHANNELS = [("push", 0.0, 0.50), ("sms", 4.0, 0.65), ("digital_ads", 22.0, 0.85), ("call", 160.0, 1.20)]
START, END = "# === PRIOR START (генерируется prior/build_prior.py) ===", "# === PRIOR END ==="


def _conversion(hist: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    totals = cells.groupby(["tariff_from", "arpu_segment"])["n_obs"].transform("sum")
    seg_share = hist.groupby(["arpu_segment", "tariff_to"]).size() / hist.groupby("arpu_segment").size()
    p0 = np.array([seg_share.get((s, t), 0.0) for s, t in zip(cells["arpu_segment"], cells["tariff_to"])])
    cells["conversion_raw"] = cells["n_obs"] / totals
    cells["conversion"] = (cells["n_obs"] + CONV_ALPHA * p0) / (totals + CONV_ALPHA)
    return cells


def _decompose(df: pd.DataFrame, model: dict) -> pd.DataFrame:
    pred = pd.DataFrame([hier.predict(model, f, s, t)
                         for f, s, t in zip(df["tariff_from"], df["arpu_segment"], df["tariff_to"])], index=df.index)
    df = df.join(pred.rename(columns={"pct": "pct_hat", "source": "source"}))
    df["q"] = df["pct_hat"] * df["conversion"]
    df["q_level"] = df["level"] * df["conversion"]
    df["q_contrast"] = df["contrast"] * df["conversion"]
    df["q_sd"] = df["pct_sd"] * df["conversion"]
    return df


def build(hist: pd.DataFrame, tariffs: pd.DataFrame, profile: pd.DataFrame):
    model = hier.fit(hist, tariffs)

    # --- тройки с историей ---------------------------------------------------------------
    g = hist.groupby(KEYS)["pct"]
    seen = g.agg(n_obs="size", pct_mean="mean", pct_std_raw="std").reset_index()
    seen = _conversion(hist, seen)
    sigma2 = pooled_sigma2(hist)
    dof = (seen["n_obs"] - 1).clip(lower=0)
    seg_var = seen["arpu_segment"].map(sigma2)
    seen["pct_std"] = np.sqrt((dof * seen["pct_std_raw"].fillna(0) ** 2 + VAR_PRIOR_DOF * seg_var)
                              / (dof + VAR_PRIOR_DOF))
    seen = _decompose(seen, model)
    seen["q_hier"] = seen["q"]
    seen["q_raw"] = seen["pct_mean"] * seen["conversion_raw"]           # как считает мок
    seen["pct_std_raw"] = seen["pct_std_raw"].fillna(seen["pct_std_raw"].median())
    seen["reliability"] = np.select([seen["n_obs"] >= 20, seen["n_obs"] >= 6], ["high", "medium"], "low")
    totals = seen.groupby(["tariff_from", "arpu_segment"])["n_obs"].transform("sum")
    # стандартная ошибка q = pct × conv (дельта-метод), в единицах q, размер выборки уже учтён
    se_pct_raw = np.sqrt(seg_var) / np.sqrt(seen["n_obs"])                  # ошибка сырого среднего Δ%
    se_pct_raw = np.where(seen["n_obs"] > 1, seen["pct_std_raw"] / np.sqrt(seen["n_obs"]), se_pct_raw)
    conv_se_raw = np.sqrt(seen["conversion_raw"] * (1 - seen["conversion_raw"]) / totals)
    seen["q_se_raw"] = np.sqrt((seen["conversion_raw"] * se_pct_raw) ** 2 + (seen["pct_mean"] * conv_se_raw) ** 2)
    conv_se = np.sqrt(seen["conversion"] * (1 - seen["conversion"]) / (totals + CONV_ALPHA + 1))
    seen["q_se_hier"] = np.sqrt((seen["conversion"] * seen["pct_sd"]) ** 2 + (seen["pct_hat"] * conv_se) ** 2)
    if PRIOR_MODE == "raw":
        seen["prior_q"], seen["prior_std"], seen["prior_se"] = seen["q_raw"], seen["pct_std_raw"], seen["q_se_raw"]
        seen["prior_pct"], seen["prior_conv"] = seen["pct_mean"], seen["conversion_raw"]
    else:
        seen["prior_q"], seen["prior_std"], seen["prior_se"] = seen["q_hier"], seen["pct_std"], seen["q_se_hier"]
        seen["prior_pct"], seen["prior_conv"] = seen["pct_hat"], seen["conversion"]
    seen["q_se"] = seen["q_se_hier"]

    # --- тройки ячеек аудитории без истории -------------------------------------------------
    fit = package_fit(profile, tariffs)
    known = set(zip(seen["tariff_from"], seen["arpu_segment"], seen["tariff_to"]))
    unseen = fit[[k not in known for k in zip(fit["tariff_from"], fit["arpu_segment"], fit["tariff_to"])]].copy()
    conv_fallback = float(seen["conversion_raw"].median())
    unseen["n_obs"] = 0
    unseen["conversion"] = conv_fallback * (0.5 + 0.5 * unseen["fit"])
    unseen = _decompose(unseen.reset_index(drop=True), model)
    unseen["pct_std"] = unseen["pct_sd"]
    unseen["q_se"] = np.sqrt((unseen["conversion"] * unseen["pct_sd"]) ** 2
                             + (unseen["pct_hat"] * UNSEEN_CONV_UNC * unseen["conversion"]) ** 2)
    unseen["prior_q"], unseen["prior_se"], unseen["prior_std"] = unseen["q"], unseen["q_se"], unseen["pct_sd"]
    unseen["prior_pct"], unseen["prior_conv"] = unseen["pct_hat"], unseen["conversion"]

    # --- пороги каналов по ячейкам аудитории --------------------------------------------------
    be = (profile.dropna(subset=["current_tariff", "arpu_segment"])
          .groupby(["current_tariff", "arpu_segment"])["predicted_arpu"]
          .agg(n_customers="size", arpu_mean="mean").reset_index()
          .rename(columns={"current_tariff": "tariff_from"}))
    for i, (ch, cost, m) in enumerate(CHANNELS):
        # окупается сам: m·q·A > cost
        be[f"be_{ch}"] = cost / (m * be["arpu_mean"].clip(lower=1.0))
        if i > 0:
            pch, pcost, pm = CHANNELS[i - 1]
            # лучше канала дешевле с учётом цены бюджета: (m - m_prev)·q·A > (cost - cost_prev)·BUDGET_VALUE
            be[f"be_{ch}_vs_{pch}"] = (cost - pcost) * BUDGET_VALUE / ((m - pm) * be["arpu_mean"].clip(lower=1.0))

    meta = {"level": model["level_params"], "tau_contrast": model["tau"], "transfer": model["transfer"],
            "target_contrast": model["ct"].round(4).to_dict(), "conv_fallback": conv_fallback,
            "budget_value": BUDGET_VALUE}
    return seen, unseen, be, meta


def _dict_lines(name: str, table: pd.DataFrame, cols: list, comment: str) -> list:
    lines = [comment, f"{name} = {{"]
    for r in table.sort_values(KEYS).itertuples():
        vals = []
        for c in cols:
            v = getattr(r, c)
            vals.append(str(int(v)) if c == "n_obs" else f"{float(v):.5f}")
        lines.append(f'    ("{r.tariff_from}", "{r.arpu_segment}", "{r.tariff_to}"): ({", ".join(vals)}),')
    lines.append("}")
    return lines


def embed(seen: pd.DataFrame, unseen: pd.DataFrame, agent_path: Path) -> None:
    both = pd.concat([seen, unseen], ignore_index=True)
    if PRIOR_TUPLE == "legacy":
        fmt, cols = ("q", "n_obs", "pct_std"), ["prior_q", "n_obs", "prior_std"]
    else:
        fmt, cols = ("q", "q_se", "n_obs"), ["prior_q", "prior_se", "n_obs"]
    lines = [START, f"PRIOR_FORMAT = {fmt!r}",
             f"PRIOR_MODE = {PRIOR_MODE!r}   # raw — q как в моке; hier — уровень страты + контраст цели"]
    lines += _dict_lines("PRIOR", seen, cols, f"# (from, seg, to) -> {fmt}: история смен тарифов")
    lines += _dict_lines("PRIOR_UNSEEN", unseen, cols, f"# тройки ячеек аудитории без истории -> {fmt}, n_obs = 0")
    lines += _dict_lines("PRIOR_PARTS", both, ["prior_pct", "prior_conv"],
                         "# (from, seg, to) -> (pct, conv): q = pct × conv; conv — для масштаба неопределённости")
    lines += _dict_lines("PRIOR_SCORE", both, ["q_level", "q_contrast", "q_se"],
                         "# (from, seg, to) -> (q_level, q_contrast, q_se): уровень страты + контраст цели, единицы q")
    lines.append(END)
    text = agent_path.read_text(encoding="utf-8")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    agent_path.write_text(head + "\n".join(lines) + tail, encoding="utf-8")


def main() -> None:
    hist, clean_report = load_history()
    tariffs = pd.read_csv(ROOT / "data" / "dict_tariff.csv")
    profile = pd.read_csv(ROOT / "customer_profile.csv")
    seen, unseen, be, meta = build(hist, tariffs, profile)

    cols = ["tariff_from", "arpu_segment", "tariff_to", "n_obs", "reliability", "source", "pct_mean", "level",
            "level_sd", "contrast", "contrast_sd", "pct_hat", "pct_sd", "pct_std", "pct_std_raw", "conversion_raw",
            "conversion", "q_raw", "q_se_raw", "q_hier", "q_se_hier", "q_level", "q_contrast", "prior_q", "prior_se"]
    seen[cols].to_csv(ROOT / "prior" / "prior_table.csv", index=False)
    ucols = ["tariff_from", "arpu_segment", "tariff_to", "source", "fit_label", "fit", "fit_current", "data_cover",
             "min_cover", "level", "contrast", "pct_hat", "pct_sd", "conversion", "q", "q_se", "q_level", "q_contrast"]
    unseen[ucols].to_csv(ROOT / "prior" / "prior_unseen.csv", index=False)
    be.round(5).to_csv(ROOT / "prior" / "breakeven.csv", index=False)
    (ROOT / "prior" / "reports").mkdir(exist_ok=True)
    (ROOT / "prior" / "reports" / "prior_meta.json").write_text(
        json.dumps({"cleaning": clean_report, **meta}, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    embed(seen, unseen, ROOT / "agent.py")

    print("Чистка change_tariff:", ", ".join(f"{k}={v}" for k, v in clean_report.items()))
    lp = meta["level"]
    print("Уровень страты (среднее по сегменту, τ): " +
          ", ".join(f"{s} {lp[s]['mean']:+.3f} (τ {lp[s]['tau']:.3f})" for s in ("LOW", "MID", "HIGH")))
    print("Контраст цели: " + ", ".join(f"{t.split('_')[1]}:{v:+.2f}" for t, v in
                                         sorted(meta["target_contrast"].items(), key=lambda x: -x[1])))
    tr = meta["transfer"]
    print(f"Перенос контраста на 12 тарифов без истории по атрибутам: LOO RMSE {tr['kernel_loo_rmse']:.3f} "
          f"против {tr['zero_rmse']:.3f} у нуля → метод: {tr['method']}")
    moved = (np.sign(seen["q_hier"]) != np.sign(seen["q_raw"])).sum()
    print(f"Формат: PRIOR_FORMAT = {('q', 'q_se', 'n_obs') if PRIOR_TUPLE != 'legacy' else ('q', 'n_obs', 'pct_std')}; "
          f"q_se медиана {seen['prior_se'].median():.4f} (PRIOR), {unseen['q_se'].median():.4f} (PRIOR_UNSEEN)")
    print(f"PRIOR (режим {PRIOR_MODE}): {len(seen)} троек, q>0 у {(seen['prior_q'] > 0).sum()}; "
          f"иерархическая оценка q>0 у {(seen['q_hier'] > 0).sum()}, знак отличается у {moved}; "
          f"PRIOR_UNSEEN: {len(unseen)} троек, q>0 у {(unseen['q'] > 0).sum()}; PRIOR_SCORE: {len(seen) + len(unseen)}; "
          f"пороги каналов: prior/breakeven.csv; вшито в agent.py")


if __name__ == "__main__":
    main()
