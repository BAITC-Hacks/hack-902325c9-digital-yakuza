"""
Осторожная оценка для 12 тарифов, на которые в истории нет ни одного перехода
(tariff_2, 3, 5, 6, 7, 14, 15, 16, 17, 18, 19, 20).

Правило «подходит ли пакет ячейке» (ячейка = текущий тариф × ARPU-сегмент аудитории):
  * data_cover — доля абонентов ячейки, чей трафик (DATA_VOLUME) помещается в пакет тарифа;
  * min_cover  — доля, чьи минуты на других операторов (OUT_LOC_OFFNET_MIN) помещаются в пакет;
  * fit = (data_cover + min_cover) / 2; сравниваем с тем же показателем текущего тарифа;
  * метка: «подходит» (fit ≥ 0.8 и не хуже текущего), «частично» (fit ≥ 0.5), «не подходит».

Оценка q (эффект до множителя канала) — две независимые прикидки, берём меньшую:
  1. правило: Δ% из регрессии истории «Δ% ~ разница цен» для сегмента × доля перехода
     (медианная доля перехода истории × fit — пакет, который закрывает потребление, берут охотнее);
  2. аналог: q ближайшего по пакету и цене тарифа, на который история есть
     (tariff_5/6/7 — копии tariff_8, tariff_16 — почти tariff_9 и т.д.), с поправкой на цену.
Положительную оценку делим пополам: пока пилот не подтвердил, половина эффекта может не случиться.
n_obs = 0, разброс pct_std — неопределённость Δ% такой связки: межъячеечный разброс сегмента
плюс расхождение Δ% по правилу и по аналогу (в тех же единицах, что pct_std в PRIOR).
"""
import numpy as np
import pandas as pd

UNSEEN_CAUTION = 0.5        # доля положительного эффекта, которую закладываем без пилота


def _package(tariffs: pd.DataFrame) -> pd.DataFrame:
    t = tariffs.set_index("tariff_plan_code")
    return pd.DataFrame({"data": t["Data_in_PKG"].astype(float),
                         "minutes": (t["Min_another_operator_in_PKG"]
                                     + t["Min_another_operator_and_city_in_PKG"]).astype(float),
                         "price": t["price_tariff"].astype(float)})


def nearest_seen(pkg: pd.DataFrame, target: str, seen: list) -> str:
    """Ближайший тариф с историей по пакету (лог-трафик, минуты) и цене."""
    feats = pd.DataFrame({"ldata": np.log2(1 + pkg["data"]), "minutes": pkg["minutes"], "price": pkg["price"]})
    z = (feats - feats.mean()) / feats.std()
    d = (z.loc[seen] - z.loc[target]).abs().sum(axis=1)
    return str(d.idxmin())


def build_unseen(prior: pd.DataFrame, segments: dict, median_price: float, profile: pd.DataFrame,
                 tariffs: pd.DataFrame, conv_median: float) -> pd.DataFrame:
    pkg = _package(tariffs)
    seen = sorted(set(prior["tariff_to"]))
    unseen = [t for t in pkg.index if t not in seen]
    seen_idx = prior.set_index(["tariff_from", "arpu_segment", "tariff_to"])[["pct_eb", "conversion"]]
    prof = profile.dropna(subset=["current_tariff", "arpu_segment"])
    rows = []
    for (f, s), cell in prof.groupby(["current_tariff", "arpu_segment"]):
        data = cell["DATA_VOLUME"].dropna()
        mins = cell["OUT_LOC_OFFNET_MIN"].dropna()

        def fit_of(t):
            dc = float((data <= pkg.at[t, "data"]).mean()) if len(data) else 0.0
            mc = float((mins <= pkg.at[t, "minutes"]).mean()) if len(mins) else 0.0
            return dc, mc, (dc + mc) / 2

        _, _, fit_cur = fit_of(f) if f in pkg.index else (0, 0, 0)
        par = segments[s]
        for t in unseen:
            if t == f:
                continue
            dc, mc, fit = fit_of(t)
            gain = fit - fit_cur
            label = "подходит" if (fit >= 0.8 and gain >= -0.05) else ("частично" if fit >= 0.5 else "не подходит")
            dprice = (pkg.at[t, "price"] - pkg.at[f, "price"]) / median_price
            pct_rule = par["a"] + par["b"] * dprice
            conv_rule = conv_median * fit
            q_rule = pct_rule * conv_rule
            analog = nearest_seen(pkg, t, seen)
            q_analog, pct_analog = np.nan, np.nan
            if (f, s, analog) in seen_idx.index:
                # аналог: тот же переход, что на похожий тариф, Δ% сдвинут по наклону регрессии на разницу цен
                pct_a, conv_a = seen_idx.loc[(f, s, analog)]
                shift = par["b"] * (pkg.at[t, "price"] - pkg.at[analog, "price"]) / median_price
                pct_analog = pct_a + shift
                q_analog = pct_analog * conv_a
            q_min = q_rule if np.isnan(q_analog) else min(q_rule, q_analog)
            q = q_min * UNSEEN_CAUTION if q_min > 0 else q_min
            # неопределённость Δ%: межъячеечный разброс сегмента + расхождение Δ% правила и аналога
            disagreement = 0.0 if np.isnan(pct_analog) else abs(pct_rule - pct_analog)
            pct_std = float(np.sqrt(par["tau"] ** 2 + disagreement ** 2))
            rows.append({"tariff_from": f, "arpu_segment": s, "tariff_to": t, "n_customers": int(len(cell)),
                         "fit_label": label, "fit": round(fit, 3), "fit_current": round(fit_cur, 3),
                         "data_cover": round(dc, 3), "min_cover": round(mc, 3),
                         "dprice": dprice, "pct_rule": pct_rule, "conv_rule": conv_rule, "q_rule": q_rule,
                         "analog_tariff": analog, "pct_analog": pct_analog, "q_analog": q_analog,
                         "q": q, "n_obs": 0, "pct_std": pct_std})
    return pd.DataFrame(rows).sort_values(["tariff_from", "arpu_segment", "tariff_to"]).reset_index(drop=True)
