"""
Соответствие пакета тарифа потреблению ячейки аудитории (для троек без истории).

Для ячейки «текущий тариф × ARPU-сегмент» и целевого тарифа:
  * data_cover — доля абонентов ячейки, чей трафик (DATA_VOLUME) помещается в пакет тарифа;
  * min_cover  — доля, чьи минуты на других операторов (OUT_LOC_OFFNET_MIN) помещаются в пакет;
  * fit = (data_cover + min_cover) / 2, сравнивается с тем же показателем текущего тарифа;
  * метка: «подходит» (fit ≥ 0.8 и не хуже текущего), «частично» (fit ≥ 0.5), «не подходит».

Как это используется (prior/build_prior.py): у троек без истории доля согласившихся берётся
как у fallback среды (медианная доля перехода) и умножается на (0.5 + 0.5·fit) — пакет,
который не закрывает потребление, берут вдвое реже. Проверка на истории: абоненты с перерасходом
трафика в 75% случаев уходят на тариф с бОльшим пакетом данных (без перерасхода — 44%);
но такие «переходы по потреблению» дают на 0.10 меньший Δ% ARPU, чем остальные переходы из той же
страты (переплата за перерасход исчезает) — поэтому fit влияет на отклик, а не на Δ%.
"""
import pandas as pd


def _package(tariffs: pd.DataFrame) -> pd.DataFrame:
    t = tariffs.set_index("tariff_plan_code")
    return pd.DataFrame({"data": t["Data_in_PKG"].astype(float),
                         "minutes": (t["Min_another_operator_in_PKG"]
                                     + t["Min_another_operator_and_city_in_PKG"]).astype(float)})


def package_fit(profile: pd.DataFrame, tariffs: pd.DataFrame) -> pd.DataFrame:
    """Таблица (tariff_from, arpu_segment, tariff_to) → data_cover, min_cover, fit, fit_current, fit_label."""
    pkg = _package(tariffs)
    prof = profile.dropna(subset=["current_tariff", "arpu_segment"])
    rows = []
    for (f, s), cell in prof.groupby(["current_tariff", "arpu_segment"]):
        data = cell["DATA_VOLUME"].dropna().values
        mins = cell["OUT_LOC_OFFNET_MIN"].dropna().values

        def cover(t):
            dc = float((data <= pkg.at[t, "data"]).mean()) if len(data) else 0.0
            mc = float((mins <= pkg.at[t, "minutes"]).mean()) if len(mins) else 0.0
            return dc, mc, (dc + mc) / 2

        fit_cur = cover(f)[2] if f in pkg.index else 0.0
        for t in pkg.index:
            if t == f:
                continue
            dc, mc, fit = cover(t)
            label = "подходит" if (fit >= 0.8 and fit - fit_cur >= -0.05) else ("частично" if fit >= 0.5 else "не подходит")
            rows.append({"tariff_from": f, "arpu_segment": s, "tariff_to": t, "data_cover": round(dc, 3),
                         "min_cover": round(mc, 3), "fit": round(fit, 3), "fit_current": round(fit_cur, 3),
                         "fit_label": label})
    return pd.DataFrame(rows)
