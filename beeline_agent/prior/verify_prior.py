"""
Независимая перепроверка приора: каждое число пересчитывается заново, без кода build_prior.py и hier.py.

    cd beeline_agent
    python prior/verify_prior.py        # код выхода 1, если хоть одна проверка не сошлась

Что проверяется:
  1. счётчики чистки из reports/prior_meta.json — пересчёт с нуля по data/change_tariff.csv;
  2. q в PRIOR (режим raw) = Δ% × доля сменивших, посчитанные функцией среды (_mock_impact_model);
  3. q_se в prior_table.csv / prior_unseen.csv — по формулам из prior/README.md;
  4. покрытие: каждая ячейка аудитории × 20 целей (кроме своего тарифа) есть ровно в одной таблице;
  5. таблицы, вшитые в agent.py, совпадают с CSV (если нет — build_prior.py не перезапускали после изменений;
     это предупреждение, а не ошибка: вшивку делает владелец agent.py).
"""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mock_environment import _mock_impact_model  # noqa: E402

KEYS = ["tariff_from", "arpu_segment", "tariff_to"]
TOL = 1e-9
failed = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"{'OK  ' if ok else 'FAIL'} {name}: {detail}")
    if not ok:
        failed.append(name)


def main() -> int:
    raw = pd.read_csv(ROOT / "data" / "change_tariff.csv")
    table = pd.read_csv(ROOT / "prior" / "prior_table.csv")
    unseen = pd.read_csv(ROOT / "prior" / "prior_unseen.csv")
    meta = json.loads((ROOT / "prior" / "reports" / "prior_meta.json").read_text(encoding="utf-8"))
    price = pd.read_csv(ROOT / "data" / "dict_tariff.csv").set_index("tariff_plan_code")["price_tariff"]

    # 1. чистка
    dedup = raw.drop_duplicates()
    prev = dedup["AVG_ARPU_PREV_3M"]
    used = dedup[prev >= 100]
    pct_raw = (used["AVG_ARPU_NEXT_3M"] - used["AVG_ARPU_PREV_3M"]) / used["AVG_ARPU_PREV_3M"]
    recount = {"rows_in_file": len(raw), "removed_full_duplicates": int(raw.duplicated().sum()),
               "prev_negative": int((prev < 0).sum()), "prev_zero": int((prev == 0).sum()),
               "prev_0_100": int(((prev > 0) & (prev < 100)).sum()), "rows_used": len(used),
               "next_zero_kept": int((used["AVG_ARPU_NEXT_3M"] == 0).sum()),
               "pct_clipped_at_+300%": int((pct_raw > 3).sum()),
               "pct_at_most_-99.9%_kept": int((pct_raw.clip(-1, 3) <= -0.999).sum())}
    bad = {k: (meta["cleaning"].get(k), v) for k, v in recount.items() if meta["cleaning"].get(k) != v}
    check("чистка", not bad, f"{len(recount)} счётчиков пересчитаны с нуля" + (f", расхождения: {bad}" if bad else ""))

    # 2. q режима raw = формула среды
    env = _mock_impact_model(raw).rename(columns={"tariff_plan_code_from": "tariff_from",
                                                   "tariff_plan_code_to": "tariff_to"})
    env["arpu_segment"] = env["arpu_segment"].astype(str)
    env["q_env"] = env["arpu_change_pct"] * env["conversion_rate"]
    j = table.merge(env[KEYS + ["q_env"]], on=KEYS, how="outer", indicator=True)
    both = j[j["_merge"] == "both"]
    diff = float((both["prior_q"] - both["q_env"]).abs().max())
    check("q = формула среды", len(both) == len(table) == len(env) and diff < TOL,
          f"{len(both)} троек, у среды {len(env)}, макс. |разница| {diff:.1e}")

    # 3. q_se по формулам README
    tot = table.groupby(["tariff_from", "arpu_segment"])["n_obs"].transform("sum")
    se_pct = table["pct_std"] / np.sqrt(table["n_obs"])
    conv_se = np.sqrt(table["conversion_raw"] * (1 - table["conversion_raw"]) / tot)
    q_se = np.sqrt((table["conversion_raw"] * se_pct) ** 2 + (table["pct_mean"] * conv_se) ** 2)
    d1 = float((q_se - table["prior_se"]).abs().max())
    fallback = (0.4 * (unseen["tariff_to"].map(price) - unseen["tariff_from"].map(price))
                / max(float(price.median()), 1.0)).clip(-1, 3)
    cv = table["conversion_raw"].std() / table["conversion_raw"].mean()
    unc = np.sqrt(unseen["pct_sd"] ** 2 + (unseen["pct_hat"] - fallback) ** 2)
    q_se_u = np.sqrt((unseen["conversion"] * unc) ** 2 + (unseen["pct_hat"] * cv * unseen["conversion"]) ** 2)
    d2 = float((q_se_u - unseen["q_se"]).abs().max())
    check("q_se по формулам", d1 < TOL and d2 < TOL,
          f"PRIOR макс. |разница| {d1:.1e} (медиана q_se {table['prior_se'].median():.4f}); "
          f"PRIOR_UNSEEN {d2:.1e} (медиана {unseen['q_se'].median():.4f}); правило среды для троек без истории — "
          f"расхождение {float((fallback - unseen['pct_fallback_rule']).abs().max()):.1e}")

    # 4. покрытие
    profile = pd.read_csv(ROOT / "customer_profile.csv")
    cells = profile.dropna(subset=["current_tariff", "arpu_segment"])[["current_tariff", "arpu_segment"]].drop_duplicates()
    need = {(f, s, t) for f, s in cells.itertuples(index=False) for t in price.index if t != f}
    seen_keys, unseen_keys = set(map(tuple, table[KEYS].values)), set(map(tuple, unseen[KEYS].values))
    check("покрытие", need <= (seen_keys | unseen_keys) and not (seen_keys & unseen_keys),
          f"{len(cells)} ячеек аудитории × цели = {len(need)} троек; без оценки {len(need - seen_keys - unseen_keys)}, "
          f"в обеих таблицах {len(seen_keys & unseen_keys)}")

    # 5. вшитые таблицы = CSV
    found = {}
    for node in ast.parse((ROOT / "agent.py").read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in ("PRIOR", "PRIOR_UNSEEN"):
            found[node.targets[0].id] = ast.literal_eval(node.value)
    stale = []
    for name, df, q, se in (("PRIOR", table, "prior_q", "prior_se"), ("PRIOR_UNSEEN", unseen, "q", "q_se")):
        emb = found.get(name, {})
        for r in df.itertuples():
            v = emb.get((r.tariff_from, r.arpu_segment, r.tariff_to))
            if v is None or abs(v[0] - getattr(r, q)) > 6e-6 or abs(v[1] - getattr(r, se)) > 6e-6:
                stale.append((name, r.tariff_from, r.arpu_segment, r.tariff_to))
    print(f"{'OK  ' if not stale else 'WARN'} agent.py: " + (
        "вшитые PRIOR и PRIOR_UNSEEN совпадают с CSV (с точностью до округления 5 знаков)" if not stale else
        f"{len(stale)} троек отличаются от CSV, например {stale[0]} — перевшить: python prior/build_prior.py"))

    print(f"\nитог: {'все проверки сошлись' if not failed else 'НЕ СОШЛОСЬ: ' + ', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
