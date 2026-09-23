"""
Графики для README: регрессия к среднему в истории, где лежат деньги, проверка режима по пилотам.

    cd beeline_agent
    python -m sim.charts      # читает prior/reports/*.json|csv и sim/reports/headroom_milp.csv, пишет *.png рядом
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sim.worlds import ROOT  # noqa: E402

HISTORY_RIGHT = ["mock", "resample", "noise", "high_rich", "stingy", "causal"]
HISTORY_WRONG = ["random_soft", "random", "flip", "shift", "unknown_rich"]
COLORS = {"history_only": "#9aa5b1", "agent": "#f2a900", "known": "#6c8ebf", "known_milp": "#2e7d32",
          "ceiling": "#d0d0d0"}
LABELS = {"history_only": "только история", "agent": "наш агент", "known": "знаем эффекты,\nпланировщик агента",
          "known_milp": "знаем эффекты,\nоптимальный план", "ceiling": "потолок"}


def placebo_chart():
    rc = json.loads((ROOT / "prior" / "reports" / "research_check.json").read_text(encoding="utf-8"))["placebo"]
    segs = ["LOW", "MID", "HIGH"]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(x - 0.2, [rc["clipped_pct"][s] for s in segs], 0.4, color="#9aa5b1", label="без смены тарифа (плацебо)")
    ax.bar(x + 0.2, [rc["real_switches_clipped_pct"][s] for s in segs], 0.4, color="#f2a900", label="после смены тарифа")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, [f"{s}" for s in segs])
    ax.set_ylabel("средний Δ% ARPU")
    ax.set_title("«Рост» LOW и «падение» HIGH есть и без смены тарифа:\nэто регрессия к среднему, а не эффект кампании",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "prior" / "reports" / "placebo.png", dpi=150)
    plt.close(fig)


def headroom_chart():
    d = pd.read_csv(ROOT / "sim" / "reports" / "headroom_milp.csv")
    groups = {"история верна\n(mock, resample, noise…)": HISTORY_RIGHT,
              "история врёт\n(random, flip, shift…)": HISTORY_WRONG, "история почти бесполезна\n(random_hard)": ["random_hard"]}
    cols = ["history_only", "agent", "known", "known_milp", "ceiling"]
    shares = []
    for scen in groups.values():
        g = d[d["scenario"].isin(scen)]
        shares.append([100 * g[c].sum() / g["ceiling"].sum() for c in cols])
    shares = np.array(shares)
    x = np.arange(len(groups))
    w = 0.16
    fig, ax = plt.subplots(figsize=(8, 4))
    for j, c in enumerate(cols):
        bars = ax.bar(x + (j - 2) * w, shares[:, j], w, color=COLORS[c], label=LABELS[c])
        for b, v in zip(bars, shares[:, j]):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontsize=7)
    ax.set_xticks(x, list(groups))
    ax.set_ylabel("% от потолка (сумма по мирам)")
    ax.set_ylim(0, 112)
    ax.set_title(f"Где лежат деньги: {len(d)} миров симулятора, 5 на сценарий", fontsize=10)
    ax.legend(frameon=False, fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.savefig(ROOT / "sim" / "reports" / "headroom.png", dpi=150)
    plt.close(fig)


def regime_chart():
    r = pd.read_csv(ROOT / "prior" / "reports" / "regime_calibration.csv")
    flags = sorted([c for c in r.columns if c.startswith("flag_")], key=lambda c: int(c.split("_")[1]))
    k = [int(c.split("_")[1]) for c in flags]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for name, scen, color in [("история верна (mock, resample)", ["mock", "resample"], "#2e7d32"),
                              ("история врёт (random, random_hard, flip, shift, unknown_rich)",
                               ["random", "random_hard", "flip", "shift", "unknown_rich"], "#c62828")]:
        g = r[r["scenario"].isin(scen)]
        ax.plot(k, [100 * g[c].mean() for c in flags], marker="o", color=color, label=name)
    ax.set_xticks(k)
    ax.set_xlabel("пилотов SMS по 200 абонентов на лучшие связки истории")
    ax.set_ylabel("% миров: «история не подходит»")
    ax.set_ylim(0, 100)
    ax.set_title("Проверка режима: χ²-тест пилотов против прогноза истории (95%)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "prior" / "reports" / "regime.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    placebo_chart()
    headroom_chart()
    regime_chart()
    print("готово: prior/reports/placebo.png, sim/reports/headroom.png, prior/reports/regime.png")
