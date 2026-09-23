"""
Тесты совместного обновления ячеек после пилота по группе (шаг 2).
Проверяем смысл, а не числа из того же кода: кто сдвигается сильнее, кто не меняется, к чему сходится.
"""

import numpy as np
import pytest

import agent
from agent import Agent, Candidate, Group, _prior_cov

SMS = 0.65


def cell(tariff, size, mu=0.1, sd=0.1):
    return Candidate("MID", "tariff_8", (tariff,), size=size, arpu_sum=size * 4000.0, mu=mu, sd=sd)


def group(cells, corr, monkeypatch):
    monkeypatch.setattr(agent, "CELL_CORR", corr)
    return Group("MID", "tariff_8", cells, _prior_cov(cells))


def test_bigger_share_moves_more(monkeypatch):
    big, small = cell("tariff_4", 900), cell("tariff_13", 100)
    g = group([big, small], 0.0, monkeypatch)
    Agent._update_group(g, [big, small], observed_ratio=0.0, multiplier=SMS, n=200)
    assert (0.1 - big.mu) > (0.1 - small.mu) > 0          # обе вниз, крупная — сильнее


def test_cell_outside_pilot_is_unchanged_when_independent(monkeypatch):
    inside, outside = cell("tariff_4", 500), cell("tariff_13", 500)
    g = group([inside, outside], 0.0, monkeypatch)
    Agent._update_group(g, [inside], observed_ratio=-0.1, multiplier=SMS, n=200)
    assert outside.mu == pytest.approx(0.1) and outside.sd == pytest.approx(0.1)
    assert inside.mu < 0.1


def test_correlated_cell_outside_pilot_moves_with_group(monkeypatch):
    inside, outside = cell("tariff_4", 500), cell("tariff_13", 500)
    g = group([inside, outside], 0.8, monkeypatch)
    Agent._update_group(g, [inside], observed_ratio=-0.1, multiplier=SMS, n=200)
    assert outside.mu < 0.1 and outside.sd < 0.1           # общая часть неопределённости переноса


def test_precise_pilot_pins_the_weighted_mean(monkeypatch):
    a, b = cell("tariff_4", 300, mu=0.2), cell("tariff_13", 700, mu=0.0)
    g = group([a, b], 0.0, monkeypatch)
    Agent._update_group(g, [a, b], observed_ratio=0.065, multiplier=SMS, n=10**12)   # q̂ = 0.1, почти без шума
    assert 0.3 * a.mu + 0.7 * b.mu == pytest.approx(0.1, abs=1e-6)


def test_uncertainty_never_grows(monkeypatch):
    cells = [cell(f"tariff_{i}", 100 * i) for i in range(1, 5)]
    g = group(cells, 0.5, monkeypatch)
    before = [c.sd for c in cells]
    Agent._update_group(g, cells[:2], observed_ratio=0.3, multiplier=SMS, n=150)
    assert all(c.sd <= s + 1e-12 for c, s in zip(cells, before))
    assert np.all(np.linalg.eigvalsh(g.cov) > -1e-12)      # ковариация остаётся корректной


# ---------------------------------------------------------------- проверки из ревью шага 2
def test_plus_and_minus_cells_averaging_to_zero_are_not_both_pinned(monkeypatch):
    # правда: +0.2 и −0.2, равные доли → пилот видит ≈ 0. Модель не должна объявить обе ячейки точно оценёнными.
    plus, minus = cell("tariff_4", 500, mu=0.0, sd=0.1), cell("tariff_13", 500, mu=0.0, sd=0.1)
    g = group([plus, minus], 0.0, monkeypatch)
    Agent._update_group(g, [plus, minus], observed_ratio=0.0, multiplier=SMS, n=10**9)
    assert plus.sd > 0.06 and minus.sd > 0.06              # по отдельности почти ничего не узнали
    assert plus.lcb() < 0 and minus.lcb() < 0              # ни одна не проходит порог
    assert g.cov[0, 1] < 0                                 # но их сумма зажата: оценки связаны отрицательно


def test_subset_estimate_uses_joint_covariance(monkeypatch):
    from agent import _subset_estimate
    cells = [cell("tariff_4", 400), cell("tariff_13", 300), cell("tariff_9", 300)]
    g = group(cells, 0.6, monkeypatch)
    Agent._update_group(g, cells, observed_ratio=0.08, multiplier=SMS, n=200)
    subset = [cells[0], cells[2]]                          # ячейку tariff_13 исключили из кампании
    mean, sd = _subset_estimate(g, subset)
    a = np.array([c.arpu_sum for c in subset])
    idx = [0, 2]
    assert mean == pytest.approx(float(a @ np.array([cells[i].mu for i in idx])))
    assert sd == pytest.approx(float(np.sqrt(a @ g.cov[np.ix_(idx, idx)] @ a)))
    naive_sd = float(np.sqrt(sum((c.arpu_sum * c.sd) ** 2 for c in subset)))
    assert sd != pytest.approx(naive_sd, rel=1e-3)         # связь оценок учтена, это не сумма дисперсий


def test_composition_uncertainty_only_for_mixed_cells(monkeypatch):
    from agent import _composition_var
    same = [cell("tariff_4", 500, mu=0.1, sd=1e-6), cell("tariff_13", 500, mu=0.1, sd=1e-6)]
    diff = [cell("tariff_4", 500, mu=0.3, sd=1e-6), cell("tariff_13", 500, mu=-0.1, sd=1e-6)]
    w = np.array([0.5, 0.5])
    g_same, g_diff = group(same, 0.0, monkeypatch), group(diff, 0.0, monkeypatch)
    assert _composition_var(g_same, w, np.array([0.1, 0.1]), 1000, 200) == pytest.approx(0.0, abs=1e-9)
    # независимый расчёт: (1/n)·Σw(q−q̄)²·(N−n)/(N−1) = (1/200)·0.04·(800/999)
    assert _composition_var(g_diff, w, np.array([0.3, -0.1]), 1000, 200) == pytest.approx(0.04 / 200 * 800 / 999, rel=1e-6)
