"""
Тесты оценок агента без среды: априор из таблицы и байесовское обновление после пилота.

Ожидаемые числа посчитаны отдельно от проверяемого кода (нормальная модель, вручную):
  пилот: q̂ = y / m,  σ = 0.804 / (m·√n);  апостериор = взвешивание по точности 1/σ².
Запуск: cd beeline_agent && python -m pytest -q tests
"""

import math

import pytest

import agent
from agent import Agent, Candidate
from eval.prior_tables import validate

SMS = 0.65


def candidate(mu=0.10, sd=0.10):
    return Candidate("MID", "tariff_8", ("tariff_4",), size=1000, arpu_sum=4.7e6, mu=mu, sd=sd)


# ---------------------------------------------------------------- обновление после пилота
def test_update_matches_hand_calculation():
    c = candidate()
    Agent._update(c, observed_ratio=0.0, multiplier=SMS, n=100)
    assert c.mu == pytest.approx(0.0604740, abs=1e-6)
    assert c.sd == pytest.approx(0.0777650, abs=1e-6)


def test_bigger_pilot_moves_estimate_more_and_shrinks_uncertainty_more():
    small, big = candidate(), candidate()
    Agent._update(small, observed_ratio=0.0, multiplier=SMS, n=50)
    Agent._update(big, observed_ratio=0.0, multiplier=SMS, n=200)
    assert abs(big.mu - 0.10) > abs(small.mu - 0.10)
    assert big.sd < small.sd < 0.10
    assert big.mu == pytest.approx(0.0433424, abs=1e-6)
    assert small.mu == pytest.approx(0.0753692, abs=1e-6)


def test_pilot_contradicting_history_pulls_estimate_below_zero():
    c = candidate(mu=0.10, sd=0.10)
    Agent._update(c, observed_ratio=-0.10, multiplier=SMS, n=200)   # история «+», пилот «−»
    assert c.mu < 0


def test_same_base_effect_on_different_channels_gives_same_estimate():
    # q = 0.1: push покажет 0.05, sms — 0.065. При почти пустом априоре оценка ≈ q на обоих каналах.
    push, sms = candidate(mu=0.0, sd=1e3), candidate(mu=0.0, sd=1e3)
    Agent._update(push, observed_ratio=0.05, multiplier=0.50, n=200)
    Agent._update(sms, observed_ratio=0.065, multiplier=SMS, n=200)
    assert push.mu == pytest.approx(0.10, abs=1e-6)
    assert sms.mu == pytest.approx(0.10, abs=1e-6)
    assert push.sd > sms.sd                                          # у push сигнал слабее


# ---------------------------------------------------------------- априор из таблицы
@pytest.fixture
def table(monkeypatch):
    def install(rows):
        monkeypatch.setattr(agent, "PRIOR_FORMAT", ("q", "q_se", "n_obs"))
        monkeypatch.setattr(agent, "PRIOR", rows)
    return install


def test_prior_sd_uses_ready_standard_error_without_sqrt_n(table):
    table({("tariff_4", "MID", "tariff_8"): (0.20, 0.03, 50)})
    mu, sd = Agent(verbose=False)._cell_prior("tariff_4", "MID", "tariff_8")
    assert mu == pytest.approx(agent.PRIOR_SHRINK * 0.20)
    assert sd == pytest.approx(0.1044031, abs=1e-6)                 # √(0.03² + 0.10²)


def test_prior_sd_does_not_depend_on_n_obs(table):
    table({("tariff_4", "MID", "tariff_8"): (0.20, 0.03, 5), ("tariff_4", "MID", "tariff_9"): (0.20, 0.03, 500)})
    a = Agent(verbose=False)
    assert a._cell_prior("tariff_4", "MID", "tariff_8") == a._cell_prior("tariff_4", "MID", "tariff_9")


def test_old_table_format_is_rejected_explicitly(monkeypatch):
    monkeypatch.setattr(agent, "PRIOR_FORMAT", None)
    with pytest.raises(ValueError, match="формате"):
        Agent(verbose=False)._cell_prior("tariff_4", "MID", "tariff_8")


# ---------------------------------------------------------------- проверка содержимого таблиц (R6)
GOOD = {"PRIOR_FORMAT": ("q", "q_se", "n_obs"),
        "PRIOR": {("tariff_4", "MID", "tariff_8"): (0.2, 0.03, 50)},
        "PRIOR_UNSEEN": {("tariff_4", "MID", "tariff_14"): (0.01, 0.05, 0)}}


def test_valid_tables_pass():
    assert validate(GOOD) == []


def test_old_rows_under_new_signature_are_caught():
    old_rows = dict(GOOD, PRIOR={("tariff_4", "MID", "tariff_8"): (0.2, 50, 0.9)})   # (q, n_obs, pct_std)
    assert validate(old_rows)


@pytest.mark.parametrize("row", [(float("nan"), 0.03, 5), (0.1, -0.01, 5), (0.1, 0.03, 2.5), (0.1, 0.03, -1)])
def test_bad_values_are_caught(row):
    assert validate(dict(GOOD, PRIOR={("tariff_4", "MID", "tariff_8"): row}))
