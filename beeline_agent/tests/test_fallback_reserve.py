"""
Тесты запасной кампании и резерва ресурсов без настоящей среды.
Запуск: cd beeline_agent && python -m pytest -q tests
"""

import pandas as pd
import pytest

import agent
from agent import Agent, Candidate

CHANNELS = {"push": {"cost_per_contact": 0, "conversion_multiplier": 0.50},
            "sms": {"cost_per_contact": 4, "conversion_multiplier": 0.65}}


def profile(rows):
    cols = ["ID_NUMBER", "current_tariff", "arpu_segment", "data_segment", "call_segment", "predicted_arpu"]
    return pd.DataFrame(rows, columns=cols)


class FakeEnv:
    def __init__(self, customer_profile, budget=100_000, contacts=15_000):
        self.customer_profile = customer_profile
        self.channels = CHANNELS
        self.total_budget = self.remaining_budget = budget
        self.max_total_contacts = self.remaining_contacts = contacts
        self.pilots_left = 20
        self.calls = []

    def run_pilot(self, target_tariff, channel, n_customers, **filters):
        cost = n_customers * CHANNELS[channel]["cost_per_contact"]
        self.remaining_budget -= cost
        self.remaining_contacts -= n_customers
        self.pilots_left -= 1
        self.calls.append((target_tariff, channel, n_customers, cost))
        return {"observed_lift_ratio": 0.05, "n_customers": n_customers}


@pytest.fixture
def prior(monkeypatch):
    def install(rows):
        monkeypatch.setattr(agent, "PRIOR_FORMAT", ("q", "q_se", "n_obs"))
        monkeypatch.setattr(agent, "PRIOR", rows)
    return install


# ---------------------------------------------------------------- запасная кампания
def test_fallback_is_one_homogeneous_cell_with_all_filters(prior):
    prior({("tariff_4", "MID", "tariff_8"): (0.4, 0.01, 100)})
    env = FakeEnv(profile([(1, "tariff_4", "MID", "HEAVY", "LOW", 5000),
                           (2, "tariff_4", "MID", "LITE", "LOW", 5000)]))
    [camp] = Agent(verbose=False)._minimal_campaign([], env)
    assert camp["filter_current_tariff"] == "tariff_4"            # ровно один тариф — однородная ячейка
    assert camp["filter_arpu_segment"] == "MID" and camp["target_tariff"] == "tariff_8"
    assert camp["channel"] == "push"
    assert camp["filter_data_segment"] in ("HEAVY", "LITE") and camp["filter_call_segment"] == "LOW"


def test_fallback_prefers_no_downside_then_smallest_exposure(prior):
    # tariff_4 → tariff_8: уверенно положительный (нижняя граница > 0); tariff_10 → tariff_8: отрицательный
    prior({("tariff_4", "MID", "tariff_8"): (0.4, 0.01, 100), ("tariff_10", "MID", "tariff_8"): (-0.4, 0.01, 100)})
    env = FakeEnv(profile([(1, "tariff_10", "MID", "LITE", "LOW", 100),       # маленький, но убыточный
                           (2, "tariff_4", "MID", "HEAVY", "LOW", 9000),      # без убытка, охват больше
                           (3, "tariff_4", "MID", "LITE", "LOW", 3000)]))     # без убытка, охват меньше
    [camp] = Agent(verbose=False)._minimal_campaign([], env)
    assert camp["filter_current_tariff"] == "tariff_4"
    assert camp["filter_data_segment"] == "LITE"                   # из безубыточных — меньший охват


def test_pooled_pilot_is_not_transferred_to_a_single_cell(prior):
    # пилот по смеси двух тарифов дал сильный плюс, но на ячейку tariff_4 его оценку переносить нельзя
    prior({("tariff_4", "MID", "tariff_8"): (-0.4, 0.01, 100)})
    pooled = Candidate("MID", "tariff_8", ("tariff_10", "tariff_4"), size=2, arpu_sum=1e4, mu=0.9, sd=0.01,
                       pilots=[{"observed": 0.5, "n": 200}])
    a = Agent(verbose=False)
    a.trace, a._t0 = [], 0.0
    a._minimal_campaign([pooled], FakeEnv(profile([(1, "tariff_4", "MID", "LITE", "LOW", 1000)])))
    [log] = [e for e in a.trace if e["kind"] == "minimal_campaign"]
    assert log["estimate"] == "априор истории" and log["mu"] < 0


def test_single_cell_pilot_estimate_is_used(prior):
    prior({("tariff_4", "MID", "tariff_8"): (-0.4, 0.01, 100)})
    single = Candidate("MID", "tariff_8", ("tariff_4",), size=1, arpu_sum=1e3, mu=0.3, sd=0.02,
                       pilots=[{"observed": 0.2, "n": 200}])
    a = Agent(verbose=False)
    a.trace, a._t0 = [], 0.0
    a._minimal_campaign([single], FakeEnv(profile([(1, "tariff_4", "MID", "LITE", "LOW", 1000)])))
    [log] = [e for e in a.trace if e["kind"] == "minimal_campaign"]
    assert log["estimate"].startswith("пилот") and log["mu"] == pytest.approx(0.3)


# ---------------------------------------------------------------- резерв ресурсов
def test_exploration_respects_budget_and_contact_reserve():
    env = FakeEnv(profile([]), budget=100_000, contacts=15_000)
    candidates = [Candidate("MID", f"tariff_{i}", ("tariff_4",), size=5000, arpu_sum=2.5e7, mu=0.2, sd=0.1)
                  for i in range(20)]
    a = Agent(verbose=False)
    a.trace, a._t0 = [], __import__("time").monotonic()
    a._explore(env, candidates)
    spent_budget = sum(c[3] for c in env.calls)
    spent_contacts = sum(c[2] for c in env.calls)
    assert spent_budget <= agent.EXPLORE_BUDGET_SHARE * 100_000
    assert spent_contacts <= agent.EXPLORE_CONTACT_SHARE * 15_000


def test_reserve_stops_exploration_when_budget_is_small():
    env = FakeEnv(profile([]), budget=5_000, contacts=15_000)          # 25% = 1 250 у.е. = один пилот sms-200 (800)
    candidates = [Candidate("MID", f"tariff_{i}", ("tariff_4",), size=5000, arpu_sum=2.5e7, mu=0.2, sd=0.1)
                  for i in range(5)]
    a = Agent(verbose=False)
    a.trace, a._t0 = [], __import__("time").monotonic()
    a._explore(env, candidates)
    assert len(env.calls) == 1
    assert any(e["kind"] == "explore_stop" and "резерв" in e["reason"] for e in a.trace)
