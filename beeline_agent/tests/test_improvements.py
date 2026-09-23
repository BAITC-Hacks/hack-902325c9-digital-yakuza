"""
Тесты улучшений поверх a631a16: экономика каналов, повторный пилот только при неясном решении,
тарифы без истории. Ожидаемые значения посчитаны вручную (в комментариях).
"""

import time
from types import SimpleNamespace

import pytest

import agent
from agent import Agent, Candidate

CHANNELS = {"push": {"cost_per_contact": 0, "conversion_multiplier": 0.50},
            "sms": {"cost_per_contact": 4, "conversion_multiplier": 0.65},
            "digital_ads": {"cost_per_contact": 22, "conversion_multiplier": 0.85},
            "call": {"cost_per_contact": 160, "conversion_multiplier": 1.20}}


def item(mu, sd, size, arpu):
    c = Candidate("MID", "tariff_8", ("tariff_4",), size=size, arpu_sum=arpu, mu=mu, sd=sd)
    return {"candidate": c, "tariffs": ("tariff_4",), "size": size, "arpu": arpu}


def assign(items, budget):
    a = Agent(verbose=False)
    a._assign_channels(items, SimpleNamespace(channels=CHANNELS, remaining_budget=budget))
    return [it["channel"] for it in items]


# ---------------------------------------------------------------- каналы
def test_valuable_campaign_is_upgraded_within_budget():
    # нижняя граница 0.1, 100 контактов, ARPU 800 000: push 40 000; sms 51 600; ads 65 800; call 80 000.
    # Бюджет 5 000: push→sms (29 у.е. ценности на 1 у.е.), затем sms→ads (7.9); на call (доплата 13 800) денег нет.
    assert assign([item(0.2, 0.1, 100, 800_000)], budget=5_000) == ["digital_ads"]


def test_call_is_chosen_only_when_affordable_and_worth_it():
    assert assign([item(0.2, 0.1, 100, 800_000)], budget=20_000) == ["call"]      # call 80 000 > ads 65 800


def test_weak_campaign_stays_on_free_push():
    # нижняя граница 0.01, 1 000 контактов, ARPU 100 000: sms даёт +150 эффекта за 4 000 у.е. — невыгодно
    assert assign([item(0.02, 0.01, 1000, 100_000)], budget=100_000) == ["push"]


def test_budget_is_never_exceeded():
    items = [item(0.2, 0.1, 500, 4_000_000), item(0.3, 0.1, 500, 5_000_000), item(0.1, 0.05, 2000, 8_000_000)]
    channels = assign(items, budget=30_000)
    spent = sum(CHANNELS[ch]["cost_per_contact"] * it["size"] for ch, it in zip(channels, items))
    assert spent <= 30_000


# ---------------------------------------------------------------- повторный пилот
class FakeEnv:
    def __init__(self, profile_rows=0):
        self.customer_profile = SimpleNamespace()
        self.channels = CHANNELS
        self.total_budget = self.remaining_budget = 100_000
        self.max_total_contacts = self.remaining_contacts = 15_000
        self.pilots_left = 1
        self.calls = []

    def run_pilot(self, target_tariff, channel, n_customers, **filters):
        self.calls.append(target_tariff)
        self.pilots_left -= 1
        self.remaining_budget -= 4 * n_customers
        self.remaining_contacts -= n_customers
        return {"observed_lift_ratio": 0.05, "n_customers": n_customers}


@pytest.mark.parametrize("flag, expected", [(False, "tariff_8"), (True, "tariff_9")])
def test_confirmed_hypothesis_gets_no_repeat_pilot(monkeypatch, flag, expected):
    monkeypatch.setattr(agent, "PILOT_REPEAT_ONLY_IF_UNCLEAR", flag)
    confirmed = Candidate("MID", "tariff_8", ("tariff_4",), size=5000, arpu_sum=4e7, mu=0.3, sd=0.05,
                          pilots=[{"observed": 0.2, "n": 200}])        # нижняя граница 0.25 > 0 — решение ясно
    fresh = Candidate("MID", "tariff_9", ("tariff_4",), size=5000, arpu_sum=1e7, mu=0.05, sd=0.1)
    env = FakeEnv()
    a = Agent(verbose=False)
    a.trace, a._t0 = [], time.monotonic()
    a._explore(env, [confirmed, fresh])
    assert env.calls == [expected]


# ---------------------------------------------------------------- тарифы без истории
@pytest.mark.parametrize("flag", [False, True])
def test_unseen_tariffs_become_hypotheses_only_when_enabled(monkeypatch, flag):
    monkeypatch.setattr(agent, "PRIOR_FORMAT", ("q", "q_se", "n_obs"))
    monkeypatch.setattr(agent, "PRIOR", {})
    monkeypatch.setattr(agent, "PRIOR_UNSEEN", {("tariff_4", "MID", "tariff_14"): (0.2, 0.05, 0)})
    monkeypatch.setattr(agent, "EXPLORE_UNSEEN", flag)
    prior = Agent(verbose=False)._cell_prior("tariff_4", "MID", "tariff_14")
    if flag:
        assert prior[0] == pytest.approx(agent.PRIOR_SHRINK * 0.2)
        assert prior[1] == pytest.approx((0.05 ** 2 + agent.TRANSFER_SD ** 2) ** 0.5)
    else:
        assert prior is None
