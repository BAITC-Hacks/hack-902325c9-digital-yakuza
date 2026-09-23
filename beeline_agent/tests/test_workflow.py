"""
Тесты ИИ-обвязки workflow.py с подменой API (платных вызовов нет).
Запуск: cd beeline_agent && python -m pytest -q tests
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest

import workflow

ROOT = Path(__file__).resolve().parents[1]
REQUEST = httpx.Request("POST", "https://api.openai.com/v1/responses")


def status_error(status, code=None):
    cls = {401: openai.AuthenticationError, 403: openai.PermissionDeniedError, 429: openai.RateLimitError}[status]
    body = {"code": code, "message": "test"}
    return cls("test", response=httpx.Response(status, request=REQUEST, json={"error": body}), body=body)


def ok_response(answer, tokens_in=1000, tokens_out=500):
    usage = SimpleNamespace(input_tokens=tokens_in, output_tokens=tokens_out,
                            output_tokens_details=SimpleNamespace(reasoning_tokens=10))
    return SimpleNamespace(usage=usage, output_text=json.dumps(answer, ensure_ascii=False))


class FakeAPI:
    """Сценарий ответов по номеру ключа; записывает, какими ключами и сколько раз звали."""

    def __init__(self, script_by_key):
        self.script = {k: (v if k.endswith("_default") else list(v)) for k, v in script_by_key.items()}
        self.calls = []

    def client(self, key, timeout):
        api = self

        class Responses:
            def create(self, **kwargs):
                api.calls.append(key)
                item = api.script[key].pop(0) if api.script[key] else api.script[key + "_default"]
                if isinstance(item, Exception):
                    raise item
                return item
        return SimpleNamespace(responses=Responses())


@pytest.fixture(scope="module")
def plan():
    run = workflow.run_agent(42)
    facts, warnings, links = workflow.build_facts(run)
    return facts, warnings, links


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(workflow, "load_env", lambda *a, **k: None)
    monkeypatch.setattr(workflow.time, "sleep", lambda s: None)
    for name, value in {"AGENT_LLM": "1", "OPENAI_API_KEY": "k1", "OPENAI_API_KEY_2": "k2", "OPENAI_API_KEY_3": "",
                        "OPENAI_MODEL_FAST": "gpt-5.6-luna", "AGENT_LLM_MAX_ATTEMPTS": "2",
                        "AGENT_LLM_RUN_BUDGET_USD": "0.05", "AGENT_LLM_PRICE_INPUT_PER_1M": "",
                        "AGENT_LLM_PRICE_OUTPUT_PER_1M": ""}.items():
        monkeypatch.setenv(name, value)
    return monkeypatch


def install(monkeypatch, api):
    monkeypatch.setattr(workflow, "_make_client", api.client)


def valid_answer(facts, links):
    return {"summary": "План из {M.campaigns} кампаний, результат на мок-среде {M.net}.",
            "campaigns": [{"id": c, "explanation": f"Кампания {c}: оценка {{{c}.mu}}, пилоты {' и '.join(l['pilots']) or 'нет'}.",
                           "fact_ids": [c] + l["pilots"]} for c, l in links.items()],
            "warnings": [{"severity": "info", "text": "Результат по мок-среде.", "fact_ids": ["M"]}],
            "next_steps": [{"text": "Проверить кампании после запуска.", "fact_ids": ["M"]}]}


# ---------------------------------------------------------------- 1. лимит расходов
def test_budget_blocks_request_before_it_is_made(env, plan):
    env.setenv("AGENT_LLM_RUN_BUDGET_USD", "0")
    api = FakeAPI({"k1": [], "k1_default": ok_response({}), "k2": [], "k2_default": ok_response({})})
    install(env, api)
    answer, meta, reason = workflow.explain_with_llm(*plan)
    assert answer is None and api.calls == [] and meta["attempts"] == 0
    assert "бюджет" in reason


def test_budget_without_prices_does_not_call(env, plan):
    env.setenv("OPENAI_MODEL_FAST", "unknown-model")
    api = FakeAPI({"k1": [], "k1_default": ok_response({}), "k2": [], "k2_default": ok_response({})})
    install(env, api)
    answer, meta, reason = workflow.explain_with_llm(*plan)
    assert answer is None and api.calls == [] and "цены" in reason


def test_rejected_answer_is_still_paid_for(env, plan, tmp_path):
    facts, warnings, links = plan
    bad = valid_answer(facts, links)
    bad["summary"] = "Итог 4777157 у.е."                      # цифры вне ссылок → отклонить
    api = FakeAPI({"k1": [ok_response(bad, 2000, 1000)], "k1_default": ok_response(bad), "k2": [], "k2_default": ok_response(bad)})
    install(env, api)
    report = workflow.run_workflow(42, use_llm=True, out=tmp_path)
    assert report["explanation"]["source"] == "template"
    assert "отклонён" in report["explanation"]["fallback_reason"]
    assert report["llm_usage"]["cost_usd"] == pytest.approx((2000 * 0.20 + 1000 * 1.20) / 1e6)


def test_timeouts_count_as_possible_spend(env, plan):
    timeout = openai.APITimeoutError(request=REQUEST)
    api = FakeAPI({"k1": [timeout, timeout], "k1_default": timeout, "k2": [], "k2_default": timeout})
    install(env, api)
    answer, meta, _ = workflow.explain_with_llm(*plan)
    assert answer is None and meta["cost_usd"] == 0 and meta["cost_upper_bound_usd"] > 0
    assert api.calls == ["k1", "k1"]                         # таймаут — не повод менять ключ


# ---------------------------------------------------------------- 2. ключи
def test_rate_limit_waits_on_same_key_and_never_switches(env, plan):
    api = FakeAPI({"k1": [], "k1_default": status_error(429), "k2": [], "k2_default": ok_response({})})
    install(env, api)
    answer, meta, reason = workflow.explain_with_llm(*plan)
    assert answer is None and "429" in reason
    assert api.calls == ["k1", "k1"]                         # только первый ключ, ровно MAX_ATTEMPTS раз


@pytest.mark.parametrize("error", [status_error(401), status_error(403), status_error(429, "insufficient_quota")])
def test_auth_or_quota_error_switches_to_next_key(env, plan, error):
    facts, warnings, links = plan
    api = FakeAPI({"k1": [error], "k1_default": error, "k2": [ok_response(valid_answer(facts, links))],
                   "k2_default": ok_response({})})
    install(env, api)
    answer, meta, reason = workflow.explain_with_llm(*plan)
    assert answer is not None and meta["key_slot"] == 2 and api.calls == ["k1", "k2"]


# ---------------------------------------------------------------- 3. проверка смысла ссылок
def test_valid_answer_passes(plan):
    facts, _, links = plan
    assert workflow.validate(valid_answer(facts, links), facts, links) == []


def test_missing_campaign_is_rejected(plan):
    facts, _, links = plan
    answer = valid_answer(facts, links)
    answer["campaigns"] = answer["campaigns"][1:]
    assert any("кампании в объяснении" in p for p in workflow.validate(answer, facts, links))


def test_empty_and_duplicate_campaigns_are_rejected(plan):
    facts, _, links = plan
    empty = dict(valid_answer(facts, links), campaigns=[])
    dup = valid_answer(facts, links)
    dup["campaigns"].append(dup["campaigns"][0])
    assert workflow.validate(empty, facts, links) and workflow.validate(dup, facts, links)


def test_pilot_id_in_place_of_campaign_is_rejected(plan):
    facts, _, links = plan
    answer = valid_answer(facts, links)
    answer["campaigns"][0]["id"] = "P1"
    assert workflow.validate(answer, facts, links)


def test_foreign_pilot_as_evidence_is_rejected(plan):
    facts, _, links = plan
    foreign = next(p for p in (k for k in facts if k.startswith("P")) if p not in links["C1"]["pilots"])
    answer = valid_answer(facts, links)
    answer["campaigns"][0]["explanation"] = f"Кампания C1 подтверждается пилотом {foreign}."
    problems = workflow.validate(answer, facts, links)
    assert any("чужие пилоты" in p for p in problems)


def test_foreign_tariff_in_campaign_explanation_is_rejected(plan):
    facts, _, links = plan
    other = next(f"tariff_{i}" for i in range(1, 22) if f"tariff_{i}" not in links["C1"]["tariffs"])
    answer = valid_answer(facts, links)
    answer["campaigns"][0]["explanation"] = f"Кампания C1 переводит на {other}."
    assert any("тарифы не этой кампании" in p for p in workflow.validate(answer, facts, links))


# ---------------------------------------------------------------- 4–5. настройка и отсутствие SDK
def test_agent_llm_zero_disables_the_model(env, plan):
    env.setenv("AGENT_LLM", "0")
    api = FakeAPI({"k1": [], "k1_default": ok_response({}), "k2": [], "k2_default": ok_response({})})
    install(env, api)
    answer, meta, reason = workflow.explain_with_llm(*plan)
    assert answer is None and api.calls == [] and "AGENT_LLM" in reason


def test_missing_openai_package_falls_back(env, plan, tmp_path):
    env.setitem(sys.modules, "openai", None)                 # import openai → ImportError
    report = workflow.run_workflow(42, use_llm=True, out=tmp_path)
    assert report["explanation"]["source"] == "template"
    assert "openai" in report["explanation"]["fallback_reason"]


# ---------------------------------------------------------------- план и CSV не зависят от ИИ
def test_csv_is_identical_with_and_without_llm(env, plan, tmp_path):
    facts, _, links = plan
    api = FakeAPI({"k1": [ok_response(valid_answer(facts, links))], "k1_default": ok_response({}),
                   "k2": [], "k2_default": ok_response({})})
    install(env, api)
    with_llm = workflow.run_workflow(42, use_llm=True, out=tmp_path / "llm")
    without = workflow.run_workflow(42, use_llm=False, out=tmp_path / "plain")
    assert with_llm["explanation"]["source"] == "llm"
    csv_llm, csv_plain = (tmp_path / "llm" / "submission.csv").read_bytes(), (tmp_path / "plain" / "submission.csv").read_bytes()
    assert csv_llm == csv_plain == (ROOT / "submission.csv").read_bytes()
    assert with_llm["plan"] == without["plan"]
