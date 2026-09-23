"""
Workflow аналитика: агент → финальный план и CSV → структурированные факты → ИИ-объяснение → отчёт.

    cd beeline_agent
    python workflow.py                 # seed 42 (как make_submission.py), объяснение через модель из .env
    python workflow.py --no-llm        # без модели: шаблонное объяснение
    python workflow.py --seed 7 --out outputs/run7

Выход (папка --out, по умолчанию outputs/):
    submission.csv    — план агента (тот же, что make_submission.py при том же seed)
    report.json       — план, метрики, факты, объяснение, расход модели, журнал агента (для backend)
    explanation.md    — объяснение для человека

Правила: ИИ не меняет план и CSV (объяснение строится после плана из готовых фактов). Модель пишет текст
со ссылками на факты вида {C1.contacts}; числа подставляет код. Ответ с цифрами вне ссылок, с
неизвестными ссылками или битым JSON отклоняется → шаблонное объяснение с явной пометкой.
Ключи берутся из окружения или beeline_agent/.env (окружение важнее); в отчёт и логи не попадают.
"""

import argparse
import hashlib
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from agent import Agent
from make_submission import CAMPAIGN_COLUMNS
from mock_environment import _mock_fallback, _mock_impact_model, make_mock_env
from scoring_core import score_campaigns

ROOT = Path(__file__).resolve().parent
PLACEHOLDER = re.compile(r"\{([A-Z]+[0-9]*)\.([a-z_]+)\}")
ALLOWED_TOKENS = re.compile(r"\{[A-Z]+[0-9]*\.[a-z_]+\}|tariff_[0-9]+|\b[CPWM][0-9]*\b")
KEY_NAMES = ("OPENAI_API_KEY", "OPENAI_API_KEY_2", "OPENAI_API_KEY_3")


# ---------------------------------------------------------------- настройки
def load_env(path=ROOT / ".env"):
    """Читает .env, не перезаписывая уже заданные переменные окружения. Значения никуда не выводятся."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip())


def setting(name, default, cast=str):
    value = os.environ.get(name, "")
    return cast(value) if value != "" else default


# ---------------------------------------------------------------- 1. агент и план
def run_agent(seed):
    env, internals = make_mock_env(seed=seed, data_dir=str(ROOT / "data"),
                                   profile_path=str(ROOT / "customer_profile.csv"))
    agent = Agent(verbose=False)
    campaigns = agent.act(env)

    submission = pd.DataFrame(campaigns)                 # ровно как make_submission.build_submission
    for col in CAMPAIGN_COLUMNS:
        if col not in submission.columns:
            submission[col] = None
    submission = submission[CAMPAIGN_COLUMNS]

    pilots = internals.executed_pilot_campaigns()
    strategy = pd.DataFrame(pilots + campaigns)
    for col in CAMPAIGN_COLUMNS + ["explicit_ids"]:
        if col not in strategy.columns:
            strategy[col] = None
    model = _mock_impact_model(pd.read_csv(ROOT / "data" / "change_tariff.csv"))
    score = score_campaigns(strategy, env.customer_profile, model, env.tariffs,
                            float(env.customer_profile["predicted_arpu"].sum()), _mock_fallback, team_id="workflow")
    return {"campaigns": campaigns, "submission": submission, "score": score, "trace": agent.trace,
            "pilot_history": env.pilot_history, "n_pilots": len(pilots), "env": env}


# ---------------------------------------------------------------- 2. факты
def build_facts(run):
    score, trace, env = run["score"], run["trace"], run["env"]
    details = score["campaigns_detail"][run["n_pilots"]:]
    plan_logs = [e for e in trace if e["kind"] == "plan_add"]
    pilot_logs = [e for e in trace if e["kind"] == "pilot"]
    facts = {}

    pilot_cost = sum(p["cost"] for p in run["pilot_history"])
    facts["M"] = {"kind": "metrics", "fields": {
        "net": score["net_arpu_gain"], "gross": score["gross_arpu_lift"], "cost": score["total_cost"],
        "contacts": score["total_contacts"], "unique": score["unique_customers_targeted"],
        "budget_used_pct": score["budget_used_pct"], "growth_pct": score["growth_vs_baseline_pct"],
        "pilots": run["n_pilots"], "pilot_cost": pilot_cost,
        "remaining_budget": env.total_budget - score["total_cost"],   # среда списывает только пилоты
        "campaigns": len(run["campaigns"])}}

    for i, camp in enumerate(run["campaigns"], 1):
        detail = details[i - 1] if i - 1 < len(details) else {}
        log = plan_logs[i - 1] if i - 1 < len(plan_logs) else {}
        tariffs = (camp.get("filter_current_tariff") or "").split(";")
        facts[f"C{i}"] = {"kind": "campaign", "fields": {
            "name": camp.get("campaign_name"), "segment": camp.get("filter_arpu_segment"),
            "target": camp["target_tariff"], "channel": camp["channel"], "tariffs": len(tariffs),
            "contacts": detail.get("n_contacts"), "cost": detail.get("cost"),
            "gross_lift": detail.get("gross_lift"), "negative": detail.get("n_negative"),
            "mu": log.get("mu"), "lcb": log.get("lcb"), "pilots": log.get("pilots")}}

    # связи «кампания → её пилоты» формирует код: пилот проверял ровно ту гипотезу, из которой собрана кампания
    links = {}
    for i, camp in enumerate(run["campaigns"], 1):
        log = plan_logs[i - 1] if i - 1 < len(plan_logs) else {}    # у запасной кампании записи plan_add нет
        pilots_of = [f"P{j}" for j, p in enumerate(pilot_logs, 1) if log and p["candidate"] == log.get("candidate")]
        links[f"C{i}"] = {"pilots": pilots_of,
                          "tariffs": {camp["target_tariff"], *(camp.get("filter_current_tariff") or "").split(";")}}
        facts[f"C{i}"]["fields"]["pilot_ids"] = ",".join(pilots_of) or "нет"

    for j, p in enumerate(pilot_logs, 1):
        hist = run["pilot_history"][j - 1] if j - 1 < len(run["pilot_history"]) else {}
        segment, target = p["candidate"].split(":")[:2]
        facts[f"P{j}"] = {"kind": "pilot", "fields": {
            "segment": segment, "target": target, "n": p["n"], "cost": hist.get("cost"),
            "observed": p["observed"], "mu": p["mu"], "sd": p["sd"], "lcb": p["lcb"]}}

    warnings = []
    warnings.append(("info", "Результат посчитан на мок-среде организаторов: это не балл судейства, "
                             "скрытые эффекты на судействе другие.", ["M"]))
    for cid, f in facts.items():
        v = f["fields"]
        if f["kind"] == "campaign" and v["mu"] and v["lcb"] is not None and v["lcb"] < 0.25 * v["mu"]:
            warnings.append(("attention", f"Кампания {cid}: запас нижней границы над нулём мал "
                                          f"({{{cid}.lcb}} при оценке {{{cid}.mu}}).", [cid]))
        if f["kind"] == "campaign" and v["pilots"] == 1:
            warnings.append(("attention", f"Кампания {cid} подтверждена одной пробой.", [cid]))
    failed = [cid for cid, f in facts.items() if f["kind"] == "pilot" and f["fields"]["lcb"] < 0]
    if failed:
        facts["W1"] = {"kind": "aggregate", "fields": {
            "failed_pilots": len(failed), "failed_cost": sum(facts[c]["fields"]["cost"] or 0 for c in failed)}}
        warnings.append(("info", "Часть проб не подтвердила гипотезы: {W1.failed_pilots} проб, "
                                 "затраты {W1.failed_cost}. Это цена разведки.", ["W1"] + failed))
    if any(e["kind"] == "minimal_campaign" for e in trace):
        warnings.append(("attention", "Ни одна кампания не прошла порог надёжности: план — запасная кампания "
                                      "с минимальным риском, чтобы выполнить требование ТЗ.", ["M"]))
    if facts["M"]["fields"]["remaining_budget"] > 0.3 * env.total_budget:
        warnings.append(("info", "Больше трети бюджета не распределено ({M.remaining_budget}): "
                                 "экономический выбор каналов в этой версии не реализован.", ["M"]))
    return facts, [{"severity": s, "text": t, "fact_ids": ids} for s, t, ids in warnings], links


def fmt(field, value):
    if value is None:
        return "н/д"
    if field in ("mu", "lcb", "sd", "observed"):
        return f"{value * 100:+.1f}%"
    if field.endswith("_pct"):
        return f"{value:.1f}%"
    if isinstance(value, (int, float)) and field in ("net", "gross", "cost", "gross_lift", "pilot_cost",
                                                     "remaining_budget", "failed_cost"):
        return f"{value:,.0f} у.е.".replace(",", " ")
    if isinstance(value, (int, float)):
        return f"{value:,.0f}".replace(",", " ")
    return str(value)


def render(text, facts):
    def sub(m):
        fid, field = m.group(1), m.group(2)
        return fmt(field, facts[fid]["fields"][field])
    # модель иногда ставит свой «%» или точку сразу после ссылки — убираем дубли после подстановки
    return PLACEHOLDER.sub(sub, text).replace("%%", "%").replace("у.е..", "у.е.")


# ---------------------------------------------------------------- 3. объяснение
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["summary", "campaigns", "warnings", "next_steps"],
    "properties": {
        "summary": {"type": "string"},
        "campaigns": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["id", "explanation", "fact_ids"],
            "properties": {"id": {"type": "string"}, "explanation": {"type": "string"},
                           "fact_ids": {"type": "array", "items": {"type": "string"}}}}},
        "warnings": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["severity", "text", "fact_ids"],
            "properties": {"severity": {"type": "string", "enum": ["attention", "info"]},
                           "text": {"type": "string"},
                           "fact_ids": {"type": "array", "items": {"type": "string"}}}}},
        "next_steps": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["text", "fact_ids"],
            "properties": {"text": {"type": "string"},
                           "fact_ids": {"type": "array", "items": {"type": "string"}}}}},
    },
}

INSTRUCTIONS = """Ты объясняешь аналитику маркетинга Beeline готовый план тарифных кампаний. План уже выбран
статистическим агентом по результатам пилотов; ты его НЕ меняешь и не предлагаешь как уже сделанное.
Факты даны в JSON: у каждого есть id (M — итоги, C1… — кампании, P1… — пилоты, W1 — сводка) и поля.

Правила:
1. Используй только данные из фактов. Ничего не додумывай о данных, которых нет.
2. НЕ пиши цифры сам. Любое число указывай ссылкой {ID.поле}, например {C1.contacts} или {P3.observed}.
   Названия тарифов (tariff_8) и id фактов (C1, P2) писать можно.
3. У каждого объяснения, предупреждения и шага перечисли fact_ids, на которые оно опирается.
   Объясни КАЖДУЮ кампанию C… ровно один раз. В объяснении кампании ссылайся только на её пилоты из
   campaign_pilots (связи задал код), на M и на кампании; чужие пилоты там не упоминай.
4. Предположение, которого нет в фактах, начинай словом «Гипотеза:».
5. Предупреждения: что может пойти не так и почему это существенно (или нет), какие данные противоречат.
6. next_steps: что аналитику проверить дальше, конкретно и коротко.
7. mu/lcb — оценка базового эффекта и её нижняя граница (доля ARPU до множителя канала), observed — что показал
   пилот на своём канале. Результат посчитан на мок-среде, это не балл судейства.
Пиши по-русски, коротко и по делу."""


ID_MENTION = re.compile(r"\{([A-Z]+[0-9]*)\.[a-z_]+\}|\b([CPW][0-9]+|M)\b")
TARIFF_MENTION = re.compile(r"tariff_[0-9]+")


def _mentioned_ids(text):
    return {a or b for a, b in ID_MENTION.findall(text)}


def validate(answer, facts, links):
    """
    Список проблем ответа модели (пусто = годен). Кроме синтаксиса проверяем смысл ссылок:
    - объяснены все кампании плана, без повторов и лишних, id — именно кампании;
    - в объяснении кампании упоминаются только её пилоты (связи формирует код), итоги M и кампании;
    - упомянутые в объяснении кампании тарифы относятся к ней;
    - цифр вне ссылок нет, все ссылки и id существуют.
    """
    problems = []
    texts = [answer["summary"]] + [c["explanation"] for c in answer["campaigns"]] + \
            [w["text"] for w in answer["warnings"]] + [s["text"] for s in answer["next_steps"]]
    for text in texts:
        for fid, field in PLACEHOLDER.findall(text):
            if fid not in facts or field not in facts[fid]["fields"]:
                problems.append(f"неизвестная ссылка {{{fid}.{field}}}")
        if re.search(r"[0-9]", ALLOWED_TOKENS.sub("", text)):
            problems.append(f"цифры вне ссылок: «{text[:60]}…»")
        problems += [f"неизвестный id {i}" for i in sorted(_mentioned_ids(text)) if i not in facts]

    ids = [c["id"] for c in answer["campaigns"]]
    if sorted(ids) != sorted(links):
        problems.append(f"кампании в объяснении {sorted(ids)} ≠ план {sorted(links)} (пропуск, лишняя или повтор)")
    for c in answer["campaigns"]:
        cid = c["id"]
        if cid not in links:
            continue
        allowed = {cid, "M"} | set(links[cid]["pilots"]) | set(links)
        used = set(c["fact_ids"]) | _mentioned_ids(c["explanation"])
        foreign = sorted(i for i in used if i not in allowed)
        if foreign:
            problems.append(f"{cid}: ссылки на чужие пилоты/факты {foreign} (свои пилоты: {links[cid]['pilots'] or 'нет'})")
        tariffs = set(TARIFF_MENTION.findall(c["explanation"])) - links[cid]["tariffs"]
        if tariffs:
            problems.append(f"{cid}: упомянуты тарифы не этой кампании {sorted(tariffs)}")
    for item in answer["warnings"] + answer["next_steps"]:
        problems += [f"неизвестный id {i}" for i in item["fact_ids"] if i not in facts]
    return problems


# Официальные цены OpenAI, стандартный тариф, короткий контекст (USD за 1 млн токенов: вход, выход).
PRICES_USD_PER_1M = {"gpt-5.6-luna": (0.20, 1.20), "gpt-5.6-terra": (2.00, 12.00), "gpt-6-astra": (10.00, 50.00)}
PRICES_SOURCE = "https://developers.openai.com/api/docs/pricing — Standard, short context, 23.09.2026"


def prices(model):
    """(вход, выход) USD за 1 млн токенов и источник. Окружение важнее встроенной таблицы."""
    p_in, p_out = setting("AGENT_LLM_PRICE_INPUT_PER_1M", None, float), setting("AGENT_LLM_PRICE_OUTPUT_PER_1M", None, float)
    if p_in is not None and p_out is not None:
        return (p_in, p_out), "из окружения"
    if model in PRICES_USD_PER_1M:
        return PRICES_USD_PER_1M[model], PRICES_SOURCE
    return None, "цены модели неизвестны"


def _make_client(key, timeout):
    from openai import OpenAI
    return OpenAI(api_key=key, timeout=timeout, max_retries=0)


def explain_with_llm(facts, code_warnings, links):
    """
    Один вызов модели с жёсткими правилами расходов и ключей:
    - AGENT_LLM=1 — иначе модель не вызывается;
    - бюджет AGENT_LLM_RUN_BUDGET_USD проверяется ДО каждой попытки по худшему случаю
      (оценка входа с запасом + максимум выхода); при заданном бюджете без цен вызова нет;
    - каждая оплаченная попытка учитывается: по факту токенов, а при таймауте/обрыве — верхней оценкой;
    - следующий ключ — только при 401/403/исчерпанной квоте; при 429 — ожидание на том же ключе,
      сеть/таймаут — повтор на том же ключе; попытки кончились → шаблон, ключ не меняется.
    """
    model = setting("OPENAI_MODEL_FAST", "gpt-5.6-luna")
    meta = {"model": model, "attempts": 0, "key_slot": None, "input_tokens": 0, "output_tokens": 0,
            "reasoning_tokens": 0, "latency_s": 0.0, "cost_usd": 0.0, "cost_upper_bound_usd": 0.0,
            "budget_usd": setting("AGENT_LLM_RUN_BUDGET_USD", None, float), "prices_source": None}
    if setting("AGENT_LLM", "0") != "1":
        return None, meta, "AGENT_LLM не равен 1 — модель выключена настройкой"
    keys = [os.environ[n] for n in KEY_NAMES if os.environ.get(n)]
    if not keys:
        return None, meta, "нет ключа OPENAI_API_KEY"
    try:
        from openai import APIConnectionError, APIStatusError, APITimeoutError
    except ImportError:
        return None, meta, "пакет openai не установлен (pip install -r requirements-workflow.txt)"
    price, meta["prices_source"] = prices(model)
    budget = meta["budget_usd"]
    if budget is not None and price is None:
        return None, meta, f"задан бюджет {budget} USD, но цены модели {model} неизвестны — вызов не выполнен"

    payload = json.dumps({"facts": {k: v["fields"] for k, v in facts.items()},
                          "campaign_pilots": {c: l["pilots"] for c, l in links.items()},
                          "code_warnings": code_warnings}, ensure_ascii=False, default=str)
    max_output = setting("AGENT_LLM_MAX_OUTPUT_TOKENS", 2000, int)
    input_estimate = math.ceil(len(INSTRUCTIONS + payload) / 2)          # с запасом: ~2 символа на токен
    worst = (input_estimate * price[0] + max_output * price[1]) / 1e6 if price else 0.0
    timeout = setting("AGENT_LLM_TIMEOUT_SECONDS", 30.0, float)
    max_attempts = setting("AGENT_LLM_MAX_ATTEMPTS", 2, int)
    last_error = "неизвестная ошибка"
    for slot, key in enumerate(keys, 1):
        try:
            client = _make_client(key, timeout)
        except ImportError:
            return None, meta, "пакет openai не установлен (pip install -r requirements-workflow.txt)"
        for attempt in range(max_attempts):
            if budget is not None and meta["cost_upper_bound_usd"] + worst > budget:
                return None, meta, (f"бюджет запуска {budget} USD: следующая попытка (до {worst:.4f} USD) "
                                    f"могла бы его превысить")
            meta["attempts"] += 1
            started = time.monotonic()
            try:
                response = client.responses.create(
                    model=model, instructions=INSTRUCTIONS, input=payload,
                    reasoning={"effort": setting("AGENT_LLM_REASONING_EFFORT", "low")},
                    max_output_tokens=max_output,
                    text={"format": {"type": "json_schema", "name": "plan_explanation",
                                     "schema": SCHEMA, "strict": True}})
            except (APITimeoutError, APIConnectionError) as exc:
                meta["latency_s"] += time.monotonic() - started
                meta["cost_upper_bound_usd"] += worst                     # ответ мог быть оплачен
                last_error = f"сеть/таймаут: {type(exc).__name__}"
                continue
            except APIStatusError as exc:
                meta["latency_s"] += time.monotonic() - started
                code = getattr(exc, "code", None) or ""
                last_error = f"API {exc.status_code} {code}".strip()
                if exc.status_code in (401, 403) or code == "insufficient_quota":
                    break                                               # этот ключ не годится — следующий
                if exc.status_code == 429:
                    time.sleep(min(2.0 * (attempt + 1), 8.0))           # лимит частоты: ждём на том же ключе
                    continue
                return None, meta, last_error                            # 400/404 и т.п.: повтор не поможет
            meta["latency_s"] += time.monotonic() - started
            meta["key_slot"] = slot
            usage = response.usage
            meta["input_tokens"] += usage.input_tokens
            meta["output_tokens"] += usage.output_tokens
            details = getattr(usage, "output_tokens_details", None)
            meta["reasoning_tokens"] += getattr(details, "reasoning_tokens", 0) or 0
            if price:
                spent = (usage.input_tokens * price[0] + usage.output_tokens * price[1]) / 1e6
                meta["cost_usd"] += spent
                meta["cost_upper_bound_usd"] += spent
            try:
                return json.loads(response.output_text), meta, None
            except (json.JSONDecodeError, TypeError) as exc:
                return None, meta, f"ответ не JSON: {exc}"
        else:
            return None, meta, last_error                                # попытки на ключе кончились → шаблон
    return None, meta, last_error


def template_explanation(facts, code_warnings):
    campaigns = [{"id": cid, "fact_ids": [cid], "explanation":
                  f"Сегмент {f['fields']['segment']}, переход на {f['fields']['target']} через {f['fields']['channel']}: "
                  f"{{{cid}.contacts}} контактов, оценка эффекта {{{cid}.mu}}, нижняя граница {{{cid}.lcb}}, "
                  f"проб {{{cid}.pilots}}."}
                 for cid, f in facts.items() if f["kind"] == "campaign"]
    return {"summary": "План из {M.campaigns} кампаний, {M.pilots} проб. Чистый результат на мок-среде {M.net}, "
                       "затраты {M.cost}, охват {M.unique} абонентов.",
            "campaigns": campaigns, "warnings": code_warnings,
            "next_steps": [{"text": "Проверить кампании с предупреждениями и решить, запускать ли их.", "fact_ids": ["M"]}]}


# ---------------------------------------------------------------- 4. отчёт
def run_workflow(seed=42, use_llm=True, out=ROOT / "outputs"):
    load_env()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    run = run_agent(seed)
    run["submission"].to_csv(out / "submission.csv", index=False)   # ровно как make_submission.py
    csv_bytes = (out / "submission.csv").read_bytes()
    facts, code_warnings, links = build_facts(run)

    answer, meta, reason = (None, {"model": None, "attempts": 0}, "объяснение выключено (--no-llm)")
    if use_llm:
        answer, meta, reason = explain_with_llm(facts, code_warnings, links)
        if answer is not None:
            problems = validate(answer, facts, links)
            if problems:
                answer, reason = None, "ответ модели отклонён: " + "; ".join(problems[:3])
    source = "llm" if answer is not None else "template"
    if answer is None:
        answer = template_explanation(facts, code_warnings)

    rendered = {"summary": render(answer["summary"], facts),
                "campaigns": [dict(c, explanation=render(c["explanation"], facts)) for c in answer["campaigns"]],
                "warnings": [dict(w, text=render(w["text"], facts)) for w in answer["warnings"]],
                "next_steps": [dict(s, text=render(s["text"], facts)) for s in answer["next_steps"]]}
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "environment": "official_mock (не балл судейства)",
        "agent_sha256": hashlib.sha256((ROOT / "agent.py").read_bytes()).hexdigest(),
        "submission_csv": {"path": "submission.csv", "sha256": hashlib.sha256(csv_bytes).hexdigest()},
        "plan": run["campaigns"],
        "metrics": facts["M"]["fields"],
        "facts": facts,
        "explanation": {"source": source, "fallback_reason": reason if source == "template" else None,
                        "raw": answer, "rendered": rendered},
        "campaign_pilot_links": {c: l["pilots"] for c, l in links.items()},
        "llm_usage": meta,       # расход учтён всегда — и когда ответ отклонён и показан шаблон
        "trace": run["trace"],
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    md = [f"# Объяснение плана (seed {seed})", "",
          f"Источник объяснения: **{'модель ' + str(meta.get('model')) if source == 'llm' else 'шаблон'}**"
          + (f" — {reason}" if source == "template" else ""), "", rendered["summary"], "", "## Кампании"]
    md += [f"- **{c['id']}**: {c['explanation']}" for c in rendered["campaigns"]]
    md += ["", "## Предупреждения"] + [f"- [{w['severity']}] {w['text']}" for w in rendered["warnings"]]
    md += ["", "## Что проверить дальше"] + [f"- {s['text']}" for s in rendered["next_steps"]]
    (out / "explanation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    args = ap.parse_args()
    report = run_workflow(args.seed, use_llm=not args.no_llm, out=args.out)
    u = report["llm_usage"]
    print(f"план: {report['metrics']['campaigns']} кампаний, net (мок) {report['metrics']['net']:,.0f}")
    print(f"объяснение: {report['explanation']['source']}"
          + (f" ({report['explanation']['fallback_reason']})" if report['explanation']['fallback_reason'] else ""))
    if u.get("attempts"):
        print(f"модель {u['model']}: попыток {u['attempts']}, ключ №{u['key_slot']}, токены in/out "
              f"{u['input_tokens']}/{u['output_tokens']} (reasoning {u['reasoning_tokens']}), задержка "
              f"{u['latency_s']:.1f} с, стоимость {u['cost_usd']:.5f} USD (верхняя оценка "
              f"{u['cost_upper_bound_usd']:.5f}, бюджет {u['budget_usd']})")
    print(f"файлы: {args.out}/submission.csv, report.json, explanation.md")


if __name__ == "__main__":
    main()
