# Контракт backend ↔ агент Beeline (актуальная версия)

Агент: `beeline_agent/agent.py`, ветка `Agent`, **sha256 `59547e1d9a5695c67085d6db528c6d1636da5cbe34c0ee4f8f4a4f08dff52871`**
(экономика каналов; прошлая версия `a12541f0…` — `OUTDATED`).
ИИ-объяснение: `beeline_agent/workflow.py` (`explain_result`), ветка `Agent`, коммит `2d8cceb`+.
Backend ничего не считает сам: запускает агента, сохраняет его вывод, отдаёт фронту. Любые числа — из агента,
метрики — из `scoring_core.score_campaigns` организаторов.

## 0. Сначала
Влить `origin/Agent` в `main` (конфликтов нет, меняется только `beeline_agent/`). Backend импортирует агента прямо
из `beeline_agent/` (коммит `a731154`) — пересобирать пакет не нужно, новая версия подхватится после перезапуска.

## 1. Версия агента
- `agent_sha256()` считает sha файла на диске. На Heroku (Linux, файлы из git с LF) это `59547e1d…` → `CURRENT`.
  Локально на Windows при `core.autocrlf=true` файл с CRLF, sha `d19efe79…` — это та же версия, не `OUTDATED`.
- Запуск — отдельный процесс (`app.services.agent_worker`), seed по умолчанию 42, общий лимит 600 с
  (агент сам укладывается в ~1 с, свой таймер 180 с).

## 2. Что отдаёт агент
`Agent(verbose=False).act(env) -> list[dict]` — 1–10 кампаний (пустым план не бывает: есть запасная кампания).
Кампания: `campaign_name, filter_arpu_segment, filter_current_tariff ("t1;t2"), [filter_data_segment,
filter_call_segment — только у запасной], target_tariff, channel`.

`agent.trace` — журнал решений, список `{"t": секунды, "kind": ..., ...}`. Типы и поля (только эти):

| kind | поля | смысл |
|---|---|---|
| `candidates` | `count, top[(key, mu, size)]` | сколько гипотез построено из истории |
| `pilot` | `candidate, channel, n, observed, mu, sd, lcb` | пилот и оценка **после** него (доля ARPU, до множителя канала; observed — на канале пилота) |
| `pilot_error` | `candidate, error` | пилот не выполнился |
| `explore_stop` | `reason` [+ `spent_budget, spent_contacts`] | почему разведка остановилась (если нет — израсходованы все 20 пилотов) |
| `channels` | `assigned{candidate: channel}, budget_left` | какой канал выбран каждой кампании и сколько бюджета осталось |
| `plan_add` | `candidate, channel, contacts, mu, lcb, pilots` | кампания в плане и её доказательства |
| `plan_drop` | `campaign, reason` | кампания отброшена проверкой |
| `minimal_campaign` | `cell, contacts, channel, mu, sd, estimate, downside, exposure_arpu, expected_gain, reason` | **запасная** кампания (ни одна не прошла порог) |
| `minimal_skip` / `minimal_error` / `fallback_error` / `error` | `reason` / `error` | ошибки запасного пути |
| `done` | `campaigns, seconds` | финал |

Производные поля (backend их не придумывает, а берёт так):
- `stop_reason` = `reason` из `explore_stop`, иначе «израсходованы все пилоты».
- `is_fallback` = есть `minimal_campaign` **или** `error` (запасной план после ошибки).
- `fallback_reason` = `minimal_campaign.reason` / `error.error`.
- `estimate_source` = `minimal_campaign.estimate` («априор истории» / «пилот ×N»); у обычных кампаний — «пилоты».
- `risk_info` = `minimal_campaign.{downside, exposure_arpu}`; у обычных — `mu, lcb` из `plan_add`.
- Связь кампания → пилоты: `plan_add.candidate == pilot.candidate` (готово в `explanation.campaign_pilot_links`).

`env.pilot_history[i]` — i-й пилот (тот же порядок, что `pilot` в журнале): `pilot, target_tariff, channel,
n_customers, cost, observed_lift_ratio, observed_lift_total, remaining_budget, remaining_contacts`.

Метрики — `score_campaigns` (без `campaigns_detail`): `net_arpu_gain, gross_arpu_lift, total_cost,
total_contacts, unique_customers_targeted, coverage_pct, avg_gain_per_customer, roi, risk_score_pct,
budget_used_pct, growth_vs_baseline_pct, status (PASS/FAIL), baseline_total_arpu, total_arpu_after, n_campaigns`.
Кампания в результате = кампания агента + её строка `campaigns_detail` (`n_contacts, cost, gross_lift, n_negative,
capped_*`). **`n_campaigns` включает пилоты** — число кампаний плана = `len(campaigns)`.

## 3. Параметры стратегии (для `/api/case/summary`) — читать из модуля `agent`, не хардкодить
`PRIOR_MODE='raw'`, `PRIOR_SHRINK=0.5`, `TRANSFER_SD=0.10`, `RISK_K=1.0`, `MAX_PILOTS_PER_CANDIDATE=2`,
`PILOT_CHANNEL='sms'`, `PILOT_N_LARGE=200`, `PILOT_N_SMALL=100`, `SMALL_SEGMENT=600`, `TIME_BUDGET_S=180`,
`EXPLORE_BUDGET_SHARE=0.25`, `EXPLORE_CONTACT_SHARE=0.30`, `FALLBACK_CHANNEL='push'`, `FALLBACK_RISK_K=1.0`,
`MAX_CAMPAIGNS=10`, `MAX_PER_CAMPAIGN=5000`, `MIN_CANDIDATE_SIZE=30`, `CHANNEL_ECONOMICS=True`,
`PILOT_REPEAT_ONLY_IF_UNCLEAR=False`, `EXPLORE_UNSEEN=False`. Канал кампании выбирается по нижней границе
ценности: push → sms → digital_ads → call, пока доплата окупается и есть бюджет (на seed 42 — sms и digital_ads).

## 4. ИИ-объяснение (уже в `origin/Agent`)
- После `complete_run` (план закоммичен) → `explain_run(run_id)`: кампании, пилоты, журнал, метрики **этого run_id из
  базы** → `workflow.explain_result(result, use_llm=True)` в `asyncio.to_thread` → `agent_runs.explanation`.
  Агент повторно не запускается. Ошибка объяснения только логируется — план остаётся.
- Формат `explanation`: `{explanation: {source: "llm"|"template"|"unavailable", fallback_reason, raw, rendered:
  {summary, campaigns[{id, explanation, fact_ids}], warnings[{severity, text, fact_ids}], next_steps[{text, fact_ids}]}},
  facts, campaign_pilot_links, llm_usage: {model, attempts, key_slot, input_tokens, output_tokens, reasoning_tokens,
  latency_s, cost_usd, cost_upper_bound_usd, budget_usd, prices_source}}`.
- `warnings` для фронта брать из `explanation.rendered.warnings` — один источник с отчётом, не дублировать логику.
- Окружение сервера: `AGENT_LLM=1`, `OPENAI_API_KEY` (опц. `_2`, `_3`), `OPENAI_MODEL_FAST=gpt-5.6-luna`,
  `AGENT_LLM_TIMEOUT_SECONDS=30`, `AGENT_LLM_MAX_ATTEMPTS=2`, `AGENT_LLM_MAX_OUTPUT_TOKENS=2000`,
  `AGENT_LLM_RUN_BUDGET_USD=0.05`. Без них — шаблон с пометкой `source="template"` (не выдавать за ИИ).

## 5. API (минимум для фронта)
| Endpoint | Должен отдавать |
|---|---|
| `GET /api/case/summary` | статистика кейса + параметры стратегии (п. 3) + версия агента (sha, CURRENT/OUTDATED) |
| `POST /api/agent/run` | 202 + `run_id`; запуск в фоне |
| `GET /api/agent/result?run_id` | `status`, `metrics`, `campaigns` (+ детали подсчёта), `trace`, производные поля (п. 2), `explanation` (может быть `null`, пока считается) |
| `GET /api/agent/pilots?run_id` | пилоты: `pilot_history` + `mu, sd, lcb, candidate` из журнала |
| `GET /api/agent/submission?run_id` | CSV без изменений |

## 6. Приёмка (seed 42)
- 5 кампаний (каналы: sms, digital_ads, sms, sms, digital_ads), 20 пилотов, `total_cost = 95 094`,
  `total_contacts = 13 320`, самая большая кампания 4 108, `net_arpu_gain = 5 165 453`, `status = PASS`.
- CSV совпадает с `beeline_agent/submission.csv` (переводы строк не в счёт).
- `agent_sha256 = 59547e1d…` (на Heroku) → `CURRENT`.
- `explanation.campaign_pilot_links`: C1→P5,P6; C2→P13,P14; C3→P15,P16; C4→P7,P8; C5→P9,P10.
- Сломать объяснение (например, неверный ключ) → план и CSV на месте, `explanation.source = "template"`.
- Ключей API нет ни в базе, ни в логах, ни в ответах.

## 7. Heroku
- Приложение в `backend/`: `git subtree push --prefix backend heroku main` (или monorepo buildpack).
- `backend/Procfile`: добавить `release: alembic upgrade head`.
- Config Vars обязательны: `APP_NAME, APP_ENV, DEBUG, CORS_ORIGINS` (с доменом фронта), `DATABASE_URL` — от аддона
  (`postgres://` уже обрабатывается), плюс переменные п. 4.

## Нельзя
Пересчитывать математику, менять план или CSV, вызывать модель до сохранения плана, запускать агента повторно ради
объяснения, показывать шаблон как ответ ИИ, хранить ключи.
