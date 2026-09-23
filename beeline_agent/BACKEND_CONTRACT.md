# Контракт backend ↔ агент Beeline (актуальная версия)

Агент: `beeline_agent/agent.py`, коммит `a631a16`, **sha256 `a12541f033a91d74730dd067c7ad04b017a2e0e4212aed9c00a5ef2041a00d8e`**.
ИИ-объяснение: `beeline_agent/workflow.py` (`explain_result`), ветка `Agent`, коммит `2d8cceb`+.
Backend ничего не считает сам: запускает агента, сохраняет его вывод, отдаёт фронту. Любые числа — из агента,
метрики — из `scoring_core.score_campaigns` организаторов.

## 0. Сначала
1. Влить `origin/Agent` в свою ветку. Там уже есть: объяснение после плана (`services/agent_runs.py::explain_run`),
   миграция `0003` (`agent_runs.explanation`), поле `explanation` в `RunResponse`, фикс распаковки на Windows,
   `workflow.py` в пакете. Конфликт в `vendor/beeline_agent.zip` не сливать — после слияния пересобрать:
   `python backend/scripts/package_beeline.py`.
2. Своя миграция тоже `0003`? Перенумеровать в `0004` с `down_revision = "0003"` — иначе у Alembic две головы.

## 1. Пакет агента
- Файлы: `agent.py, environment.py, mock_environment.py, scoring_core.py, local_eval.py, make_submission.py,
  requirements.txt, workflow.py, customer_profile.csv, tariff_dictionary.csv, feature_dictionary.csv,
  data/change_tariff.csv, data/dict_tariff.csv` + `manifest.json` (sha256 каждого файла).
- `CURRENT`/`OUTDATED`: сравнивать `manifest.json["agent.py"]` с ожидаемым sha выше (константа в backend или
  переменная `EXPECTED_AGENT_SHA256`). Папки `beeline_agent/` на Heroku не будет — сравнивать не с ней.
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
`MAX_CAMPAIGNS=10`, `MAX_PER_CAMPAIGN=5000`, `MIN_CANDIDATE_SIZE=30`. Каналы плана: sms (push — если не хватает
бюджета и у запасной кампании); `call`/`digital_ads` агент не использует.

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
- 5 кампаний, 20 пилотов, `total_cost = 53 280`, `total_contacts = 13 320`, самая большая кампания 4 108,
  `net_arpu_gain = 4 777 157`, `status = PASS`.
- CSV совпадает с `beeline_agent/submission.csv` (переводы строк не в счёт).
- `manifest.json["agent.py"] = a12541f0…` → `CURRENT`.
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
