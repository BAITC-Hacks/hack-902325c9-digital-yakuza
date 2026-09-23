# Beeline API

Агент берётся из beeline_agent/ без изменения исходников. FastAPI запускает его в отдельном процессе, ограниченном 600 секундами. Основная стратегия не зависит от OpenAI; существующий workflow может сформировать объяснение уже сохранённого плана. Все результаты относятся к официальной mock-среде, а не к скрытому судейству.

## Локальный запуск

Из backend:

```sh
docker compose up --build -d
docker compose exec backend alembic upgrade head
```

Документация: http://localhost:8000/docs. Существующие / и /health сохранены.

## API для frontend

| Метод | URL | Результат |
| --- | --- | --- |
| GET | /api/case/summary | Аудитория, тарифы, сегменты, каналы, ограничения, версия агента |
| POST | /api/agent/run | Запуск; JSON {"seed":42} опционален; HTTP 202 и run_id |
| GET | /api/agent/result | Последний запуск, статус, trace, метрики и финальные кампании |
| GET | /api/agent/pilots | Пилоты последнего запуска, включая сегмент, гипотезу, канал, размер, эффект и стоимость |
| GET | /api/agent/submission | CSV последнего завершённого запуска, если он является выбранным запуском |

Для result, pilots и submission можно передать ?run_id=<UUID>, чтобы опрашивать конкретный запуск и скачивать его CSV после новых запусков.

1. Загрузить summary.
2. Отправить POST /api/agent/run и сохранить run_id.
3. Опросить result и pilots с этим run_id раз в 1–2 секунды.
4. При status=completed показать campaigns, metrics и скачать submission.
5. При status=failed показать error.

Статусы: running, completed, failed. HTTP 404 означает, что запуск ещё не создан или UUID неизвестен. HTTP 409 при старте означает, что уже выполняется другой запуск; при скачивании — CSV ещё не готов. Ошибочный seed даёт 422. Одновременно допускается один запуск для приложения, блокировка создания работает через PostgreSQL.

Нового статуса pending нет: запуск сразу создаётся как running. Старые поля и их типы сохранены; новые поля добавлены без изменения контракта frontend.

## Данные стратегии и решений

- summary.strategy: max_pilot_budget_fraction, max_pilot_contacts_fraction, exploration_timeout_seconds, max_pilots_per_candidate, pilot_channel, fallback_channel, prior_format, prior_mode, max_pilots; также channel_economics, risk_k и остальные параметры, перечисленные в BACKEND_CONTRACT.md. Значения читаются непосредственно из исходников агента и официальной среды. Это настройки, а не результаты последнего запуска.
- pilots: optional mu, sd, lcb, candidate и совместимый candidate_id. Значения дословно берутся из соответствующего события pilot после обновления оценки агентом. В первом ответе о новом пилоте этих полей ещё может не быть. mu/sd/lcb относятся к базовому эффекту q, observed_lift — к наблюдаемому эффекту канала; backend не пересчитывает их.
- result: warnings, stop_reason, is_fallback, fallback_reason, estimate_source, risk_info. Эти же данные хранятся в metrics JSONB. Для старых запусков warnings=[], остальные новые поля могут быть null.
- warnings: единственный источник — workflow.explain_result → explanation.rendered.warnings (severity, text, fact_ids). До готовности объяснения список пуст. Backend не классифицирует предупреждения самостоятельно; технические ошибки и решения остаются в trace. Эти предупреждения также сохраняются в metrics.warnings.
- stop_reason берётся из explore_stop конкретного запуска. Если события нет и env.pilots_left=0, причина — «израсходованы все пилоты»; иначе null. В summary причины конкретного запуска нет.
- is_fallback=true указывает на обработанную агентом ошибку основного плана или выбор minimal_campaign. У завершённого запуска это запасной результат; у failed — попытка перейти к запасному плану. При минимальной кампании estimate_source сохраняет исходное estimate, risk_info — имеющиеся mu, sd, downside, exposure_arpu, expected_gain. У обычных кампаний estimate_source=пилоты, risk_info содержит mu/lcb соответствующего plan_add, candidate_id ссылается на исходного кандидата. Эти поля также добавляются к финальной кампании. Значения не вычисляются backend.

Backend-адаптер наследует Agent и дополняет существующий _log пересылкой записанного события. Он не меняет act, выбор кандидатов, пилотов, оценки или план. Исходные события candidates, pilot, channels (assigned и budget_left), plan_add, explore_stop, minimal_campaign и ошибки сохраняются постепенно в trace JSONB. Новые названия событий и отсутствующие объяснения не генерируются. После завершения сохраняется полный trace без дублирования. При обновлении агента совместимость этого внутреннего метода проверяется локальным скриптом.

Пилот и его уточнённая оценка сохраняются последовательно; frontend может продолжать polling прежних endpoints. Для отображения новых полей нужна отдельная доработка frontend.

Пилоты сохраняются по мере выполнения. После перезапуска backend готовые результаты и CSV остаются в БД. Прерванный запуск не возобновляется автоматически; зависший running будет отмечен failed при следующем обращении к API спустя 630 секунд от старта.

metrics.total_cost и metrics.total_contacts включают пилоты. final_campaign_count — только финальные кампании; n_campaigns в официальном score включает пилоты. gross_lift и net_arpu_gain — оценки официальной mock-модели. Реальный результат судейства может отличаться.

Агент полагается на обрезку аудитории официальным скорингом. В campaigns возвращаются фактические n_contacts и флаги capped_at_campaign_limit / capped_at_reach_budget / capped_at_money_budget; исходные фильтры CSV сохраняются без изменения стратегии.

## Загрузка исходников агента

Единственный источник — beeline_agent/ в корне monorepo. BEELINE_AGENT_DIR переопределяет каталог; относительный путь считается от корня репозитория, абсолютный используется напрямую. Если переменная не задана, loader берёт соседний с backend каталог beeline_agent.

Loader проверяет agent.py, environment.py, mock_environment.py, scoring_core.py, workflow.py, официальные скрипты проверки и необходимые CSV, затем добавляет точный путь в sys.path. При отсутствии файлов endpoints возвращают 503 с описанием проблемы. Конфликт уже импортированных одноимённых модулей приводит к явной ошибке. Agent не импортируется при startup FastAPI; запуск остаётся в отдельном Python-процессе с рабочим каталогом backend.

agent_sha256 в summary и метриках вычисляется непосредственно из agent.py. Дополнительный agent_sha256_lf нормализует CRLF в LF только в памяти; исходник не меняется. agent_status=CURRENT/OUTDATED сравнивает фактический/нормализованный SHA с EXPECTED_AGENT_SHA256 (по умолчанию версия 59547e1d… из текущего контракта). Это одинаковая версия на Windows и Linux. После изменения исходников нужно перезапустить backend: Python кеширует импорты, summary кешируется внутри процесса. При одновременном редактировании файлов во время запуска воспроизводимость не гарантируется.

В Docker исходники агента доступны по /app/beeline_agent через read-only mount. Backend находится в /app/backend. Образ также включает обе папки из одного checkout; отдельного генерируемого экземпляра кода в репозитории нет. Зависимости агента должны быть установлены через backend/requirements.txt.

## Heroku

Репозиторий подготовлен к сборке всего monorepo. В корне находятся Procfile, requirements.txt и .python-version. DevOps отдельно удаляет APP_BASE и monorepo buildpack, оставляет heroku/python и сохраняет существующие DATABASE_URL и приложение. До этого изменения новая схема на старом deployment работать не будет.

После отдельно согласованного commit в main и настройки Heroku будущая команда:

```sh
git push heroku main
```

Переход на загрузку исходников не требует новой миграции. Существующая модель explanation из текущего main требует уже имеющуюся миграцию 0003. Корневой release process выполняет cd backend && alembic upgrade head. После разрешённого deployment проверить успешный release, /health, /api/case/summary и SHA агента, затем запуск через /docs.

## Полная локальная проверка

Из корня репозитория:

```sh
docker compose -f backend/docker-compose.yml up --build -d --wait
docker compose -f backend/docker-compose.yml exec -T backend python -B -m scripts.check_beeline
```

Проверка допускается только с hostname БД postgres, работает через localhost контейнера и создаёт один обычный запуск в локальной PostgreSQL. Проверяет health/docs, все пять endpoints, старые поля, оценки, trace, JSONB-записи, лимиты и официальный evaluate_agent(). CSV из API сравнивается с build_submission() при официальном seed=42 без записи в исходную папку агента. Сценарий ошибки основного плана отдельно проверяет адаптер и fallback без записи искусственного результата в БД и без изменения исходников агента.

incremental_trace_observed показывает, успел ли polling увидеть промежуточный trace; окончательное сохранение проверяется всегда. Существующий скрипт check_beeline.py сохранён, поскольку проверяет интеграцию API и PostgreSQL.

## Официальная проверка в Docker

Из backend:

```sh
docker compose exec backend python -B -c "import os; from app.core.beeline import agent_directory; os.chdir(agent_directory()); from agent import Agent; from local_eval import evaluate_agent; result=evaluate_agent(Agent(verbose=False), seed=42); assert result['status']=='PASS'"
```

Для получения submission используется API или build_submission() с возвратом CSV в памяти. Не запускать make_submission.py с записью в каталог агента: в Compose он подключён read-only. CSV каждого запуска хранится в PostgreSQL.

## Объяснение готового плана

explanation может оставаться null после status=completed: workflow обрабатывает уже сохранённые результаты этого run_id, агент повторно не запускается. Сохранены исходные explanation.explanation, facts, campaign_pilot_links, llm_usage. Для удобства добавлены совместимые aliases explanation.source, explanation.rendered и explanation.fallback_reason, без удаления вложенного формата.

source=llm означает ответ модели, template — явно обозначенный шаблон, unavailable — невозможность построить факты. Сбой объяснения не меняет status=completed, кампании и CSV. Все предупреждения возвращаются из rendered.warnings; правила предупреждений backend не создаёт. Значения настроенных OpenAI-ключей удаляются из объяснения перед сохранением; исключения этапа объяснения логируются по типу без текста с возможными секретами.

Полная локальная проверка ожидает параметры текущего seed=42 из BACKEND_CONTRACT.md, совпадение CSV с исходным submission.csv, сохранение channels и связей C1→P5,P6; C2→P13,P14; C3→P15,P16; C4→P7,P8; C5→P9,P10. Проверка сбоя LLM имитируется без сетевого запроса и проверяет, что план не меняется, Agent повторно не вызывается и ключ не остаётся в результате.
