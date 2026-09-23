# Beeline API

Агент берётся из beeline_agent/ без изменения исходников. FastAPI запускает его в отдельном процессе, ограниченном 600 секундами. OpenAI не используется. Все результаты относятся к официальной mock-среде, а не к скрытому судейству.

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

- summary.strategy: max_pilot_budget_fraction, max_pilot_contacts_fraction, exploration_timeout_seconds, max_pilots_per_candidate, pilot_channel, fallback_channel, prior_format, prior_mode, max_pilots. Значения читаются из упакованного агента и официальной среды. Это настройки, а не результаты последнего запуска.
- pilots: optional mu, sd, lcb и candidate_id. Значения дословно берутся из соответствующего события pilot после обновления оценки агентом. В первом ответе о новом пилоте этих полей ещё может не быть. mu/sd/lcb относятся к базовому эффекту q, observed_lift — к наблюдаемому эффекту канала; backend не пересчитывает их.
- result: warnings, stop_reason, is_fallback, fallback_reason, estimate_source, risk_info. Эти же данные хранятся в metrics JSONB. Для старых запусков warnings=[], остальные новые поля могут быть null.
- warnings: список объектов с code и исходным event либо reason. Коды: fallback_used, strategy_warning, partial_exploration, recoverable_error. Они не заменяют технический status. partial_exploration означает явную остановку исследования агентом, в том числе плановое сохранение резерва.
- stop_reason берётся только из explore_stop конкретного запуска. В summary его нет. Если агент вышел из цикла без такого события, например исчерпал пилоты или время, поле null: backend не придумывает причину.
- is_fallback=true указывает на обработанную агентом ошибку основного плана или выбор minimal_campaign. У завершённого запуска это запасной результат; у failed — попытка перейти к запасному плану. При минимальной кампании estimate_source сохраняет исходное estimate, risk_info — имеющиеся mu, sd, downside, exposure_arpu, expected_gain. Эти поля также добавляются к финальной кампании. Значения не вычисляются backend.

Backend-адаптер наследует Agent и дополняет существующий _log пересылкой записанного события. Он не меняет act, выбор кандидатов, пилотов, оценки или план. Исходные события candidates, pilot, plan_add, explore_stop, minimal_campaign и ошибки сохраняются постепенно в trace JSONB. Новые названия событий и отсутствующие объяснения не генерируются. После завершения сохраняется полный trace без дублирования. При обновлении агента совместимость этого внутреннего метода проверяется локальным скриптом.

Пилот и его уточнённая оценка сохраняются последовательно; frontend может продолжать polling прежних endpoints. Для отображения новых полей нужна отдельная доработка frontend.

Пилоты сохраняются по мере выполнения. После перезапуска backend готовые результаты и CSV остаются в БД. Прерванный запуск не возобновляется автоматически; зависший running будет отмечен failed при следующем обращении к API спустя 630 секунд от старта.

metrics.total_cost и metrics.total_contacts включают пилоты. final_campaign_count — только финальные кампании; n_campaigns в официальном score включает пилоты. gross_lift и net_arpu_gain — оценки официальной mock-модели. Реальный результат судейства может отличаться.

Агент полагается на обрезку аудитории официальным скорингом. В campaigns возвращаются фактические n_contacts и флаги capped_at_campaign_limit / capped_at_reach_budget / capped_at_money_budget; исходные фильтры CSV сохраняются без изменения стратегии.

## Обновление пакета агента

Heroku использует APP_BASE=backend и не включает соседнюю папку beeline_agent автоматически. В backend/vendor/beeline_agent.zip находится воспроизводимый пакет неизменённых исходников и нужных CSV. manifest.json хранит SHA256 каждого файла.

После изменений агентской командой пересобрать пакет из корня репозитория:

```powershell
docker run --rm --mount "type=bind,source=$PWD,target=/workspace" -w /workspace python:3.12-slim python backend/scripts/package_beeline.py
```

Если Python уже установлен, эквивалент: python backend/scripts/package_beeline.py. Скрипт читает beeline_agent/, записывает только backend/vendor/beeline_agent.zip. Новые версии numpy/pandas из beeline_agent/requirements.txt нужно согласовать с backend/requirements.txt.

Проверка без записи, из корня репозитория:

```powershell
python -B backend/scripts/package_beeline.py --check
```

Без локального Python:

```powershell
docker run --rm --mount "type=bind,source=$PWD,target=/workspace,readonly" -w /workspace python:3.12-slim python -B backend/scripts/package_beeline.py --check
```

AGENT PACKAGE: CURRENT / exit 0 — все 12 исходных файлов, состав ZIP и manifest совпадают. AGENT PACKAGE: OUTDATED / exit 1 — есть отличие, отсутствующий файл или повреждённый пакет. Сравниваются все runtime-файлы из FILES, а не только agent.py. Проверку выполнять после merge агента и перед согласованным commit/deploy. При запуске Heroku соседние исходники недоступны, поэтому startup проверяет только целостность пакета и не проверяет его актуальность относительно репозитория.

## Heroku

Сохранить APP_BASE=backend, текущие buildpacks, Procfile, DATABASE_URL и Config Vars. Новые env-переменные не требуются.

Deployment выполняется вручную, только после отдельного разрешения на commit и push. После фиксации проверенных изменений в main:

```sh
git push heroku main
```

Незафиксированные изменения эта команда не отправляет. APP_BASE=backend, lstoll/heroku-buildpack-monorepo, heroku/python, Procfile и DATABASE_URL сохраняются. Для обновления объяснений агента новая миграция не требуется: используются существующие agent_runs, pilot_results и campaign_results. Существующая миграция 0002 должна быть уже применена; проверка: heroku run "alembic current" --app hackalem-backend.

После deployment проверить /health, /api/case/summary и SHA агента, затем запуск через /docs. CORS_ORIGINS должен содержать origin frontend. Локальная проверка ниже не обращается к production.

## Полная локальная проверка

Из корня репозитория, PowerShell:

```powershell
python -B backend/scripts/package_beeline.py --check
docker compose -f backend/docker-compose.yml up --build -d --wait
Get-Content -Raw -Encoding utf8 backend/scripts/check_beeline.py | docker compose -f backend/docker-compose.yml exec -T backend python -B -
```

Первая команда требует Python; Docker-эквивалент указан выше. Для Bash последняя команда: docker compose -f backend/docker-compose.yml exec -T backend python -B - < backend/scripts/check_beeline.py.

Проверка допускается только с hostname БД postgres, работает через localhost контейнера и создаёт один обычный запуск в локальной PostgreSQL. Проверяет health/docs, все пять endpoints, старые поля, новые оценки, trace, JSONB-записи, лимиты и официальный evaluate_agent(). CSV сравнивается с build_submission() и make_submission.py при официальном seed=42. Сценарий ошибки основного плана отдельно проверяет адаптер и fallback без записи искусственного результата в БД и без изменения исходников агента. incremental_trace_observed показывает, успел ли polling увидеть промежуточный trace; окончательное сохранение проверяется всегда.

## Официальная проверка в Docker

```sh
docker compose exec backend python -c "import os, runpy; from app.core.beeline import agent_directory; p=agent_directory(); os.chdir(p); runpy.run_path(str(p/'local_eval.py'), run_name='__main__')"
docker compose exec backend python -c "import os, runpy; from app.core.beeline import agent_directory; p=agent_directory(); os.chdir(p); runpy.run_path(str(p/'make_submission.py'), run_name='__main__'); print((p/'submission.csv').read_text())"
```

Официальные скрипты выполняются во временной распакованной копии; исходная папка beeline_agent/ не меняется. API хранит CSV каждого запуска в PostgreSQL, поэтому для frontend файловая система Heroku не нужна.
