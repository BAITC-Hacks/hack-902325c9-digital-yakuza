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

## Heroku

Сохранить APP_BASE=backend, текущие buildpacks, Procfile, DATABASE_URL и Config Vars. Новые env-переменные не требуются.

1. Включить изменения backend, включая vendor/beeline_agent.zip, в согласованный commit и push ветки main. Heroku Automatic Deploys соберёт backend.
2. После успешного deployment применить только новую миграцию:

```sh
heroku run "alembic upgrade head" --app hackalem-backend
heroku run "alembic current" --app hackalem-backend
```

3. Проверить /health и /api/case/summary, затем запуск через /docs. До применения миграции endpoints запусков недоступны.
4. CORS_ORIGINS должен содержать origin frontend.

Миграция добавляет только agent_runs, pilot_results и campaign_results. Downgrade удаляет эти таблицы вместе с историей запусков. Production deployment и миграция выполняются отдельно от локальной проверки.

## Официальная проверка в Docker

```sh
docker compose exec backend python -c "import os, runpy; from app.core.beeline import agent_directory; p=agent_directory(); os.chdir(p); runpy.run_path(str(p/'local_eval.py'), run_name='__main__')"
docker compose exec backend python -c "import os, runpy; from app.core.beeline import agent_directory; p=agent_directory(); os.chdir(p); runpy.run_path(str(p/'make_submission.py'), run_name='__main__'); print((p/'submission.csv').read_text())"
```

Официальные скрипты выполняются во временной распакованной копии; исходная папка beeline_agent/ не меняется. API хранит CSV каждого запуска в PostgreSQL, поэтому для frontend файловая система Heroku не нужна.
