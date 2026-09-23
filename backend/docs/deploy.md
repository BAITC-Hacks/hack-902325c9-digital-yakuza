# Heroku

Приложение: `hackalem-backend`.
Production: https://hackalem-backend-5a3d74f6419c.herokuapp.com/

## Настройки deployment

1. Сохранить ручной deployment: `git push heroku main`, только после отдельного разрешения. Automatic Deploys и backend deployment через GitHub Actions для этой схемы не использовать.
2. Сохранить Config Var `APP_BASE=backend` для monorepo.
3. Порядок buildpacks: `https://github.com/lstoll/heroku-buildpack-monorepo`, затем `heroku/python`. Python 3.12 задаётся файлом `backend/.python-version`.
4. Использовать существующий Aiven PostgreSQL. Production Config Vars:

| Переменная | Значение |
| --- | --- |
| APP_NAME | Название приложения |
| APP_BASE | backend |
| APP_ENV | production |
| DEBUG | false |
| DATABASE_URL | Полный URL production PostgreSQL от провайдера, включая требуемые SSL-параметры |
| CORS_ORIGINS | JSON-массив реальных frontend origins, например ["https://example.com"]; ["*"] разрешает любой origin |
| OPENAI_API_KEY | Ключ при подключении агента; пока можно не задавать |

POSTGRES_DB, POSTGRES_USER и POSTGRES_PASSWORD нужны только локальному Compose.
В production DATABASE_URL не должен содержать Docker hostname postgres.
URL postgres:// и postgresql:// преобразуются в postgresql+psycopg://; hostname, credentials и query-параметры сохраняются. Для Heroku использовать psycopg, включая выданные провайдером параметры SSL.

5. Команда запуска из `backend/Procfile`:

```text
web: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

PORT задаёт Heroku. Docker Compose для этого деплоя не используется.

## Проверка PostgreSQL и миграций

Сначала проверить состояние миграций без изменения production-данных:

```sh
heroku run "alembic current" --app hackalem-backend
heroku run "alembic heads" --app hackalem-backend
heroku run "alembic check" --app hackalem-backend
```

Для проверки SSL выполнить через соединение приложения SQL:

```sql
SELECT ssl, version FROM pg_stat_ssl WHERE pid = pg_backend_pid();
```

Ожидается `ssl=true`. HTTP 200 от /health подтверждает доступность БД, но сам по себе не подтверждает SSL или актуальность схемы.

После проверки содержимого новых миграций применять их отдельно:

```sh
heroku run "alembic upgrade head" --app hackalem-backend
```

Во время deployment-аудита не запускать upgrade, downgrade, reset и команды удаления данных. Миграции автоматически не выполняются текущим Procfile.

## Ручной deployment

1. Из корня репозитория проверить `python -B backend/scripts/package_beeline.py --check`. Ожидается `AGENT PACKAGE: CURRENT`.
2. Выполнить локальные проверки из `backend/docs/beeline-api.md`, показать diff и отчёт. Обновление объяснений агента использует существующие JSONB-поля и не требует новой миграции.
3. Только после отдельного разрешения зафиксировать изменения в main. Проверить `git status`, `git log -1 --oneline` и `git remote get-url heroku`: remote должен указывать на hackalem-backend.
4. Только после отдельного разрешения выполнить `git push heroku main`. Команда отправляет коммиты, а не незакоммиченные изменения. Дождаться успешных Build и Release для нужного SHA.
5. Проверить startup logs:

```sh
heroku logs --num 200 --app hackalem-backend
```

6. Проверить HTTP 200 для /, /health и /docs. В логах не должно быть H10, ошибок обязательных настроек, DATABASE_URL, подключения БД и PORT.

Официальный repository: https://github.com/BAITC-Hacks/hack-902325c9-digital-yakuza. Frontend: https://hackalem-frontend.vercel.app/. Существующие Config Vars, APP_BASE, buildpacks, Heroku app и PostgreSQL не менять при обновлении агента.

Каталог beeline_case_participants/ исключён из Git на любой глубине. Перед push проверить git status и отсутствие .env среди отслеживаемых файлов.

Документация: [GitHub integration](https://devcenter.heroku.com/articles/github-integration), [Python](https://devcenter.heroku.com/articles/python-support), [Procfile](https://devcenter.heroku.com/articles/procfile).
