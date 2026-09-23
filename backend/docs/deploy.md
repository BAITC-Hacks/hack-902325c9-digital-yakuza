# Heroku

1. Деплоить содержимое backend как корень приложения: requirements.txt, Procfile и .python-version должны находиться в корне Heroku source. При использовании монорепозитория публиковать backend отдельным subtree или настраивать CI на этот каталог.
2. Использовать Python buildpack и Python 3.12 из .python-version.
3. Подключить PostgreSQL и задать Config Vars:

| Переменная | Значение |
| --- | --- |
| APP_NAME | Название приложения |
| APP_ENV | production |
| DEBUG | false |
| DATABASE_URL | Полный URL production PostgreSQL от провайдера, включая требуемые SSL-параметры |
| CORS_ORIGINS | JSON-массив, например ["https://example.com"] |
| OPENAI_API_KEY | Ключ при подключении агента; пока можно не задавать |

POSTGRES_DB, POSTGRES_USER и POSTGRES_PASSWORD нужны только локальному Compose.
В production DATABASE_URL не должен содержать Docker hostname postgres.
URL postgres:// и postgresql:// преобразуются в postgresql+psycopg://; hostname, credentials и query-параметры сохраняются. Для Heroku использовать psycopg, включая выданные провайдером параметры SSL.

4. Команда запуска из Procfile:

```text
web: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

PORT задаёт Heroku. Docker Compose для этого деплоя не используется.

5. Перед открытием приложения применить миграции:

```sh
heroku run alembic upgrade head --app YOUR_APP_NAME
```

6. Проверить /health и /docs по HTTPS URL приложения.

Документация: [Python](https://devcenter.heroku.com/articles/python-support), [Procfile](https://devcenter.heroku.com/articles/procfile).
