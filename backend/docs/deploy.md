# Heroku

Приложение: `hackalem-backend`.
Production: https://hackalem-backend-5a3d74f6419c.herokuapp.com/

## Настройки deployment

1. Сохранить ручной deployment: `git push heroku main`, только после отдельного разрешения. Automatic Deploys и backend deployment через GitHub Actions для этой схемы не использовать.
2. DevOps отдельно удаляет Config Var `APP_BASE`: сборка должна видеть весь monorepo.
3. DevOps отдельно удаляет monorepo buildpack и оставляет `heroku/python`. Python 3.12 задаётся корневым `.python-version`. До этих действий deployment новой версии не готов.
4. Использовать существующий Aiven PostgreSQL. Production Config Vars:

| Переменная | Значение |
| --- | --- |
| APP_NAME | Название приложения |
| APP_ENV | production |
| DEBUG | false |
| DATABASE_URL | Полный URL production PostgreSQL от провайдера, включая требуемые SSL-параметры |
| CORS_ORIGINS | JSON-массив реальных frontend origins, например ["https://example.com"]; ["*"] разрешает любой origin |
| OPENAI_API_KEY | Необязательный ключ для объяснений; стратегия работает без него |
| BEELINE_AGENT_DIR | Необязательно: по умолчанию beeline_agent в корне репозитория |

POSTGRES_DB, POSTGRES_USER и POSTGRES_PASSWORD нужны только локальному Compose.
В production DATABASE_URL не должен содержать Docker hostname postgres.
URL postgres:// и postgresql:// преобразуются в postgresql+psycopg://; hostname, credentials и query-параметры сохраняются. Для Heroku использовать psycopg, включая выданные провайдером параметры SSL.

5. Команда запуска из корневого `Procfile`:

```text
web: uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port $PORT
```

PORT задаёт Heroku. Корневой requirements.txt содержит -r backend/requirements.txt. Docker Compose для этого деплоя не используется. Дочерний процесс агента запускается с cwd=backend, поэтому не зависит от рабочего каталога корневого Procfile.

## Проверка PostgreSQL и миграций

Сначала проверить состояние миграций без изменения production-данных:

```sh
heroku run "cd backend && alembic current" --app hackalem-backend
heroku run "cd backend && alembic heads" --app hackalem-backend
heroku run "cd backend && alembic check" --app hackalem-backend
```

Для проверки SSL выполнить через соединение приложения SQL:

```sql
SELECT ssl, version FROM pg_stat_ssl WHERE pid = pg_backend_pid();
```

Ожидается `ssl=true`. HTTP 200 от /health подтверждает доступность БД, но сам по себе не подтверждает SSL или актуальность схемы.

После проверки содержимого новых миграций применять их отдельно:

```sh
heroku run "cd backend && alembic upgrade head" --app hackalem-backend
```

Во время deployment-аудита не запускать upgrade, downgrade, reset и команды удаления данных. Миграции автоматически не выполняются текущим Procfile.

## Ручной deployment

1. Убедиться, что checkout содержит backend/ и beeline_agent/, корневые Procfile, requirements.txt и .python-version. DevOps должен завершить переход на сборку всего monorepo.
2. Выполнить локальные проверки из `backend/docs/beeline-api.md`, показать diff и отчёт. Обновление объяснений агента использует существующие JSONB-поля и не требует новой миграции.
3. Только после отдельного разрешения зафиксировать изменения в main. Проверить `git status`, `git log -1 --oneline` и `git remote get-url heroku`: remote должен указывать на hackalem-backend.
4. Только после отдельного разрешения выполнить `git push heroku main`. Команда отправляет коммиты, а не незакоммиченные изменения. Дождаться успешных Build и Release для нужного SHA.
5. Проверить startup logs:

```sh
heroku logs --num 200 --app hackalem-backend
```

6. Проверить HTTP 200 для /, /health и /docs. В логах не должно быть H10, ошибок обязательных настроек, DATABASE_URL, подключения БД и PORT.

Официальный repository: https://github.com/BAITC-Hacks/hack-902325c9-digital-yakuza. Frontend: https://hackalem-frontend.vercel.app/. Heroku app и PostgreSQL сохраняются. Изменения APP_BASE и buildpacks для этого перехода выполняет только DevOps. Новая миграция не создаётся; существующая 0003 должна соответствовать текущей модели explanation.

Каталог beeline_case_participants/ исключён из Git на любой глубине. Перед push проверить git status и отсутствие .env среди отслеживаемых файлов.

Документация: [GitHub integration](https://devcenter.heroku.com/articles/github-integration), [Python](https://devcenter.heroku.com/articles/python-support), [Procfile](https://devcenter.heroku.com/articles/procfile).
