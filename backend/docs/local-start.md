# Локальный запуск

1. Запустить Docker Desktop с Linux containers.
2. Из корня репозитория перейти в backend и создать локальную конфигурацию:

```powershell
cd backend
Copy-Item .env.example .env
```

Для Linux/macOS вместо Copy-Item: `cp .env.example .env`.
На Linux при другом UID/GID добавить в .env значения LOCAL_UID и LOCAL_GID из команд `id -u` и `id -g`, чтобы Alembic мог записывать миграции.

3. Запустить контейнеры и применить пример миграции:

```sh
docker compose config --quiet
docker compose up --build -d
docker compose ps
docker compose exec backend alembic upgrade head
docker compose logs -f backend
```

4. Открыть http://localhost:8000/, http://localhost:8000/health и http://localhost:8000/docs.
Ожидаемый health: HTTP 200, `{"status":"ok","database":"ok"}`.

5. Остановить контейнеры, сохранив данные PostgreSQL:

```sh
docker compose down
```

PostgreSQL доступен только внутри Docker по имени postgres. Для SQL-консоли:

```sh
docker compose exec postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"'
```

Compose предназначен для локальной разработки: API опубликован на 127.0.0.1:8000, backend подключён в /app/backend read-only, alembic доступен для записи, beeline_agent подключён в /app/beeline_agent read-only. Контекст сборки — корень monorepo, Dockerfile — backend/Dockerfile. После изменения Python-кода: `docker compose restart backend`; после изменения backend/requirements.txt: `docker compose up --build -d`.

CORS_ORIGINS — JSON-массив разрешённых origins. OPENAI_API_KEY можно оставить пустым до подключения агента. Пароль в примере предназначен только для локальной разработки; при замене согласовать POSTGRES_PASSWORD и DATABASE_URL. Изменение POSTGRES_* не меняет пользователей уже созданного volume.

Из корня репозитория запуск: `docker compose -f backend/docker-compose.yml up --build -d`. BEELINE_AGENT_DIR внутри Compose задаётся как /app/beeline_agent. После изменения агента перезапустить backend. Полная проверка: `docker compose -f backend/docker-compose.yml exec -T backend python -B -m scripts.check_beeline`. Она создаёт один локальный запуск и не меняет исходники агента.
