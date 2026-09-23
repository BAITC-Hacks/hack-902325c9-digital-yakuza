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

Compose предназначен для локальной разработки: API опубликован на 127.0.0.1:8000, app и alembic подключены с хоста. После изменения Python-кода: `docker compose restart backend`; после изменения requirements.txt: `docker compose up --build -d`.

CORS_ORIGINS — JSON-массив разрешённых origins. OPENAI_API_KEY можно оставить пустым до подключения агента. Пароль в примере предназначен только для локальной разработки; при замене согласовать POSTGRES_PASSWORD и DATABASE_URL. Изменение POSTGRES_* не меняет пользователей уже созданного volume.
