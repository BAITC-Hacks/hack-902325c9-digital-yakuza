# Локальный запуск HackAlem

Требования: Docker Desktop с Linux containers и Node.js 22.12+ с npm.

## 1. Backend — первый терминал

Из корня репозитория:

```powershell
cd backend
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
docker compose up --build -d
docker compose exec backend alembic upgrade head
docker compose logs -f backend
```

Backend: http://localhost:8000. Swagger: http://localhost:8000/docs.
Используется существующий Compose и только локальная PostgreSQL.

## 2. Frontend — второй терминал

Из корня репозитория:

```powershell
cd frontend
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
npm install
npm run dev
```

Frontend: http://localhost:5173. Vite завершится с ошибкой, если порт занят, вместо переключения на другой порт.
Для Linux/macOS копировать .env.example через `cp -n .env.example .env`.

В frontend/.env:

```dotenv
VITE_API_URL=http://localhost:8000
```

В backend/.env существующий CORS_ORIGINS должен включать http://localhost:5173:

```dotenv
CORS_ORIGINS=["http://localhost:5173","http://127.0.0.1:5173"]
```

После изменения backend/.env выполнить из backend: `docker compose up -d --force-recreate backend`.
После изменения frontend/.env перезапустить npm run dev.

Реальные маршруты backend имеют префикс /api. API client добавляет его к VITE_API_URL:
GET /api/case/summary; POST /api/agent/run; GET /api/agent/result; GET /api/agent/pilots; GET /api/agent/submission.
Прокси не используется: браузер обращается прямо к localhost:8000 с проверкой CORS.

## 3. Проверка

```sh
npm run typecheck
npm run build
```

Открыть dashboard, проверить Summary, нажать «Запустить агента», дождаться completed, проверить кампании и пилоты, скачать submission.csv.
Статусы UI: idle, running, completed, error. Backend failed отображается как error.
404 при первом запросе result означает отсутствие запусков. При 409 dashboard загружает уже выполняющийся запуск.
Результаты, пилоты и CSV запрашиваются с run_id. Во время running опрос выполняется каждые 2 секунды.
Observed lift отображается в процентах; стоимость и эффект — в у.е. из API. Метрики учитывают пилоты, если это указано на карточке.
Данные синтетические; mock-оценка не является результатом судейства.

## 4. Остановка

Frontend: Ctrl+C в его терминале.
Backend: из backend выполнить `docker compose down`. Volume PostgreSQL сохраняется.

Production deploy, Heroku, Vercel и GitHub Actions эта инструкция не затрагивает.
