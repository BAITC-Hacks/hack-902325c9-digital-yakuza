# Технический каркас для хакатона

## Цель

Собрать универсальный технический каркас, чтобы на хакатоне после публикации задания менять только бизнес-логику и UI.

## Структура проекта

```text
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── agents/
│   │   ├── tools/
│   │   └── main.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   ├── Dockerfile
│   └── package.json
│
├── deploy/
│   ├── docker-compose.yml
│   └── nginx/
│
├── docs/
│   ├── 00-project-brief.md
│   ├── 01-init-backend.md
│   ├── 02-init-frontend.md
│   ├── 03-server-setup.md
│   ├── 04-domain-ssl.md
│   ├── 05-backend-cicd.md
│   └── 06-frontend-cicd.md
│
└── .github/
    └── workflows/
        ├── backend-deploy.yml
        └── frontend-deploy.yml
```

## Что готовим заранее

| Часть | Стек | Что готовим |
| --- | --- | --- |
| Backend | FastAPI + Python | API, конфиги `.env`, healthcheck, Docker |
| Agentic AI | Внутри backend или отдельный модуль | Базовый агент → tools → API → ответ |
| Frontend | React + TypeScript, web | Базовый интерфейс работы с агентом |
| Git | GitHub | Один репозиторий с каталогами `frontend/` и `backend/` |
| Server | Бесплатный VPS / credits из GitHub Student Pack | Docker + Nginx + HTTPS |
| CI/CD | GitHub Actions | Push в `main` → автоматический deploy |
| Domain | Бесплатный вариант из Student Pack | Домен → сервер → SSL |
| Docs | Отдельные `.md` | Только команды и шаги; README пока не трогаем |

## Frontend

Для заготовки используем React + TypeScript web, а не React Native. Это позволит автоматически собирать и деплоить фронтенд через Nginx на тот же VPS.

## Agentic AI

Сделать максимально простого, но настоящего агента: например, агент получает вопрос студента → самостоятельно вызывает один из tools → формирует результат.

## Текущий этап

Пока создать только каркас по предложенной структуре и один Markdown-файл с описанием задачи. Реализация бизнес-логики, UI, конфигураций и автоматического деплоя будет выполняться на следующих этапах.

Файлы в структуре пока являются заготовками. `package.json` содержит пустой JSON-объект. Файлы `.gitkeep` сохраняют пустые каталоги в Git. Документы `01`–`06` зарезервированы под будущие команды и шаги.
