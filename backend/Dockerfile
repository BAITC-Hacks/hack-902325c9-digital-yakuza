FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BEELINE_AGENT_DIR=/app/beeline_agent

WORKDIR /app/backend

RUN groupadd --gid 1000 app && useradd --uid 1000 --gid app --create-home app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=app:app backend/ ./
COPY --chown=app:app beeline_agent/ /app/beeline_agent/

USER app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
