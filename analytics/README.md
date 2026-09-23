# Аналитика данных — Beeline Tariff Marketing Campaigns

Подготовка данных и априорной оценки эффектов для агента (`agent.py`).
Раздел в работе, обновляется по мере готовности шагов.

## Запуск

```bash
# пакет участника должен лежать в корне репозитория: beeline_case_participants/ (он в .gitignore)
pip install -r analytics/requirements.txt
python analytics/clean_data.py        # шаг 1: очистка + analytics/output/data_quality.json
```

Путь к данным можно переопределить: `--data-dir <папка пакета>` или переменная `BEELINE_DATA_DIR`.
