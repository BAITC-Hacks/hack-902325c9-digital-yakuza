# Аналитика данных — Beeline Tariff Marketing Campaigns

Подготовка данных и априорной оценки эффектов для агента (`agent.py`).
Раздел в работе, обновляется по мере готовности шагов.

## Запуск

```bash
# пакет участника должен лежать в корне репозитория: beeline_case_participants/ (он в .gitignore)
pip install -r analytics/requirements.txt
python analytics/clean_data.py        # шаг 1: очистка + analytics/output/data_quality.json
python analytics/build_prior.py       # шаг 2: априор эффектов → analytics/output/prior/
```

Главный выход для агента — `analytics/output/prior/prior_table.py`: словарь `PRIOR` по ключу
`(текущий тариф, ARPU-сегмент, целевой тариф)` и функции `prior_ratio()` / `posterior_base()`.
Модуль самодостаточный — его можно целиком вставить в `agent.py` (сдаётся только он).

Путь к данным можно переопределить: `--data-dir <папка пакета>` или переменная `BEELINE_DATA_DIR`.
