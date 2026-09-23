"""
Общие константы и утилиты аналитики (Beeline Tariff Marketing Campaigns Case).

Правила сегментации и расчёта эффекта повторяют среду кейса
(mock_environment.py / scoring_core.py), чтобы таблицы были сопоставимы с тем,
как среда считает результат.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "beeline_case_participants"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "output"

# --- правила из ТЗ / среды -------------------------------------------------
ARPU_BINS = [-np.inf, 1000, 5000, np.inf]          # как в mock_environment (pd.cut, right=True)
ARPU_LABELS = ["LOW", "MID", "HIGH"]
PREV_ARPU_MIN = 100                                  # ниже относительный эффект не определён (как в моке)
PCT_CLIP = (-1.0, 3.0)                               # клип относительного изменения ARPU (как в моке)
DIRECTION_THRESHOLD = 0.10                           # UPSELL / DOWNSELL / FLAT (feature_dictionary)
HISTORY_PRE_MONTHS = ["2026-07-01", "2026-08-01", "2026-09-01"]  # 3 мес. до смены тарифа (смена — 2026-10)

CHANNELS = {                                         # scoring_core.CHANNELS
    "push":        {"cost_per_contact": 0,   "conversion_multiplier": 0.50},
    "sms":         {"cost_per_contact": 4,   "conversion_multiplier": 0.65},
    "digital_ads": {"cost_per_contact": 22,  "conversion_multiplier": 0.85},
    "call":        {"cost_per_contact": 160, "conversion_multiplier": 1.20},
}
PILOT_NOISE_STD = 0.804                              # environment.PER_CUSTOMER_STD

TRAFFIC_COLS = [
    "OUT_LOC_ONNET_MIN", "OUT_LOC_OFFNET_MIN", "OUT_LOC_OFFNET_UNPAID_MIN", "OUT_LOC_OFFNET_PAID_MIN",
    "OUT_INTER_MIN", "OUT_LOC_LAND_MIN", "OUT_LOCAL_ONNET_SMS_AMT", "OUT_LOCAL_OFFNET_SMS_AMT",
    "OUT_LOCAL_LAND_PAID_SMS_AMT", "OUT_INTER_SMS_AMT", "DATA_VOLUME", "LTE_DATA_VOLUME",
    "TOTAL_ROAM_CALL_AMT", "TOTAL_ROAM_SMS_AMT", "TOTAL_ROAM_GPRS_MB", "COUNT_CONTACT",
    "AVG_TRANSACT_CONTACT", "SUM_TRANSACT_CONTACT", "AVG_DURATION_CONTACT", "COUNT_BASE_STATION",
]


def arpu_segment(values: pd.Series) -> pd.Series:
    """LOW / MID / HIGH по порогам 1000 / 5000 (как в среде)."""
    return pd.cut(values, bins=ARPU_BINS, labels=ARPU_LABELS).astype(object)


def data_segment(data_volume: pd.Series) -> pd.Series:
    """NON_USER (0 МБ) / LITE (0–2000] / HEAVY (>2000) по DATA_VOLUME — точно воспроизводит профиль."""
    out = pd.Series(np.where(data_volume <= 0, "NON_USER",
                             np.where(data_volume <= 2000, "LITE", "HEAVY")), index=data_volume.index, dtype=object)
    return out.where(data_volume.notna(), None)


def call_segment(onnet_min: pd.Series, offnet_min: pd.Series) -> pd.Series:
    """LOW (<100) / MEDIUM (100–400) / HIGH (>400) по ONNET+OFFNET минутам; пропуск → LOW (как в профиле)."""
    total = (onnet_min + offnet_min).fillna(0)
    return pd.Series(np.where(total < 100, "LOW", np.where(total <= 400, "MEDIUM", "HIGH")),
                     index=total.index, dtype=object)


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-dir", default=os.environ.get("BEELINE_DATA_DIR", str(DEFAULT_DATA_DIR)),
                        help="папка пакета участника (customer_profile.csv, data/...)")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR), help="папка для результатов")
    return parser.parse_args()


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else round(float(o), 6)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)
