"""
Агент тарифных маркетинговых кампаний Beeline (HackAlem AI).

Прозрачный детерминированный workflow — каждое решение видно в self.trace:
  1. Априор: история смен тарифов (вшитая таблица PRIOR) → оценка эффекта связки
  2. Кандидаты: ARPU-сегмент × целевой тариф × список текущих тарифов
  3. Пилоты: проверяем кандидата с лучшей оптимистичной оценкой ровно по его фильтрам
  4. Байесовское обновление оценки после каждого пилота (учёт размера пилота)
  5. План: до 10 кампаний, у которых нижняя граница оценки > 0
  6. Каналы и лимиты: бюджет, охват, 5 000 абонентов на кампанию
  7. Проверка плана; при ошибке — запасной план только из подтверждённых пилотами кампаний

Агент использует только публичный интерфейс env и не читает файлы во время работы.
"""

import math
import time
from dataclasses import dataclass, field

import pandas as pd

SEGMENTS = ("LOW", "MID", "HIGH")
PILOT_NOISE_SD = 0.804          # разброс эффекта на одного абонента (environment.py)
MAX_CAMPAIGNS = 10
MAX_PER_CAMPAIGN = 5000
MIN_CANDIDATE_SIZE = 30

# Параметры стратегии. Подбираются экспериментом на искажённых мирах, а не под мок-среду.
PRIOR_SHRINK = 0.5              # история описывает другую выборку → доверяем наполовину
PRIOR_MODEL_SD = 0.10           # неопределённость переноса истории на целевую аудиторию
RISK_K = 1.0                    # кампания идёт в план, если mu − RISK_K·sd > 0
MAX_PILOTS_PER_CANDIDATE = 2
PILOT_CHANNEL = "sms"           # расчёт ожидаемой пользы пилота: sms-200 — лучший вариант по умолчанию
PILOT_N_LARGE, PILOT_N_SMALL, SMALL_SEGMENT = 200, 100, 600
TIME_BUDGET_S = 180

# === PRIOR START (генерируется prior/build_prior.py) ===
PRIOR = {
    ("tariff_1", "HIGH", "tariff_10"): (0.02239, 9, 0.5516),
    ("tariff_1", "HIGH", "tariff_11"): (0.00135, 4, 0.3518),
    ("tariff_1", "HIGH", "tariff_12"): (0.01155, 2, 0.0848),
    ("tariff_1", "HIGH", "tariff_13"): (-0.06995, 14, 0.2951),
    ("tariff_1", "HIGH", "tariff_4"): (-0.00447, 10, 0.6423),
    ("tariff_1", "HIGH", "tariff_8"): (-0.10690, 32, 0.4638),
    ("tariff_1", "LOW", "tariff_10"): (0.12181, 14, 1.5556),
    ("tariff_1", "LOW", "tariff_11"): (0.01078, 2, 2.4520),
    ("tariff_1", "LOW", "tariff_13"): (0.19859, 22, 1.0300),
    ("tariff_1", "LOW", "tariff_21"): (0.01785, 10, 1.6558),
    ("tariff_1", "LOW", "tariff_4"): (0.29553, 35, 1.5800),
    ("tariff_1", "LOW", "tariff_8"): (0.46793, 71, 1.6935),
    ("tariff_1", "LOW", "tariff_9"): (0.68876, 81, 1.4769),
    ("tariff_1", "MID", "tariff_10"): (0.07300, 26, 1.2201),
    ("tariff_1", "MID", "tariff_11"): (0.04546, 12, 1.5257),
    ("tariff_1", "MID", "tariff_12"): (0.01152, 2, 1.6762),
    ("tariff_1", "MID", "tariff_13"): (0.06968, 67, 0.7628),
    ("tariff_1", "MID", "tariff_21"): (0.01400, 5, 1.8825),
    ("tariff_1", "MID", "tariff_4"): (0.16142, 51, 1.1960),
    ("tariff_1", "MID", "tariff_8"): (0.31156, 125, 1.2678),
    ("tariff_1", "MID", "tariff_9"): (0.11395, 27, 1.1770),
    ("tariff_10", "HIGH", "tariff_1"): (-0.01449, 17, 0.4221),
    ("tariff_10", "HIGH", "tariff_11"): (-0.01146, 134, 0.4627),
    ("tariff_10", "HIGH", "tariff_12"): (-0.00898, 91, 0.4550),
    ("tariff_10", "HIGH", "tariff_13"): (-0.01703, 33, 0.3548),
    ("tariff_10", "HIGH", "tariff_4"): (-0.03932, 77, 0.4079),
    ("tariff_10", "HIGH", "tariff_8"): (-0.06406, 324, 0.3812),
    ("tariff_10", "LOW", "tariff_1"): (-0.01439, 1, 0.9085),
    ("tariff_10", "LOW", "tariff_11"): (-0.00221, 11, 1.5664),
    ("tariff_10", "LOW", "tariff_12"): (-0.04045, 4, 0.7864),
    ("tariff_10", "LOW", "tariff_13"): (0.06635, 2, 1.4277),
    ("tariff_10", "LOW", "tariff_21"): (0.03333, 2, 2.8284),
    ("tariff_10", "LOW", "tariff_4"): (0.09197, 13, 1.3894),
    ("tariff_10", "LOW", "tariff_8"): (0.05671, 19, 1.7182),
    ("tariff_10", "LOW", "tariff_9"): (0.25874, 8, 1.7470),
    ("tariff_10", "MID", "tariff_1"): (-0.02465, 12, 0.2641),
    ("tariff_10", "MID", "tariff_11"): (0.09866, 86, 0.9823),
    ("tariff_10", "MID", "tariff_12"): (0.02509, 30, 0.8929),
    ("tariff_10", "MID", "tariff_13"): (-0.01528, 31, 0.7010),
    ("tariff_10", "MID", "tariff_21"): (-0.00060, 1, 0.9085),
    ("tariff_10", "MID", "tariff_4"): (0.00107, 60, 0.8414),
    ("tariff_10", "MID", "tariff_8"): (0.08338, 138, 0.7745),
    ("tariff_10", "MID", "tariff_9"): (0.00188, 3, 0.8072),
    ("tariff_11", "HIGH", "tariff_1"): (-0.00304, 5, 0.3011),
    ("tariff_11", "HIGH", "tariff_10"): (-0.01122, 110, 0.4259),
    ("tariff_11", "HIGH", "tariff_12"): (0.01668, 107, 0.5272),
    ("tariff_11", "HIGH", "tariff_13"): (-0.02004, 20, 0.3250),
    ("tariff_11", "HIGH", "tariff_4"): (-0.02370, 42, 0.5891),
    ("tariff_11", "HIGH", "tariff_8"): (-0.06896, 213, 0.4232),
    ("tariff_11", "LOW", "tariff_10"): (-0.10314, 14, 1.0761),
    ("tariff_11", "LOW", "tariff_12"): (0.03765, 20, 1.5001),
    ("tariff_11", "LOW", "tariff_13"): (0.00654, 4, 1.9293),
    ("tariff_11", "LOW", "tariff_21"): (0.04348, 1, 0.9085),
    ("tariff_11", "LOW", "tariff_4"): (0.01125, 10, 1.7069),
    ("tariff_11", "LOW", "tariff_8"): (0.28328, 16, 1.8073),
    ("tariff_11", "LOW", "tariff_9"): (0.12113, 4, 1.8208),
    ("tariff_11", "MID", "tariff_1"): (-0.00613, 6, 0.5702),
    ("tariff_11", "MID", "tariff_10"): (0.02831, 29, 1.0698),
    ("tariff_11", "MID", "tariff_12"): (0.02573, 62, 0.9887),
    ("tariff_11", "MID", "tariff_13"): (-0.00472, 7, 0.4599),
    ("tariff_11", "MID", "tariff_4"): (0.03606, 42, 1.0063),
    ("tariff_11", "MID", "tariff_8"): (0.09589, 103, 0.8263),
    ("tariff_11", "MID", "tariff_9"): (0.00843, 3, 0.9949),
    ("tariff_12", "HIGH", "tariff_1"): (-0.00942, 5, 0.3772),
    ("tariff_12", "HIGH", "tariff_10"): (-0.00207, 85, 0.4688),
    ("tariff_12", "HIGH", "tariff_11"): (0.01154, 82, 0.5286),
    ("tariff_12", "HIGH", "tariff_13"): (-0.00894, 9, 0.3907),
    ("tariff_12", "HIGH", "tariff_4"): (-0.01080, 15, 0.3079),
    ("tariff_12", "HIGH", "tariff_8"): (-0.04363, 168, 0.4038),
    ("tariff_12", "HIGH", "tariff_9"): (-0.00049, 1, 0.9085),
    ("tariff_12", "LOW", "tariff_10"): (-0.18097, 8, 0.6530),
    ("tariff_12", "LOW", "tariff_11"): (0.19866, 6, 2.0715),
    ("tariff_12", "LOW", "tariff_13"): (0.03864, 3, 2.2244),
    ("tariff_12", "LOW", "tariff_4"): (-0.22070, 10, 0.6818),
    ("tariff_12", "LOW", "tariff_8"): (0.29204, 6, 1.8958),
    ("tariff_12", "LOW", "tariff_9"): (-0.02941, 1, 0.9085),
    ("tariff_12", "MID", "tariff_1"): (-0.00843, 2, 0.5555),
    ("tariff_12", "MID", "tariff_10"): (0.04000, 22, 0.9880),
    ("tariff_12", "MID", "tariff_11"): (-0.03777, 22, 0.8139),
    ("tariff_12", "MID", "tariff_13"): (-0.00467, 9, 0.6129),
    ("tariff_12", "MID", "tariff_4"): (0.00795, 10, 1.0842),
    ("tariff_12", "MID", "tariff_8"): (0.27756, 78, 0.9385),
    ("tariff_12", "MID", "tariff_9"): (0.00946, 1, 0.9085),
    ("tariff_13", "HIGH", "tariff_1"): (-0.02267, 7, 0.8126),
    ("tariff_13", "HIGH", "tariff_10"): (0.00982, 9, 0.4955),
    ("tariff_13", "HIGH", "tariff_11"): (-0.02292, 6, 0.6767),
    ("tariff_13", "HIGH", "tariff_12"): (-0.01406, 9, 0.4448),
    ("tariff_13", "HIGH", "tariff_4"): (-0.03351, 22, 0.3297),
    ("tariff_13", "HIGH", "tariff_8"): (0.02442, 33, 0.6713),
    ("tariff_13", "LOW", "tariff_1"): (0.07736, 3, 1.9381),
    ("tariff_13", "LOW", "tariff_10"): (0.13953, 2, 0.0000),
    ("tariff_13", "LOW", "tariff_11"): (0.03489, 1, 0.9085),
    ("tariff_13", "LOW", "tariff_21"): (-0.02278, 2, 0.7214),
    ("tariff_13", "LOW", "tariff_4"): (0.00839, 7, 1.4326),
    ("tariff_13", "LOW", "tariff_8"): (0.57401, 20, 1.6113),
    ("tariff_13", "LOW", "tariff_9"): (0.28584, 8, 2.0219),
    ("tariff_13", "MID", "tariff_1"): (-0.04919, 41, 0.6401),
    ("tariff_13", "MID", "tariff_10"): (0.04262, 29, 1.0978),
    ("tariff_13", "MID", "tariff_11"): (0.00712, 9, 0.6692),
    ("tariff_13", "MID", "tariff_12"): (-0.00341, 3, 0.7802),
    ("tariff_13", "MID", "tariff_21"): (0.00435, 2, 0.0118),
    ("tariff_13", "MID", "tariff_4"): (0.03615, 73, 0.6825),
    ("tariff_13", "MID", "tariff_8"): (0.25640, 234, 0.9723),
    ("tariff_13", "MID", "tariff_9"): (0.02844, 28, 0.7763),
    ("tariff_14", "HIGH", "tariff_1"): (-0.03417, 4, 0.4577),
    ("tariff_14", "HIGH", "tariff_10"): (-0.05032, 11, 0.5533),
    ("tariff_14", "HIGH", "tariff_11"): (-0.00673, 4, 0.5761),
    ("tariff_14", "HIGH", "tariff_12"): (-0.01424, 2, 0.7196),
    ("tariff_14", "HIGH", "tariff_13"): (-0.03026, 4, 0.3450),
    ("tariff_14", "HIGH", "tariff_4"): (-0.03983, 9, 0.3210),
    ("tariff_14", "HIGH", "tariff_8"): (-0.04008, 35, 0.4066),
    ("tariff_14", "LOW", "tariff_1"): (0.11321, 2, 0.0000),
    ("tariff_14", "LOW", "tariff_10"): (0.00526, 4, 1.9579),
    ("tariff_14", "LOW", "tariff_11"): (-0.01887, 1, 0.9085),
    ("tariff_14", "LOW", "tariff_12"): (0.00220, 4, 1.9813),
    ("tariff_14", "LOW", "tariff_13"): (0.04556, 3, 2.0284),
    ("tariff_14", "LOW", "tariff_21"): (0.06473, 3, 1.8740),
    ("tariff_14", "LOW", "tariff_4"): (0.14420, 6, 1.8952),
    ("tariff_14", "LOW", "tariff_8"): (0.28459, 9, 1.9863),
    ("tariff_14", "LOW", "tariff_9"): (0.48488, 21, 1.6396),
    ("tariff_14", "MID", "tariff_1"): (-0.03518, 8, 0.4722),
    ("tariff_14", "MID", "tariff_10"): (0.00376, 17, 1.1131),
    ("tariff_14", "MID", "tariff_11"): (-0.01186, 3, 0.9239),
    ("tariff_14", "MID", "tariff_12"): (-0.05022, 6, 0.0194),
    ("tariff_14", "MID", "tariff_13"): (0.01074, 9, 1.1938),
    ("tariff_14", "MID", "tariff_21"): (-0.00847, 1, 0.9085),
    ("tariff_14", "MID", "tariff_4"): (-0.00722, 25, 0.6314),
    ("tariff_14", "MID", "tariff_8"): (-0.03040, 42, 0.9443),
    ("tariff_14", "MID", "tariff_9"): (0.05324, 7, 1.2679),
    ("tariff_15", "HIGH", "tariff_1"): (0.00147, 3, 0.9328),
    ("tariff_15", "HIGH", "tariff_10"): (0.01766, 9, 0.4036),
    ("tariff_15", "HIGH", "tariff_11"): (-0.00593, 5, 0.5876),
    ("tariff_15", "HIGH", "tariff_12"): (0.02075, 3, 0.5682),
    ("tariff_15", "HIGH", "tariff_13"): (-0.03346, 11, 0.2583),
    ("tariff_15", "HIGH", "tariff_4"): (-0.01554, 16, 0.2607),
    ("tariff_15", "HIGH", "tariff_8"): (-0.03682, 30, 0.4042),
    ("tariff_15", "LOW", "tariff_1"): (0.03349, 4, 1.6156),
    ("tariff_15", "LOW", "tariff_10"): (0.10191, 4, 1.9746),
    ("tariff_15", "LOW", "tariff_11"): (-0.01266, 1, 0.9085),
    ("tariff_15", "LOW", "tariff_13"): (0.27785, 14, 1.4955),
    ("tariff_15", "LOW", "tariff_21"): (0.09626, 8, 1.7535),
    ("tariff_15", "LOW", "tariff_4"): (0.08525, 7, 1.8567),
    ("tariff_15", "LOW", "tariff_8"): (0.39441, 21, 1.7091),
    ("tariff_15", "LOW", "tariff_9"): (0.39451, 20, 1.7076),
    ("tariff_15", "MID", "tariff_1"): (-0.01787, 54, 0.5121),
    ("tariff_15", "MID", "tariff_10"): (0.06701, 180, 0.7478),
    ("tariff_15", "MID", "tariff_11"): (0.00845, 23, 0.9260),
    ("tariff_15", "MID", "tariff_12"): (0.00718, 4, 1.0092),
    ("tariff_15", "MID", "tariff_13"): (0.00695, 348, 0.5180),
    ("tariff_15", "MID", "tariff_21"): (0.00156, 8, 0.8393),
    ("tariff_15", "MID", "tariff_4"): (0.04346, 133, 0.7852),
    ("tariff_15", "MID", "tariff_8"): (0.15908, 430, 0.7584),
    ("tariff_15", "MID", "tariff_9"): (0.03638, 133, 0.5921),
    ("tariff_16", "HIGH", "tariff_1"): (-0.01773, 1, 0.9085),
    ("tariff_16", "HIGH", "tariff_11"): (0.00584, 3, 0.5419),
    ("tariff_16", "HIGH", "tariff_13"): (-0.03531, 4, 0.8606),
    ("tariff_16", "HIGH", "tariff_4"): (0.01486, 3, 0.5319),
    ("tariff_16", "HIGH", "tariff_8"): (0.01439, 4, 0.1976),
    ("tariff_16", "LOW", "tariff_1"): (0.03648, 27, 1.2173),
    ("tariff_16", "LOW", "tariff_10"): (0.00922, 2, 2.8284),
    ("tariff_16", "LOW", "tariff_11"): (0.04018, 4, 1.6408),
    ("tariff_16", "LOW", "tariff_13"): (0.09694, 9, 0.9228),
    ("tariff_16", "LOW", "tariff_21"): (0.13863, 29, 1.6474),
    ("tariff_16", "LOW", "tariff_4"): (0.10457, 13, 1.6985),
    ("tariff_16", "LOW", "tariff_8"): (0.22334, 25, 1.4304),
    ("tariff_16", "LOW", "tariff_9"): (0.61955, 108, 1.7156),
    ("tariff_16", "MID", "tariff_1"): (-0.09484, 48, 0.5157),
    ("tariff_16", "MID", "tariff_10"): (0.02891, 4, 1.2026),
    ("tariff_16", "MID", "tariff_11"): (0.03958, 10, 1.3540),
    ("tariff_16", "MID", "tariff_12"): (-0.00260, 1, 0.9085),
    ("tariff_16", "MID", "tariff_13"): (0.04515, 43, 0.7823),
    ("tariff_16", "MID", "tariff_21"): (0.02758, 7, 1.7154),
    ("tariff_16", "MID", "tariff_4"): (0.04010, 15, 0.9361),
    ("tariff_16", "MID", "tariff_8"): (0.17826, 44, 1.1459),
    ("tariff_16", "MID", "tariff_9"): (0.09170, 22, 1.3363),
    ("tariff_17", "HIGH", "tariff_1"): (-0.01031, 3, 0.5308),
    ("tariff_17", "HIGH", "tariff_10"): (0.01404, 14, 0.3156),
    ("tariff_17", "HIGH", "tariff_11"): (0.01481, 11, 0.4698),
    ("tariff_17", "HIGH", "tariff_12"): (-0.00212, 6, 0.4915),
    ("tariff_17", "HIGH", "tariff_13"): (-0.02535, 16, 0.4128),
    ("tariff_17", "HIGH", "tariff_4"): (0.01055, 30, 0.6797),
    ("tariff_17", "HIGH", "tariff_8"): (-0.02981, 114, 0.3231),
    ("tariff_17", "LOW", "tariff_1"): (-0.00982, 1, 0.9085),
    ("tariff_17", "LOW", "tariff_12"): (0.10714, 1, 0.9085),
    ("tariff_17", "LOW", "tariff_13"): (0.05336, 3, 2.1808),
    ("tariff_17", "LOW", "tariff_4"): (0.11094, 4, 1.5793),
    ("tariff_17", "LOW", "tariff_8"): (0.54229, 15, 1.7582),
    ("tariff_17", "LOW", "tariff_9"): (0.30987, 4, 1.6618),
    ("tariff_17", "MID", "tariff_1"): (-0.02350, 23, 0.9140),
    ("tariff_17", "MID", "tariff_10"): (0.03153, 16, 0.8451),
    ("tariff_17", "MID", "tariff_11"): (0.01535, 23, 0.7730),
    ("tariff_17", "MID", "tariff_12"): (0.01039, 4, 0.7612),
    ("tariff_17", "MID", "tariff_13"): (-0.00600, 35, 0.6564),
    ("tariff_17", "MID", "tariff_4"): (0.03156, 45, 0.4728),
    ("tariff_17", "MID", "tariff_8"): (0.22681, 127, 0.8340),
    ("tariff_17", "MID", "tariff_9"): (0.02563, 5, 1.5186),
    ("tariff_18", "HIGH", "tariff_1"): (-0.01595, 8, 0.4655),
    ("tariff_18", "HIGH", "tariff_10"): (0.00029, 67, 0.4101),
    ("tariff_18", "HIGH", "tariff_11"): (0.01574, 22, 0.6038),
    ("tariff_18", "HIGH", "tariff_12"): (-0.00288, 26, 0.2983),
    ("tariff_18", "HIGH", "tariff_13"): (-0.02668, 23, 0.4305),
    ("tariff_18", "HIGH", "tariff_4"): (-0.02362, 49, 0.3599),
    ("tariff_18", "HIGH", "tariff_8"): (-0.03448, 127, 0.3488),
    ("tariff_18", "LOW", "tariff_1"): (-0.02275, 4, 0.4417),
    ("tariff_18", "LOW", "tariff_10"): (0.10603, 9, 1.9312),
    ("tariff_18", "LOW", "tariff_11"): (0.11538, 2, 0.0000),
    ("tariff_18", "LOW", "tariff_12"): (0.05769, 1, 0.9085),
    ("tariff_18", "LOW", "tariff_13"): (0.14219, 6, 1.9467),
    ("tariff_18", "LOW", "tariff_4"): (0.10749, 3, 1.9690),
    ("tariff_18", "LOW", "tariff_8"): (0.55505, 14, 1.6692),
    ("tariff_18", "LOW", "tariff_9"): (0.43022, 13, 1.6847),
    ("tariff_18", "MID", "tariff_1"): (-0.02492, 5, 0.1118),
    ("tariff_18", "MID", "tariff_10"): (0.11220, 27, 1.1290),
    ("tariff_18", "MID", "tariff_11"): (0.03771, 5, 1.2255),
    ("tariff_18", "MID", "tariff_12"): (0.05510, 6, 1.3768),
    ("tariff_18", "MID", "tariff_13"): (0.03299, 25, 0.7991),
    ("tariff_18", "MID", "tariff_4"): (0.05570, 15, 1.3029),
    ("tariff_18", "MID", "tariff_8"): (0.20688, 50, 0.9244),
    ("tariff_18", "MID", "tariff_9"): (0.02179, 3, 1.9736),
    ("tariff_19", "HIGH", "tariff_1"): (-0.02521, 3, 0.4236),
    ("tariff_19", "HIGH", "tariff_10"): (-0.00011, 2, 0.1314),
    ("tariff_19", "HIGH", "tariff_11"): (-0.00090, 5, 0.9167),
    ("tariff_19", "HIGH", "tariff_12"): (-0.02240, 6, 0.3869),
    ("tariff_19", "HIGH", "tariff_13"): (-0.04094, 5, 0.2283),
    ("tariff_19", "HIGH", "tariff_4"): (-0.02260, 5, 0.4766),
    ("tariff_19", "HIGH", "tariff_8"): (-0.17148, 35, 0.3631),
    ("tariff_19", "LOW", "tariff_1"): (-0.08798, 8, 0.4492),
    ("tariff_19", "LOW", "tariff_10"): (0.05197, 3, 2.0171),
    ("tariff_19", "LOW", "tariff_11"): (-0.01349, 3, 1.3477),
    ("tariff_19", "LOW", "tariff_13"): (0.29092, 8, 1.5876),
    ("tariff_19", "LOW", "tariff_21"): (0.06122, 1, 0.9085),
    ("tariff_19", "LOW", "tariff_4"): (0.10685, 3, 2.1732),
    ("tariff_19", "LOW", "tariff_8"): (0.07545, 11, 1.7363),
    ("tariff_19", "LOW", "tariff_9"): (0.27017, 12, 1.9855),
    ("tariff_19", "MID", "tariff_1"): (-0.01816, 3, 0.7590),
    ("tariff_19", "MID", "tariff_10"): (0.04366, 13, 0.9085),
    ("tariff_19", "MID", "tariff_11"): (0.01673, 4, 1.8460),
    ("tariff_19", "MID", "tariff_13"): (0.02800, 12, 0.5656),
    ("tariff_19", "MID", "tariff_4"): (0.04285, 16, 1.1444),
    ("tariff_19", "MID", "tariff_8"): (0.21246, 21, 1.1150),
    ("tariff_19", "MID", "tariff_9"): (0.03623, 7, 1.0786),
    ("tariff_2", "HIGH", "tariff_1"): (-0.02489, 1, 0.9085),
    ("tariff_2", "HIGH", "tariff_10"): (0.00348, 2, 0.3658),
    ("tariff_2", "HIGH", "tariff_11"): (-0.00764, 1, 0.9085),
    ("tariff_2", "HIGH", "tariff_13"): (-0.00834, 3, 0.4897),
    ("tariff_2", "HIGH", "tariff_4"): (0.00250, 5, 0.3890),
    ("tariff_2", "HIGH", "tariff_8"): (0.00976, 16, 0.4514),
    ("tariff_2", "LOW", "tariff_1"): (0.03341, 12, 1.4302),
    ("tariff_2", "LOW", "tariff_10"): (0.12359, 6, 2.0627),
    ("tariff_2", "LOW", "tariff_13"): (0.40814, 15, 1.3731),
    ("tariff_2", "LOW", "tariff_21"): (0.16431, 13, 1.8360),
    ("tariff_2", "LOW", "tariff_4"): (0.03984, 4, 1.6857),
    ("tariff_2", "LOW", "tariff_8"): (0.18157, 17, 1.4376),
    ("tariff_2", "LOW", "tariff_9"): (0.19119, 14, 1.8720),
    ("tariff_2", "MID", "tariff_1"): (-0.03456, 20, 0.4343),
    ("tariff_2", "MID", "tariff_10"): (0.02981, 30, 0.6958),
    ("tariff_2", "MID", "tariff_11"): (0.00517, 4, 0.8181),
    ("tariff_2", "MID", "tariff_12"): (0.01040, 5, 1.2677),
    ("tariff_2", "MID", "tariff_13"): (0.01555, 58, 0.6475),
    ("tariff_2", "MID", "tariff_21"): (-0.00524, 3, 0.4705),
    ("tariff_2", "MID", "tariff_4"): (0.03586, 23, 0.9190),
    ("tariff_2", "MID", "tariff_8"): (0.13816, 114, 0.9005),
    ("tariff_2", "MID", "tariff_9"): (0.02388, 42, 0.6789),
    ("tariff_20", "HIGH", "tariff_1"): (-0.04855, 11, 0.3404),
    ("tariff_20", "HIGH", "tariff_10"): (0.00887, 17, 0.4015),
    ("tariff_20", "HIGH", "tariff_11"): (0.00223, 8, 0.5145),
    ("tariff_20", "HIGH", "tariff_12"): (-0.00431, 9, 0.6106),
    ("tariff_20", "HIGH", "tariff_13"): (-0.02767, 17, 0.3630),
    ("tariff_20", "HIGH", "tariff_4"): (-0.03785, 29, 0.3194),
    ("tariff_20", "HIGH", "tariff_8"): (-0.02501, 78, 0.2317),
    ("tariff_20", "LOW", "tariff_1"): (0.09868, 4, 1.3827),
    ("tariff_20", "LOW", "tariff_10"): (0.18035, 2, 1.5646),
    ("tariff_20", "LOW", "tariff_11"): (-0.04762, 1, 0.9085),
    ("tariff_20", "LOW", "tariff_12"): (-0.03860, 1, 0.9085),
    ("tariff_20", "LOW", "tariff_13"): (0.51221, 5, 0.9869),
    ("tariff_20", "LOW", "tariff_4"): (-0.09383, 2, 0.0209),
    ("tariff_20", "LOW", "tariff_8"): (0.32874, 4, 1.7294),
    ("tariff_20", "LOW", "tariff_9"): (0.09698, 2, 2.8025),
    ("tariff_20", "MID", "tariff_1"): (-0.01980, 9, 1.1310),
    ("tariff_20", "MID", "tariff_10"): (0.01242, 5, 1.1015),
    ("tariff_20", "MID", "tariff_11"): (0.01291, 2, 0.0011),
    ("tariff_20", "MID", "tariff_12"): (0.01609, 3, 1.3970),
    ("tariff_20", "MID", "tariff_13"): (-0.01772, 8, 0.2371),
    ("tariff_20", "MID", "tariff_4"): (-0.03772, 8, 0.5245),
    ("tariff_20", "MID", "tariff_8"): (0.18901, 31, 0.8458),
    ("tariff_20", "MID", "tariff_9"): (0.02919, 1, 0.9085),
    ("tariff_3", "HIGH", "tariff_1"): (-0.01448, 3, 0.4745),
    ("tariff_3", "HIGH", "tariff_10"): (-0.03047, 19, 0.5472),
    ("tariff_3", "HIGH", "tariff_11"): (-0.01123, 11, 0.4984),
    ("tariff_3", "HIGH", "tariff_12"): (0.01097, 10, 0.5872),
    ("tariff_3", "HIGH", "tariff_13"): (-0.01168, 5, 0.1478),
    ("tariff_3", "HIGH", "tariff_4"): (-0.01826, 6, 0.3402),
    ("tariff_3", "HIGH", "tariff_8"): (-0.04817, 83, 0.5487),
    ("tariff_3", "LOW", "tariff_1"): (-0.01708, 8, 1.4948),
    ("tariff_3", "LOW", "tariff_10"): (0.10698, 5, 1.7691),
    ("tariff_3", "LOW", "tariff_11"): (0.05941, 2, 0.0000),
    ("tariff_3", "LOW", "tariff_12"): (-0.00466, 5, 1.7448),
    ("tariff_3", "LOW", "tariff_13"): (0.05882, 6, 1.7199),
    ("tariff_3", "LOW", "tariff_21"): (-0.01980, 2, 0.0000),
    ("tariff_3", "LOW", "tariff_4"): (-0.01154, 15, 1.3891),
    ("tariff_3", "LOW", "tariff_8"): (0.15924, 27, 1.8268),
    ("tariff_3", "LOW", "tariff_9"): (0.33379, 31, 1.8583),
    ("tariff_3", "MID", "tariff_1"): (-0.02398, 21, 0.8945),
    ("tariff_3", "MID", "tariff_10"): (0.04069, 31, 1.4708),
    ("tariff_3", "MID", "tariff_11"): (0.03715, 29, 1.2876),
    ("tariff_3", "MID", "tariff_12"): (0.00793, 8, 1.4240),
    ("tariff_3", "MID", "tariff_13"): (0.01345, 32, 0.8272),
    ("tariff_3", "MID", "tariff_21"): (-0.00017, 8, 0.9978),
    ("tariff_3", "MID", "tariff_4"): (0.01454, 47, 1.1206),
    ("tariff_3", "MID", "tariff_8"): (0.31464, 241, 1.3373),
    ("tariff_3", "MID", "tariff_9"): (0.01131, 30, 0.8097),
    ("tariff_4", "HIGH", "tariff_1"): (-0.02418, 21, 0.3347),
    ("tariff_4", "HIGH", "tariff_10"): (-0.01217, 45, 0.4218),
    ("tariff_4", "HIGH", "tariff_11"): (-0.01024, 32, 0.6179),
    ("tariff_4", "HIGH", "tariff_12"): (0.00024, 19, 0.5539),
    ("tariff_4", "HIGH", "tariff_13"): (-0.02974, 31, 0.3252),
    ("tariff_4", "HIGH", "tariff_8"): (-0.09537, 340, 0.3760),
    ("tariff_4", "LOW", "tariff_1"): (0.02769, 3, 1.1763),
    ("tariff_4", "LOW", "tariff_10"): (0.18513, 17, 1.8912),
    ("tariff_4", "LOW", "tariff_11"): (-0.10285, 13, 0.5655),
    ("tariff_4", "LOW", "tariff_12"): (0.01935, 9, 1.3629),
    ("tariff_4", "LOW", "tariff_13"): (0.12338, 8, 1.4943),
    ("tariff_4", "LOW", "tariff_21"): (0.02421, 3, 1.9550),
    ("tariff_4", "LOW", "tariff_8"): (0.05189, 22, 1.5585),
    ("tariff_4", "LOW", "tariff_9"): (0.41827, 17, 1.3799),
    ("tariff_4", "MID", "tariff_1"): (-0.02643, 37, 0.6197),
    ("tariff_4", "MID", "tariff_10"): (0.00945, 58, 0.9516),
    ("tariff_4", "MID", "tariff_11"): (0.02046, 41, 1.1860),
    ("tariff_4", "MID", "tariff_12"): (-0.00733, 13, 0.8393),
    ("tariff_4", "MID", "tariff_13"): (-0.00728, 100, 0.6882),
    ("tariff_4", "MID", "tariff_21"): (0.00192, 2, 0.4455),
    ("tariff_4", "MID", "tariff_8"): (0.24024, 408, 0.9276),
    ("tariff_4", "MID", "tariff_9"): (0.01626, 11, 1.2196),
    ("tariff_5", "HIGH", "tariff_1"): (-0.06602, 9, 0.3228),
    ("tariff_5", "HIGH", "tariff_10"): (0.00772, 32, 0.6684),
    ("tariff_5", "HIGH", "tariff_11"): (0.00876, 18, 0.9012),
    ("tariff_5", "HIGH", "tariff_12"): (-0.07363, 10, 0.3568),
    ("tariff_5", "HIGH", "tariff_13"): (-0.02785, 6, 0.3367),
    ("tariff_5", "HIGH", "tariff_4"): (-0.07759, 13, 0.4052),
    ("tariff_5", "HIGH", "tariff_8"): (-0.00388, 13, 0.9619),
    ("tariff_5", "LOW", "tariff_1"): (-0.02891, 10, 1.2181),
    ("tariff_5", "LOW", "tariff_10"): (0.04152, 32, 1.6293),
    ("tariff_5", "LOW", "tariff_11"): (-0.03704, 9, 1.3305),
    ("tariff_5", "LOW", "tariff_12"): (-0.05095, 8, 0.2793),
    ("tariff_5", "LOW", "tariff_13"): (0.33259, 36, 1.6528),
    ("tariff_5", "LOW", "tariff_21"): (-0.00752, 1, 0.9085),
    ("tariff_5", "LOW", "tariff_4"): (0.13197, 20, 1.7212),
    ("tariff_5", "LOW", "tariff_8"): (0.07640, 6, 1.9448),
    ("tariff_5", "LOW", "tariff_9"): (0.05277, 11, 1.9046),
    ("tariff_5", "MID", "tariff_1"): (-0.02761, 18, 1.2664),
    ("tariff_5", "MID", "tariff_10"): (-0.04150, 36, 1.0344),
    ("tariff_5", "MID", "tariff_11"): (-0.00656, 15, 1.2377),
    ("tariff_5", "MID", "tariff_12"): (-0.01829, 17, 1.1488),
    ("tariff_5", "MID", "tariff_13"): (-0.01005, 43, 0.7480),
    ("tariff_5", "MID", "tariff_21"): (-0.00808, 4, 1.2402),
    ("tariff_5", "MID", "tariff_4"): (0.06803, 32, 1.2410),
    ("tariff_5", "MID", "tariff_8"): (-0.00657, 21, 1.0641),
    ("tariff_5", "MID", "tariff_9"): (-0.00426, 2, 0.5049),
    ("tariff_6", "HIGH", "tariff_1"): (-0.05203, 4, 0.6258),
    ("tariff_6", "HIGH", "tariff_10"): (-0.08328, 9, 0.4824),
    ("tariff_6", "HIGH", "tariff_11"): (0.01763, 11, 0.5183),
    ("tariff_6", "HIGH", "tariff_12"): (-0.03990, 8, 1.2421),
    ("tariff_6", "HIGH", "tariff_13"): (-0.05544, 6, 0.5859),
    ("tariff_6", "HIGH", "tariff_4"): (-0.09223, 10, 0.5168),
    ("tariff_6", "HIGH", "tariff_8"): (-0.03453, 3, 0.3702),
    ("tariff_6", "HIGH", "tariff_9"): (-0.00298, 1, 0.9085),
    ("tariff_6", "LOW", "tariff_1"): (0.08790, 13, 1.4416),
    ("tariff_6", "LOW", "tariff_10"): (-0.01374, 21, 1.4591),
    ("tariff_6", "LOW", "tariff_11"): (0.00360, 7, 1.4157),
    ("tariff_6", "LOW", "tariff_12"): (0.03788, 3, 2.3094),
    ("tariff_6", "LOW", "tariff_13"): (0.40817, 37, 1.7456),
    ("tariff_6", "LOW", "tariff_4"): (0.27290, 23, 1.6738),
    ("tariff_6", "LOW", "tariff_8"): (0.07837, 10, 1.7865),
    ("tariff_6", "LOW", "tariff_9"): (0.06665, 18, 1.6453),
    ("tariff_6", "MID", "tariff_1"): (-0.05883, 10, 0.4947),
    ("tariff_6", "MID", "tariff_10"): (0.02739, 25, 1.4001),
    ("tariff_6", "MID", "tariff_11"): (0.08274, 10, 1.4504),
    ("tariff_6", "MID", "tariff_12"): (-0.00393, 2, 1.1336),
    ("tariff_6", "MID", "tariff_13"): (-0.09459, 28, 0.6185),
    ("tariff_6", "MID", "tariff_21"): (-0.00792, 1, 0.9085),
    ("tariff_6", "MID", "tariff_4"): (0.00821, 13, 0.9400),
    ("tariff_6", "MID", "tariff_8"): (0.02705, 9, 1.2745),
    ("tariff_6", "MID", "tariff_9"): (-0.00382, 3, 1.4966),
    ("tariff_7", "HIGH", "tariff_1"): (-0.05383, 6, 0.3612),
    ("tariff_7", "HIGH", "tariff_10"): (-0.03371, 26, 0.6324),
    ("tariff_7", "HIGH", "tariff_11"): (-0.00212, 8, 0.7347),
    ("tariff_7", "HIGH", "tariff_12"): (0.02378, 6, 1.4596),
    ("tariff_7", "HIGH", "tariff_13"): (-0.03696, 5, 0.3640),
    ("tariff_7", "HIGH", "tariff_4"): (-0.06102, 15, 0.7974),
    ("tariff_7", "HIGH", "tariff_8"): (-0.08717, 20, 0.5940),
    ("tariff_7", "LOW", "tariff_1"): (0.02207, 4, 1.8603),
    ("tariff_7", "LOW", "tariff_10"): (0.03571, 16, 1.5254),
    ("tariff_7", "LOW", "tariff_11"): (-0.03776, 3, 0.2012),
    ("tariff_7", "LOW", "tariff_12"): (0.12380, 4, 1.6672),
    ("tariff_7", "LOW", "tariff_13"): (0.12923, 9, 1.8390),
    ("tariff_7", "LOW", "tariff_21"): (-0.01429, 1, 0.9085),
    ("tariff_7", "LOW", "tariff_4"): (0.08744, 10, 1.9477),
    ("tariff_7", "LOW", "tariff_8"): (0.01429, 3, 2.3094),
    ("tariff_7", "LOW", "tariff_9"): (0.46764, 20, 1.7015),
    ("tariff_7", "MID", "tariff_1"): (-0.05576, 14, 0.6839),
    ("tariff_7", "MID", "tariff_10"): (-0.02338, 31, 1.2661),
    ("tariff_7", "MID", "tariff_11"): (0.00665, 6, 1.2127),
    ("tariff_7", "MID", "tariff_12"): (0.01315, 2, 0.1207),
    ("tariff_7", "MID", "tariff_13"): (-0.04627, 18, 0.6667),
    ("tariff_7", "MID", "tariff_21"): (-0.01742, 2, 0.1334),
    ("tariff_7", "MID", "tariff_4"): (0.08449, 15, 1.3411),
    ("tariff_7", "MID", "tariff_8"): (-0.01183, 13, 1.1104),
    ("tariff_7", "MID", "tariff_9"): (-0.01280, 3, 0.6430),
    ("tariff_8", "HIGH", "tariff_1"): (-0.03367, 49, 0.4414),
    ("tariff_8", "HIGH", "tariff_10"): (0.00643, 283, 0.4909),
    ("tariff_8", "HIGH", "tariff_11"): (0.00394, 133, 0.5654),
    ("tariff_8", "HIGH", "tariff_12"): (0.00358, 113, 0.6216),
    ("tariff_8", "HIGH", "tariff_13"): (-0.05753, 113, 0.2647),
    ("tariff_8", "HIGH", "tariff_21"): (-0.00023, 1, 0.9085),
    ("tariff_8", "HIGH", "tariff_4"): (-0.04132, 204, 0.4094),
    ("tariff_8", "LOW", "tariff_1"): (0.00769, 7, 1.0844),
    ("tariff_8", "LOW", "tariff_10"): (0.10218, 30, 1.7436),
    ("tariff_8", "LOW", "tariff_11"): (0.08269, 16, 1.7843),
    ("tariff_8", "LOW", "tariff_12"): (-0.01800, 10, 1.3274),
    ("tariff_8", "LOW", "tariff_13"): (0.14761, 15, 1.6195),
    ("tariff_8", "LOW", "tariff_21"): (0.05439, 6, 2.0073),
    ("tariff_8", "LOW", "tariff_4"): (0.12050, 20, 1.9231),
    ("tariff_8", "LOW", "tariff_9"): (0.48303, 26, 1.3880),
    ("tariff_8", "MID", "tariff_1"): (-0.04056, 63, 0.6505),
    ("tariff_8", "MID", "tariff_10"): (0.09673, 182, 1.1714),
    ("tariff_8", "MID", "tariff_11"): (0.03129, 99, 1.1980),
    ("tariff_8", "MID", "tariff_12"): (0.00631, 55, 1.2372),
    ("tariff_8", "MID", "tariff_13"): (-0.02530, 164, 0.5230),
    ("tariff_8", "MID", "tariff_21"): (-0.00431, 10, 1.0246),
    ("tariff_8", "MID", "tariff_4"): (-0.02448, 178, 0.7885),
    ("tariff_8", "MID", "tariff_9"): (0.00669, 19, 1.2578),
    ("tariff_9", "HIGH", "tariff_1"): (-0.09725, 4, 0.2534),
    ("tariff_9", "HIGH", "tariff_10"): (-0.08015, 4, 0.3630),
    ("tariff_9", "HIGH", "tariff_11"): (0.06143, 7, 0.8918),
    ("tariff_9", "HIGH", "tariff_13"): (-0.04460, 2, 0.2837),
    ("tariff_9", "HIGH", "tariff_4"): (-0.00829, 2, 0.0464),
    ("tariff_9", "HIGH", "tariff_8"): (-0.04303, 12, 0.4705),
    ("tariff_9", "LOW", "tariff_1"): (0.01515, 44, 1.0892),
    ("tariff_9", "LOW", "tariff_10"): (0.01826, 3, 2.0538),
    ("tariff_9", "LOW", "tariff_11"): (0.04492, 4, 1.9907),
    ("tariff_9", "LOW", "tariff_12"): (-0.01655, 2, 0.0096),
    ("tariff_9", "LOW", "tariff_13"): (0.23804, 18, 1.4318),
    ("tariff_9", "LOW", "tariff_4"): (0.09614, 12, 1.8889),
    ("tariff_9", "LOW", "tariff_8"): (0.38004, 37, 1.6890),
    ("tariff_9", "MID", "tariff_1"): (-0.06865, 44, 0.6958),
    ("tariff_9", "MID", "tariff_10"): (0.00631, 11, 0.8441),
    ("tariff_9", "MID", "tariff_11"): (0.00526, 11, 1.1326),
    ("tariff_9", "MID", "tariff_12"): (-0.01320, 10, 0.7287),
    ("tariff_9", "MID", "tariff_13"): (0.03692, 49, 0.7735),
    ("tariff_9", "MID", "tariff_4"): (0.07182, 37, 1.2863),
    ("tariff_9", "MID", "tariff_8"): (0.12380, 71, 1.2550),
}
# === PRIOR END ===


@dataclass
class Candidate:
    segment: str
    target: str
    tariffs: tuple
    size: int                   # абонентов в кампании с учётом обрезки 5 000 по ID
    arpu_sum: float             # сумма predicted_arpu этих абонентов
    mu: float                   # оценка базового эффекта q (без множителя канала)
    sd: float
    pilots: list = field(default_factory=list)

    @property
    def key(self):
        return f"{self.segment}:{self.target}:{';'.join(self.tariffs)}"

    def lcb(self, k=RISK_K):
        return self.mu - k * self.sd

    def ucb(self, k=1.0):
        return self.mu + k * self.sd


class Agent:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.trace = []

    # ------------------------------------------------------------------ public
    def act(self, env):
        self.trace = []
        self._t0 = time.monotonic()
        candidates = []
        try:
            profile = env.customer_profile
            candidates = self._make_candidates(profile, env)
            self._explore(env, candidates)
            plan = self._build_plan(candidates, env)
            plan = self._validate(plan, env)
        except Exception as exc:  # агент не должен падать: пилоты уже в зачёте
            self._log("error", error=f"{type(exc).__name__}: {exc}")
            plan = self._fallback_plan(candidates, env)
        self._log("done", campaigns=len(plan), seconds=round(time.monotonic() - self._t0, 1))
        return plan

    # ------------------------------------------------------------ 1-2: prior
    def _cell_prior(self, tariff, segment, target):
        row = PRIOR.get((tariff, segment, target))
        if row is None:
            return None
        q, n_obs, pct_std = row
        sd = math.sqrt(PRIOR_MODEL_SD ** 2 + (pct_std ** 2) / max(n_obs, 1))
        return PRIOR_SHRINK * q, sd

    def _make_candidates(self, profile, env):
        known = profile.dropna(subset=["current_tariff", "arpu_segment"])
        cells = (known.groupby(["current_tariff", "arpu_segment"])["predicted_arpu"]
                 .agg(n="size", arpu_sum="sum").reset_index())
        targets = sorted(env.tariffs["tariff_plan_code"])
        candidates = []
        for segment in SEGMENTS:
            seg_cells = cells[cells["arpu_segment"] == segment]
            for target in targets:
                parts = []
                for cell in seg_cells.itertuples():
                    if cell.current_tariff == target:
                        continue
                    prior = self._cell_prior(cell.current_tariff, segment, target)
                    if prior is not None and prior[0] > 0:
                        parts.append((cell.current_tariff, prior[0], prior[1], cell.arpu_sum))
                if not parts:
                    continue
                parts.sort(key=lambda p: -p[1])
                tariffs = tuple(sorted(p[0] for p in parts))
                size, arpu_sum = self._audience(profile, segment, tariffs)
                if size < MIN_CANDIDATE_SIZE:
                    continue
                w = sum(p[3] for p in parts)
                mu = sum(p[1] * p[3] for p in parts) / w
                sd = sum(p[2] * p[3] for p in parts) / w       # консервативно: без усреднения ошибок
                candidates.append(Candidate(segment, target, tariffs, size, arpu_sum, mu, sd))
        candidates.sort(key=lambda c: (-c.mu * c.arpu_sum, c.key))
        self._log("candidates", count=len(candidates),
                  top=[(c.key, round(c.mu, 4), c.size) for c in candidates[:5]])
        return candidates

    @staticmethod
    def _audience(profile, segment, tariffs):
        seg = profile[(profile["arpu_segment"] == segment) & (profile["current_tariff"].isin(tariffs))]
        seg = seg.sort_values("ID_NUMBER").head(MAX_PER_CAMPAIGN)
        return len(seg), float(seg["predicted_arpu"].sum())

    # ----------------------------------------------------------- 3-4: pilots
    def _explore(self, env, candidates):
        channel = env.channels[PILOT_CHANNEL]
        m, cost = channel["conversion_multiplier"], channel["cost_per_contact"]
        while env.pilots_left > 0 and time.monotonic() - self._t0 < TIME_BUDGET_S:
            pool = [c for c in candidates
                    if len(c.pilots) < MAX_PILOTS_PER_CANDIDATE and c.ucb() > 0]
            if not pool:
                self._log("explore_stop", reason="нет кандидатов с положительной оптимистичной оценкой")
                break
            c = max(pool, key=lambda c: (c.ucb() * c.arpu_sum, c.key))
            n = min(PILOT_N_LARGE if c.size >= SMALL_SEGMENT else PILOT_N_SMALL, c.size)
            if n * cost > env.remaining_budget or n > env.remaining_contacts:
                self._log("explore_stop", reason="не хватает бюджета или охвата на пилот")
                break
            try:
                res = env.run_pilot(target_tariff=c.target, channel=PILOT_CHANNEL, n_customers=n,
                                    filter_arpu_segment=c.segment,
                                    filter_current_tariff=";".join(c.tariffs))
            except (RuntimeError, ValueError) as exc:
                self._log("pilot_error", candidate=c.key, error=str(exc))
                break
            self._update(c, res["observed_lift_ratio"], m, res["n_customers"])
            self._log("pilot", candidate=c.key, channel=PILOT_CHANNEL, n=res["n_customers"],
                      observed=round(res["observed_lift_ratio"], 4),
                      mu=round(c.mu, 4), sd=round(c.sd, 4), lcb=round(c.lcb(), 4))

    @staticmethod
    def _update(c, observed_ratio, multiplier, n):
        q_hat = observed_ratio / multiplier
        obs_sd = PILOT_NOISE_SD / (multiplier * math.sqrt(n))
        prec = 1 / c.sd ** 2 + 1 / obs_sd ** 2
        c.mu = (c.mu / c.sd ** 2 + q_hat / obs_sd ** 2) / prec
        c.sd = math.sqrt(1 / prec)
        c.pilots.append({"observed": observed_ratio, "n": n})

    # ------------------------------------------------------------- 5-6: plan
    def _build_plan(self, candidates, env, require_pilot=True):
        viable = [c for c in candidates if c.lcb() > 0 and (c.pilots or not require_pilot)]
        viable.sort(key=lambda c: (-c.lcb() * c.arpu_sum, c.key))
        covered, plan = set(), []
        budget, contacts = env.remaining_budget, env.remaining_contacts
        for c in viable:
            tariffs = tuple(t for t in c.tariffs if (t, c.segment) not in covered)
            if not tariffs:
                continue
            size, _ = self._audience(env.customer_profile, c.segment, tariffs)
            size = min(size, contacts)
            if size <= 0:
                break
            channel = "sms" if size * env.channels["sms"]["cost_per_contact"] <= budget else "push"
            budget -= size * env.channels[channel]["cost_per_contact"]
            contacts -= size
            covered.update((t, c.segment) for t in tariffs)
            plan.append({"campaign_name": f"{c.segment}_{c.target}_{len(plan) + 1}",
                         "filter_arpu_segment": c.segment,
                         "filter_current_tariff": ";".join(tariffs),
                         "target_tariff": c.target, "channel": channel})
            self._log("plan_add", candidate=c.key, channel=channel, contacts=size,
                      mu=round(c.mu, 4), lcb=round(c.lcb(), 4), pilots=len(c.pilots))
            if len(plan) == MAX_CAMPAIGNS:
                break
        return plan

    # ------------------------------------------------------ 7: checks + fallback
    def _validate(self, plan, env):
        tariffs = set(env.tariffs["tariff_plan_code"])
        ok = []
        for camp in plan:
            listed = camp.get("filter_current_tariff", "").split(";")
            if (camp["target_tariff"] in tariffs and camp["channel"] in env.channels
                    and camp.get("filter_arpu_segment") in SEGMENTS and all(t in tariffs for t in listed)):
                ok.append(camp)
            else:
                self._log("plan_drop", campaign=camp, reason="невалидная кампания")
        return ok[:MAX_CAMPAIGNS]

    def _fallback_plan(self, candidates, env):
        try:
            return self._validate(self._build_plan(candidates, env), env)
        except Exception as exc:
            self._log("fallback_error", error=f"{type(exc).__name__}: {exc}")
            return []

    # ------------------------------------------------------------------ trace
    def _log(self, kind, **data):
        entry = {"t": round(time.monotonic() - self._t0, 2), "kind": kind, **data}
        self.trace.append(entry)
        if self.verbose:
            print(f"[agent] {kind}: " + ", ".join(f"{k}={v}" for k, v in data.items()))
