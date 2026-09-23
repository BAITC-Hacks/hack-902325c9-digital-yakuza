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
    ("tariff_1", "HIGH", "tariff_10"): (0.01460, 9, 0.5179),
    ("tariff_1", "HIGH", "tariff_11"): (0.00574, 4, 0.4219),
    ("tariff_1", "HIGH", "tariff_12"): (0.00802, 2, 0.4203),
    ("tariff_1", "HIGH", "tariff_13"): (-0.03760, 14, 0.3484),
    ("tariff_1", "HIGH", "tariff_4"): (-0.00242, 10, 0.5834),
    ("tariff_1", "HIGH", "tariff_8"): (-0.07162, 32, 0.4631),
    ("tariff_1", "LOW", "tariff_10"): (0.11325, 14, 1.5719),
    ("tariff_1", "LOW", "tariff_11"): (0.01460, 2, 1.7808),
    ("tariff_1", "LOW", "tariff_13"): (0.18084, 22, 1.1651),
    ("tariff_1", "LOW", "tariff_21"): (0.03329, 10, 1.6408),
    ("tariff_1", "LOW", "tariff_4"): (0.27646, 35, 1.5843),
    ("tariff_1", "LOW", "tariff_8"): (0.45557, 71, 1.6883),
    ("tariff_1", "LOW", "tariff_9"): (0.66054, 81, 1.4853),
    ("tariff_1", "MID", "tariff_10"): (0.06872, 26, 1.1757),
    ("tariff_1", "MID", "tariff_11"): (0.03855, 12, 1.3661),
    ("tariff_1", "MID", "tariff_12"): (0.00819, 2, 1.0849),
    ("tariff_1", "MID", "tariff_13"): (0.07371, 66, 0.7784),
    ("tariff_1", "MID", "tariff_21"): (0.01083, 5, 1.4309),
    ("tariff_1", "MID", "tariff_4"): (0.14286, 51, 1.1738),
    ("tariff_1", "MID", "tariff_8"): (0.30362, 125, 1.2562),
    ("tariff_1", "MID", "tariff_9"): (0.08702, 27, 1.1398),
    ("tariff_10", "HIGH", "tariff_1"): (-0.01168, 17, 0.4312),
    ("tariff_10", "HIGH", "tariff_11"): (-0.01170, 134, 0.4626),
    ("tariff_10", "HIGH", "tariff_12"): (-0.00780, 91, 0.4552),
    ("tariff_10", "HIGH", "tariff_13"): (-0.01555, 33, 0.3705),
    ("tariff_10", "HIGH", "tariff_4"): (-0.03690, 77, 0.4112),
    ("tariff_10", "HIGH", "tariff_8"): (-0.06438, 324, 0.3825),
    ("tariff_10", "LOW", "tariff_1"): (0.01144, 1, 1.6134),
    ("tariff_10", "LOW", "tariff_11"): (0.06634, 11, 1.5822),
    ("tariff_10", "LOW", "tariff_12"): (0.03493, 4, 1.3634),
    ("tariff_10", "LOW", "tariff_13"): (0.04969, 2, 1.5840),
    ("tariff_10", "LOW", "tariff_21"): (0.03378, 2, 1.8715),
    ("tariff_10", "LOW", "tariff_4"): (0.11624, 13, 1.4589),
    ("tariff_10", "LOW", "tariff_8"): (0.11116, 19, 1.6960),
    ("tariff_10", "LOW", "tariff_9"): (0.21460, 8, 1.6926),
    ("tariff_10", "MID", "tariff_1"): (-0.01599, 12, 0.5601),
    ("tariff_10", "MID", "tariff_11"): (0.09426, 86, 0.9791),
    ("tariff_10", "MID", "tariff_12"): (0.03180, 30, 0.8973),
    ("tariff_10", "MID", "tariff_13"): (-0.01216, 31, 0.7367),
    ("tariff_10", "MID", "tariff_21"): (0.00036, 1, 0.9223),
    ("tariff_10", "MID", "tariff_4"): (0.00227, 60, 0.8480),
    ("tariff_10", "MID", "tariff_8"): (0.07944, 138, 0.7802),
    ("tariff_10", "MID", "tariff_9"): (0.00051, 3, 0.8909),
    ("tariff_11", "HIGH", "tariff_1"): (-0.00361, 5, 0.3966),
    ("tariff_11", "HIGH", "tariff_10"): (-0.01391, 110, 0.4274),
    ("tariff_11", "HIGH", "tariff_12"): (0.01371, 107, 0.5243),
    ("tariff_11", "HIGH", "tariff_13"): (-0.01686, 20, 0.3570),
    ("tariff_11", "HIGH", "tariff_4"): (-0.02313, 42, 0.5763),
    ("tariff_11", "HIGH", "tariff_8"): (-0.06997, 213, 0.4241),
    ("tariff_11", "LOW", "tariff_10"): (-0.01090, 14, 1.2487),
    ("tariff_11", "LOW", "tariff_12"): (0.09967, 20, 1.5244),
    ("tariff_11", "LOW", "tariff_13"): (0.03127, 4, 1.7386),
    ("tariff_11", "LOW", "tariff_21"): (0.02104, 1, 1.6134),
    ("tariff_11", "LOW", "tariff_4"): (0.04981, 10, 1.6741),
    ("tariff_11", "LOW", "tariff_8"): (0.25269, 16, 1.7608),
    ("tariff_11", "LOW", "tariff_9"): (0.10139, 4, 1.6942),
    ("tariff_11", "MID", "tariff_1"): (-0.00893, 6, 0.7667),
    ("tariff_11", "MID", "tariff_10"): (0.02246, 29, 1.0488),
    ("tariff_11", "MID", "tariff_12"): (0.04072, 62, 0.9838),
    ("tariff_11", "MID", "tariff_13"): (-0.00593, 7, 0.7085),
    ("tariff_11", "MID", "tariff_4"): (0.02218, 42, 0.9975),
    ("tariff_11", "MID", "tariff_8"): (0.08260, 103, 0.8310),
    ("tariff_11", "MID", "tariff_9"): (0.00035, 3, 0.9436),
    ("tariff_12", "HIGH", "tariff_1"): (-0.00724, 5, 0.4245),
    ("tariff_12", "HIGH", "tariff_10"): (-0.00931, 85, 0.4682),
    ("tariff_12", "HIGH", "tariff_11"): (0.00382, 82, 0.5248),
    ("tariff_12", "HIGH", "tariff_13"): (-0.00933, 9, 0.4182),
    ("tariff_12", "HIGH", "tariff_4"): (-0.01251, 15, 0.3539),
    ("tariff_12", "HIGH", "tariff_8"): (-0.04968, 168, 0.4055),
    ("tariff_12", "HIGH", "tariff_9"): (-0.00084, 1, 0.4589),
    ("tariff_12", "LOW", "tariff_10"): (-0.01780, 8, 1.1547),
    ("tariff_12", "LOW", "tariff_11"): (0.14503, 6, 1.8566),
    ("tariff_12", "LOW", "tariff_13"): (0.04837, 3, 1.8091),
    ("tariff_12", "LOW", "tariff_4"): (-0.05476, 10, 1.1084),
    ("tariff_12", "LOW", "tariff_8"): (0.20767, 6, 1.7602),
    ("tariff_12", "LOW", "tariff_9"): (0.02891, 1, 1.6134),
    ("tariff_12", "MID", "tariff_1"): (-0.01027, 2, 0.8719),
    ("tariff_12", "MID", "tariff_10"): (0.01637, 22, 0.9757),
    ("tariff_12", "MID", "tariff_11"): (-0.01975, 22, 0.8358),
    ("tariff_12", "MID", "tariff_13"): (-0.01904, 9, 0.7472),
    ("tariff_12", "MID", "tariff_4"): (-0.00965, 10, 1.0293),
    ("tariff_12", "MID", "tariff_8"): (0.20676, 78, 0.9375),
    ("tariff_12", "MID", "tariff_9"): (-0.00208, 1, 0.9223),
    ("tariff_13", "HIGH", "tariff_1"): (-0.01748, 7, 0.6752),
    ("tariff_13", "HIGH", "tariff_10"): (0.00344, 9, 0.4817),
    ("tariff_13", "HIGH", "tariff_11"): (-0.00607, 6, 0.5781),
    ("tariff_13", "HIGH", "tariff_12"): (0.00108, 9, 0.4503),
    ("tariff_13", "HIGH", "tariff_4"): (-0.02751, 22, 0.3582),
    ("tariff_13", "HIGH", "tariff_8"): (0.01195, 33, 0.6467),
    ("tariff_13", "LOW", "tariff_1"): (0.06822, 3, 1.7124),
    ("tariff_13", "LOW", "tariff_10"): (0.09591, 2, 1.4728),
    ("tariff_13", "LOW", "tariff_11"): (0.03795, 1, 1.6134),
    ("tariff_13", "LOW", "tariff_21"): (0.03632, 2, 1.5020),
    ("tariff_13", "LOW", "tariff_4"): (0.08572, 7, 1.5175),
    ("tariff_13", "LOW", "tariff_8"): (0.50403, 20, 1.6118),
    ("tariff_13", "LOW", "tariff_9"): (0.25959, 8, 1.8626),
    ("tariff_13", "MID", "tariff_1"): (-0.03457, 41, 0.6773),
    ("tariff_13", "MID", "tariff_10"): (0.03569, 28, 1.0017),
    ("tariff_13", "MID", "tariff_11"): (0.01181, 9, 0.7763),
    ("tariff_13", "MID", "tariff_12"): (0.00494, 3, 0.8840),
    ("tariff_13", "MID", "tariff_21"): (0.00235, 2, 0.8419),
    ("tariff_13", "MID", "tariff_4"): (0.03974, 73, 0.7005),
    ("tariff_13", "MID", "tariff_8"): (0.25301, 234, 0.9712),
    ("tariff_13", "MID", "tariff_9"): (0.02549, 28, 0.8009),
    ("tariff_14", "HIGH", "tariff_1"): (-0.02070, 4, 0.4584),
    ("tariff_14", "HIGH", "tariff_10"): (-0.03383, 11, 0.5237),
    ("tariff_14", "HIGH", "tariff_11"): (-0.00544, 4, 0.5060),
    ("tariff_14", "HIGH", "tariff_12"): (-0.00233, 2, 0.5116),
    ("tariff_14", "HIGH", "tariff_13"): (-0.01794, 4, 0.4198),
    ("tariff_14", "HIGH", "tariff_4"): (-0.03093, 9, 0.3800),
    ("tariff_14", "HIGH", "tariff_8"): (-0.04940, 35, 0.4136),
    ("tariff_14", "LOW", "tariff_1"): (0.05421, 2, 1.4728),
    ("tariff_14", "LOW", "tariff_10"): (0.05289, 4, 1.7506),
    ("tariff_14", "LOW", "tariff_11"): (0.01957, 1, 1.6134),
    ("tariff_14", "LOW", "tariff_12"): (0.05599, 4, 1.7604),
    ("tariff_14", "LOW", "tariff_13"): (0.05371, 3, 1.7421),
    ("tariff_14", "LOW", "tariff_21"): (0.05571, 3, 1.6920),
    ("tariff_14", "LOW", "tariff_4"): (0.12207, 6, 1.7600),
    ("tariff_14", "LOW", "tariff_8"): (0.23808, 9, 1.8518),
    ("tariff_14", "LOW", "tariff_9"): (0.42130, 21, 1.6344),
    ("tariff_14", "MID", "tariff_1"): (-0.02332, 8, 0.6960),
    ("tariff_14", "MID", "tariff_10"): (0.01782, 17, 1.0708),
    ("tariff_14", "MID", "tariff_11"): (0.00636, 3, 0.9227),
    ("tariff_14", "MID", "tariff_12"): (0.00676, 6, 0.6523),
    ("tariff_14", "MID", "tariff_13"): (0.00131, 9, 1.0973),
    ("tariff_14", "MID", "tariff_21"): (0.00069, 1, 0.9223),
    ("tariff_14", "MID", "tariff_4"): (-0.00047, 25, 0.6903),
    ("tariff_14", "MID", "tariff_8"): (-0.00934, 42, 0.9419),
    ("tariff_14", "MID", "tariff_9"): (0.01719, 7, 1.1240),
    ("tariff_15", "HIGH", "tariff_1"): (-0.00625, 3, 0.6317),
    ("tariff_15", "HIGH", "tariff_10"): (0.00614, 9, 0.4257),
    ("tariff_15", "HIGH", "tariff_11"): (-0.00049, 5, 0.5200),
    ("tariff_15", "HIGH", "tariff_12"): (0.00844, 3, 0.4926),
    ("tariff_15", "HIGH", "tariff_13"): (-0.02406, 11, 0.3386),
    ("tariff_15", "HIGH", "tariff_4"): (-0.01631, 16, 0.3219),
    ("tariff_15", "HIGH", "tariff_8"): (-0.03271, 30, 0.4127),
    ("tariff_15", "LOW", "tariff_1"): (0.04264, 4, 1.6142),
    ("tariff_15", "LOW", "tariff_10"): (0.08790, 4, 1.7576),
    ("tariff_15", "LOW", "tariff_11"): (0.01700, 1, 1.6134),
    ("tariff_15", "LOW", "tariff_13"): (0.24043, 14, 1.5291),
    ("tariff_15", "LOW", "tariff_21"): (0.10039, 8, 1.6965),
    ("tariff_15", "LOW", "tariff_4"): (0.09503, 7, 1.7503),
    ("tariff_15", "LOW", "tariff_8"): (0.36636, 21, 1.6904),
    ("tariff_15", "LOW", "tariff_9"): (0.33350, 19, 1.6969),
    ("tariff_15", "MID", "tariff_1"): (-0.01383, 54, 0.5595),
    ("tariff_15", "MID", "tariff_10"): (0.06700, 180, 0.7531),
    ("tariff_15", "MID", "tariff_11"): (0.00958, 23, 0.9253),
    ("tariff_15", "MID", "tariff_12"): (0.00371, 4, 0.9558),
    ("tariff_15", "MID", "tariff_13"): (0.00901, 348, 0.5259),
    ("tariff_15", "MID", "tariff_21"): (0.00219, 8, 0.8748),
    ("tariff_15", "MID", "tariff_4"): (0.04225, 133, 0.7907),
    ("tariff_15", "MID", "tariff_8"): (0.15776, 429, 0.7614),
    ("tariff_15", "MID", "tariff_9"): (0.03548, 133, 0.6073),
    ("tariff_16", "HIGH", "tariff_1"): (-0.01282, 1, 0.4589),
    ("tariff_16", "HIGH", "tariff_11"): (0.00124, 3, 0.4841),
    ("tariff_16", "HIGH", "tariff_13"): (-0.02821, 4, 0.6398),
    ("tariff_16", "HIGH", "tariff_4"): (-0.01350, 3, 0.4809),
    ("tariff_16", "HIGH", "tariff_8"): (-0.01588, 4, 0.3824),
    ("tariff_16", "LOW", "tariff_1"): (0.04781, 27, 1.2894),
    ("tariff_16", "LOW", "tariff_10"): (0.01535, 2, 1.8715),
    ("tariff_16", "LOW", "tariff_11"): (0.03196, 4, 1.6237),
    ("tariff_16", "LOW", "tariff_13"): (0.07929, 9, 1.2350),
    ("tariff_16", "LOW", "tariff_21"): (0.13640, 29, 1.6423),
    ("tariff_16", "LOW", "tariff_4"): (0.09440, 13, 1.6739),
    ("tariff_16", "LOW", "tariff_8"): (0.20971, 25, 1.4636),
    ("tariff_16", "LOW", "tariff_9"): (0.59872, 108, 1.7111),
    ("tariff_16", "MID", "tariff_1"): (-0.07244, 48, 0.5676),
    ("tariff_16", "MID", "tariff_10"): (0.01594, 4, 1.0363),
    ("tariff_16", "MID", "tariff_11"): (0.03155, 9, 1.2367),
    ("tariff_16", "MID", "tariff_12"): (0.00456, 1, 0.9223),
    ("tariff_16", "MID", "tariff_13"): (0.04121, 43, 0.7984),
    ("tariff_16", "MID", "tariff_21"): (0.01654, 7, 1.4113),
    ("tariff_16", "MID", "tariff_4"): (0.03030, 15, 0.9325),
    ("tariff_16", "MID", "tariff_8"): (0.15539, 44, 1.1247),
    ("tariff_16", "MID", "tariff_9"): (0.06248, 22, 1.2672),
    ("tariff_17", "HIGH", "tariff_1"): (-0.00539, 3, 0.4805),
    ("tariff_17", "HIGH", "tariff_10"): (0.00559, 14, 0.3612),
    ("tariff_17", "HIGH", "tariff_11"): (0.00675, 11, 0.4662),
    ("tariff_17", "HIGH", "tariff_12"): (0.00077, 6, 0.4755),
    ("tariff_17", "HIGH", "tariff_13"): (-0.02019, 16, 0.4248),
    ("tariff_17", "HIGH", "tariff_4"): (0.00132, 30, 0.6519),
    ("tariff_17", "HIGH", "tariff_8"): (-0.03213, 114, 0.3300),
    ("tariff_17", "LOW", "tariff_1"): (0.03042, 1, 1.6134),
    ("tariff_17", "LOW", "tariff_12"): (0.05959, 1, 1.6134),
    ("tariff_17", "LOW", "tariff_13"): (0.08768, 3, 1.7939),
    ("tariff_17", "LOW", "tariff_4"): (0.12531, 4, 1.6007),
    ("tariff_17", "LOW", "tariff_8"): (0.45889, 15, 1.7213),
    ("tariff_17", "LOW", "tariff_9"): (0.23909, 4, 1.6317),
    ("tariff_17", "MID", "tariff_1"): (-0.01713, 23, 0.9156),
    ("tariff_17", "MID", "tariff_10"): (0.02793, 16, 0.8650),
    ("tariff_17", "MID", "tariff_11"): (0.02617, 23, 0.8028),
    ("tariff_17", "MID", "tariff_12"): (0.01084, 4, 0.8654),
    ("tariff_17", "MID", "tariff_13"): (-0.00007, 35, 0.6962),
    ("tariff_17", "MID", "tariff_4"): (0.03174, 45, 0.5362),
    ("tariff_17", "MID", "tariff_8"): (0.21436, 127, 0.8375),
    ("tariff_17", "MID", "tariff_9"): (0.00922, 5, 1.2237),
    ("tariff_18", "HIGH", "tariff_1"): (-0.01053, 8, 0.4627),
    ("tariff_18", "HIGH", "tariff_10"): (-0.00280, 67, 0.4137),
    ("tariff_18", "HIGH", "tariff_11"): (0.00899, 22, 0.5788),
    ("tariff_18", "HIGH", "tariff_12"): (-0.00103, 26, 0.3305),
    ("tariff_18", "HIGH", "tariff_13"): (-0.02228, 23, 0.4359),
    ("tariff_18", "HIGH", "tariff_4"): (-0.02370, 49, 0.3704),
    ("tariff_18", "HIGH", "tariff_8"): (-0.03600, 127, 0.3537),
    ("tariff_18", "LOW", "tariff_1"): (0.02481, 4, 1.3039),
    ("tariff_18", "LOW", "tariff_10"): (0.13126, 9, 1.8156),
    ("tariff_18", "LOW", "tariff_11"): (0.06426, 2, 1.4728),
    ("tariff_18", "LOW", "tariff_12"): (0.03526, 1, 1.6134),
    ("tariff_18", "LOW", "tariff_13"): (0.12101, 6, 1.7879),
    ("tariff_18", "LOW", "tariff_4"): (0.08340, 3, 1.7225),
    ("tariff_18", "LOW", "tariff_8"): (0.44439, 14, 1.6539),
    ("tariff_18", "LOW", "tariff_9"): (0.35949, 13, 1.6640),
    ("tariff_18", "MID", "tariff_1"): (-0.01164, 5, 0.6914),
    ("tariff_18", "MID", "tariff_10"): (0.09052, 27, 1.0983),
    ("tariff_18", "MID", "tariff_11"): (0.02204, 5, 1.0677),
    ("tariff_18", "MID", "tariff_12"): (0.03488, 6, 1.1718),
    ("tariff_18", "MID", "tariff_13"): (0.02147, 25, 0.8217),
    ("tariff_18", "MID", "tariff_4"): (0.03434, 15, 1.2144),
    ("tariff_18", "MID", "tariff_8"): (0.17450, 50, 0.9242),
    ("tariff_18", "MID", "tariff_9"): (0.00587, 3, 1.3116),
    ("tariff_19", "HIGH", "tariff_1"): (-0.01594, 3, 0.4491),
    ("tariff_19", "HIGH", "tariff_10"): (-0.00489, 2, 0.4223),
    ("tariff_19", "HIGH", "tariff_11"): (-0.00406, 5, 0.7003),
    ("tariff_19", "HIGH", "tariff_12"): (-0.00596, 6, 0.4244),
    ("tariff_19", "HIGH", "tariff_13"): (-0.02395, 5, 0.3744),
    ("tariff_19", "HIGH", "tariff_4"): (-0.01815, 5, 0.4668),
    ("tariff_19", "HIGH", "tariff_8"): (-0.14101, 35, 0.3767),
    ("tariff_19", "LOW", "tariff_1"): (-0.00036, 8, 1.0965),
    ("tariff_19", "LOW", "tariff_10"): (0.06784, 3, 1.7383),
    ("tariff_19", "LOW", "tariff_11"): (0.04195, 3, 1.5422),
    ("tariff_19", "LOW", "tariff_13"): (0.21175, 8, 1.5984),
    ("tariff_19", "LOW", "tariff_21"): (0.03090, 1, 1.6134),
    ("tariff_19", "LOW", "tariff_4"): (0.08333, 3, 1.7913),
    ("tariff_19", "LOW", "tariff_8"): (0.12510, 11, 1.6963),
    ("tariff_19", "LOW", "tariff_9"): (0.24930, 12, 1.8772),
    ("tariff_19", "MID", "tariff_1"): (-0.01117, 3, 0.8787),
    ("tariff_19", "MID", "tariff_10"): (0.04287, 13, 0.9126),
    ("tariff_19", "MID", "tariff_11"): (0.02008, 4, 1.3452),
    ("tariff_19", "MID", "tariff_13"): (0.01008, 12, 0.6970),
    ("tariff_19", "MID", "tariff_4"): (0.02850, 16, 1.0931),
    ("tariff_19", "MID", "tariff_8"): (0.14447, 21, 1.0792),
    ("tariff_19", "MID", "tariff_9"): (0.01412, 7, 1.0106),
    ("tariff_2", "HIGH", "tariff_1"): (-0.00854, 1, 0.4589),
    ("tariff_2", "HIGH", "tariff_10"): (-0.00054, 2, 0.4447),
    ("tariff_2", "HIGH", "tariff_11"): (0.00085, 1, 0.4589),
    ("tariff_2", "HIGH", "tariff_13"): (-0.01130, 3, 0.4679),
    ("tariff_2", "HIGH", "tariff_4"): (-0.00937, 5, 0.4292),
    ("tariff_2", "HIGH", "tariff_8"): (-0.00738, 16, 0.4532),
    ("tariff_2", "LOW", "tariff_1"): (0.06302, 12, 1.4899),
    ("tariff_2", "LOW", "tariff_10"): (0.11233, 6, 1.8518),
    ("tariff_2", "LOW", "tariff_13"): (0.32934, 15, 1.4402),
    ("tariff_2", "LOW", "tariff_21"): (0.15885, 13, 1.7734),
    ("tariff_2", "LOW", "tariff_4"): (0.05636, 4, 1.6409),
    ("tariff_2", "LOW", "tariff_8"): (0.19720, 17, 1.4813),
    ("tariff_2", "LOW", "tariff_9"): (0.19689, 14, 1.8039),
    ("tariff_2", "MID", "tariff_1"): (-0.01864, 20, 0.5714),
    ("tariff_2", "MID", "tariff_10"): (0.03753, 30, 0.7335),
    ("tariff_2", "MID", "tariff_11"): (0.00876, 4, 0.8846),
    ("tariff_2", "MID", "tariff_12"): (0.01364, 5, 1.0894),
    ("tariff_2", "MID", "tariff_13"): (0.02151, 58, 0.6738),
    ("tariff_2", "MID", "tariff_21"): (0.00277, 3, 0.8190),
    ("tariff_2", "MID", "tariff_4"): (0.03219, 23, 0.9196),
    ("tariff_2", "MID", "tariff_8"): (0.13907, 113, 0.9053),
    ("tariff_2", "MID", "tariff_9"): (0.02860, 42, 0.7094),
    ("tariff_20", "HIGH", "tariff_1"): (-0.03164, 11, 0.3840),
    ("tariff_20", "HIGH", "tariff_10"): (0.00079, 17, 0.4159),
    ("tariff_20", "HIGH", "tariff_11"): (-0.00072, 8, 0.4921),
    ("tariff_20", "HIGH", "tariff_12"): (-0.00101, 9, 0.5571),
    ("tariff_20", "HIGH", "tariff_13"): (-0.02442, 17, 0.3880),
    ("tariff_20", "HIGH", "tariff_4"): (-0.03465, 29, 0.3442),
    ("tariff_20", "HIGH", "tariff_8"): (-0.03020, 78, 0.2515),
    ("tariff_20", "LOW", "tariff_1"): (0.09681, 4, 1.5310),
    ("tariff_20", "LOW", "tariff_10"): (0.12471, 2, 1.6054),
    ("tariff_20", "LOW", "tariff_11"): (0.04158, 1, 1.6134),
    ("tariff_20", "LOW", "tariff_12"): (0.04685, 1, 1.6134),
    ("tariff_20", "LOW", "tariff_13"): (0.28396, 5, 1.3708),
    ("tariff_20", "LOW", "tariff_4"): (0.04857, 2, 1.4728),
    ("tariff_20", "LOW", "tariff_8"): (0.24376, 4, 1.6578),
    ("tariff_20", "LOW", "tariff_9"): (0.12977, 2, 1.8650),
    ("tariff_20", "MID", "tariff_1"): (-0.02288, 9, 1.0556),
    ("tariff_20", "MID", "tariff_10"): (0.02070, 5, 1.0059),
    ("tariff_20", "MID", "tariff_11"): (0.01408, 2, 0.8419),
    ("tariff_20", "MID", "tariff_12"): (0.02498, 3, 1.0794),
    ("tariff_20", "MID", "tariff_13"): (-0.00618, 8, 0.6222),
    ("tariff_20", "MID", "tariff_4"): (-0.00446, 8, 0.7176),
    ("tariff_20", "MID", "tariff_8"): (0.14897, 31, 0.8571),
    ("tariff_20", "MID", "tariff_9"): (0.00384, 1, 0.9223),
    ("tariff_3", "HIGH", "tariff_1"): (-0.00775, 3, 0.4634),
    ("tariff_3", "HIGH", "tariff_10"): (-0.02276, 19, 0.5293),
    ("tariff_3", "HIGH", "tariff_11"): (-0.00661, 11, 0.4856),
    ("tariff_3", "HIGH", "tariff_12"): (0.00698, 10, 0.5449),
    ("tariff_3", "HIGH", "tariff_13"): (-0.00870, 5, 0.3559),
    ("tariff_3", "HIGH", "tariff_4"): (-0.01149, 6, 0.4039),
    ("tariff_3", "HIGH", "tariff_8"): (-0.04974, 83, 0.5439),
    ("tariff_3", "LOW", "tariff_1"): (0.01644, 8, 1.5453),
    ("tariff_3", "LOW", "tariff_10"): (0.08581, 5, 1.6844),
    ("tariff_3", "LOW", "tariff_11"): (0.03645, 2, 1.4728),
    ("tariff_3", "LOW", "tariff_12"): (0.03563, 5, 1.6731),
    ("tariff_3", "LOW", "tariff_13"): (0.06109, 6, 1.6675),
    ("tariff_3", "LOW", "tariff_21"): (0.01249, 2, 1.4728),
    ("tariff_3", "LOW", "tariff_4"): (0.03465, 15, 1.4515),
    ("tariff_3", "LOW", "tariff_8"): (0.17745, 27, 1.7941),
    ("tariff_3", "LOW", "tariff_9"): (0.31943, 31, 1.8253),
    ("tariff_3", "MID", "tariff_1"): (-0.01661, 21, 0.9001),
    ("tariff_3", "MID", "tariff_10"): (0.03558, 31, 1.4056),
    ("tariff_3", "MID", "tariff_11"): (0.03476, 29, 1.2392),
    ("tariff_3", "MID", "tariff_12"): (0.01092, 8, 1.2398),
    ("tariff_3", "MID", "tariff_13"): (0.01074, 32, 0.8410),
    ("tariff_3", "MID", "tariff_21"): (0.00308, 8, 0.9671),
    ("tariff_3", "MID", "tariff_4"): (0.01530, 47, 1.1027),
    ("tariff_3", "MID", "tariff_8"): (0.30101, 241, 1.3301),
    ("tariff_3", "MID", "tariff_9"): (0.01067, 30, 0.8272),
    ("tariff_4", "HIGH", "tariff_1"): (-0.01885, 21, 0.3630),
    ("tariff_4", "HIGH", "tariff_10"): (-0.01072, 45, 0.4258),
    ("tariff_4", "HIGH", "tariff_11"): (-0.00752, 32, 0.5984),
    ("tariff_4", "HIGH", "tariff_12"): (0.00133, 19, 0.5347),
    ("tariff_4", "HIGH", "tariff_13"): (-0.02412, 31, 0.3475),
    ("tariff_4", "HIGH", "tariff_8"): (-0.09336, 340, 0.3773),
    ("tariff_4", "LOW", "tariff_1"): (0.03025, 3, 1.5015),
    ("tariff_4", "LOW", "tariff_10"): (0.18525, 17, 1.8289),
    ("tariff_4", "LOW", "tariff_11"): (-0.00897, 13, 0.9956),
    ("tariff_4", "LOW", "tariff_12"): (0.06533, 9, 1.4643),
    ("tariff_4", "LOW", "tariff_13"): (0.10894, 8, 1.5451),
    ("tariff_4", "LOW", "tariff_21"): (0.03336, 3, 1.7180),
    ("tariff_4", "LOW", "tariff_8"): (0.09709, 22, 1.5692),
    ("tariff_4", "LOW", "tariff_9"): (0.36284, 17, 1.4390),
    ("tariff_4", "MID", "tariff_1"): (-0.01983, 37, 0.6640),
    ("tariff_4", "MID", "tariff_10"): (0.01514, 58, 0.9493),
    ("tariff_4", "MID", "tariff_11"): (0.02385, 41, 1.1596),
    ("tariff_4", "MID", "tariff_12"): (0.00464, 13, 0.8645),
    ("tariff_4", "MID", "tariff_13"): (-0.00375, 100, 0.7012),
    ("tariff_4", "MID", "tariff_21"): (0.00111, 2, 0.8613),
    ("tariff_4", "MID", "tariff_8"): (0.23675, 408, 0.9275),
    ("tariff_4", "MID", "tariff_9"): (0.00890, 11, 1.1292),
    ("tariff_5", "HIGH", "tariff_1"): (-0.03919, 9, 0.3809),
    ("tariff_5", "HIGH", "tariff_10"): (-0.00199, 32, 0.6434),
    ("tariff_5", "HIGH", "tariff_11"): (0.00231, 18, 0.8219),
    ("tariff_5", "HIGH", "tariff_12"): (-0.02985, 10, 0.3963),
    ("tariff_5", "HIGH", "tariff_13"): (-0.01705, 6, 0.4025),
    ("tariff_5", "HIGH", "tariff_4"): (-0.04881, 13, 0.4217),
    ("tariff_5", "HIGH", "tariff_8"): (-0.01152, 13, 0.8456),
    ("tariff_5", "LOW", "tariff_1"): (0.00348, 10, 1.3724),
    ("tariff_5", "LOW", "tariff_10"): (0.07585, 32, 1.6271),
    ("tariff_5", "LOW", "tariff_11"): (0.01198, 9, 1.4459),
    ("tariff_5", "LOW", "tariff_12"): (0.00808, 8, 1.0631),
    ("tariff_5", "LOW", "tariff_13"): (0.30578, 36, 1.6479),
    ("tariff_5", "LOW", "tariff_21"): (0.00774, 1, 1.6134),
    ("tariff_5", "LOW", "tariff_4"): (0.13282, 20, 1.6993),
    ("tariff_5", "LOW", "tariff_8"): (0.07473, 6, 1.7868),
    ("tariff_5", "LOW", "tariff_9"): (0.06971, 11, 1.8127),
    ("tariff_5", "MID", "tariff_1"): (-0.02153, 18, 1.1969),
    ("tariff_5", "MID", "tariff_10"): (-0.00928, 36, 1.0211),
    ("tariff_5", "MID", "tariff_11"): (0.01517, 15, 1.1630),
    ("tariff_5", "MID", "tariff_12"): (0.01864, 17, 1.0991),
    ("tariff_5", "MID", "tariff_13"): (-0.00470, 43, 0.7684),
    ("tariff_5", "MID", "tariff_21"): (0.00248, 4, 1.0528),
    ("tariff_5", "MID", "tariff_4"): (0.05282, 32, 1.2018),
    ("tariff_5", "MID", "tariff_8"): (0.00792, 21, 1.0373),
    ("tariff_5", "MID", "tariff_9"): (0.00083, 2, 0.8668),
    ("tariff_6", "HIGH", "tariff_1"): (-0.02585, 4, 0.5277),
    ("tariff_6", "HIGH", "tariff_10"): (-0.04270, 9, 0.4735),
    ("tariff_6", "HIGH", "tariff_11"): (0.00366, 11, 0.4993),
    ("tariff_6", "HIGH", "tariff_12"): (-0.01069, 8, 0.9939),
    ("tariff_6", "HIGH", "tariff_13"): (-0.03092, 6, 0.5262),
    ("tariff_6", "HIGH", "tariff_4"): (-0.05389, 10, 0.4969),
    ("tariff_6", "HIGH", "tariff_8"): (-0.02333, 3, 0.4354),
    ("tariff_6", "HIGH", "tariff_9"): (-0.00261, 1, 0.4589),
    ("tariff_6", "LOW", "tariff_1"): (0.08123, 13, 1.4942),
    ("tariff_6", "LOW", "tariff_10"): (0.02988, 21, 1.4913),
    ("tariff_6", "LOW", "tariff_11"): (0.03205, 7, 1.5088),
    ("tariff_6", "LOW", "tariff_12"): (0.03481, 3, 1.8393),
    ("tariff_6", "LOW", "tariff_13"): (0.36867, 37, 1.7300),
    ("tariff_6", "LOW", "tariff_4"): (0.24398, 23, 1.6627),
    ("tariff_6", "LOW", "tariff_8"): (0.08654, 10, 1.7267),
    ("tariff_6", "LOW", "tariff_9"): (0.08686, 18, 1.6381),
    ("tariff_6", "MID", "tariff_1"): (-0.03149, 10, 0.6790),
    ("tariff_6", "MID", "tariff_10"): (0.04602, 25, 1.3300),
    ("tariff_6", "MID", "tariff_11"): (0.05729, 10, 1.2869),
    ("tariff_6", "MID", "tariff_12"): (0.01187, 2, 0.9607),
    ("tariff_6", "MID", "tariff_13"): (-0.05403, 28, 0.6750),
    ("tariff_6", "MID", "tariff_21"): (0.00179, 1, 0.9223),
    ("tariff_6", "MID", "tariff_4"): (0.01386, 13, 0.9348),
    ("tariff_6", "MID", "tariff_8"): (0.02861, 9, 1.1518),
    ("tariff_6", "MID", "tariff_9"): (0.00265, 3, 1.1169),
    ("tariff_7", "HIGH", "tariff_1"): (-0.02853, 6, 0.4129),
    ("tariff_7", "HIGH", "tariff_10"): (-0.02951, 26, 0.6070),
    ("tariff_7", "HIGH", "tariff_11"): (-0.00298, 8, 0.6345),
    ("tariff_7", "HIGH", "tariff_12"): (0.00992, 6, 1.0819),
    ("tariff_7", "HIGH", "tariff_13"): (-0.01895, 5, 0.4193),
    ("tariff_7", "HIGH", "tariff_4"): (-0.04397, 15, 0.7238),
    ("tariff_7", "HIGH", "tariff_8"): (-0.06847, 20, 0.5685),
    ("tariff_7", "LOW", "tariff_1"): (0.03567, 4, 1.7102),
    ("tariff_7", "LOW", "tariff_10"): (0.09111, 16, 1.5478),
    ("tariff_7", "LOW", "tariff_11"): (0.02332, 3, 1.3678),
    ("tariff_7", "LOW", "tariff_12"): (0.09158, 4, 1.6338),
    ("tariff_7", "LOW", "tariff_13"): (0.12146, 9, 1.7556),
    ("tariff_7", "LOW", "tariff_21"): (0.01383, 1, 1.6134),
    ("tariff_7", "LOW", "tariff_4"): (0.10442, 10, 1.8353),
    ("tariff_7", "LOW", "tariff_8"): (0.04962, 3, 1.8393),
    ("tariff_7", "LOW", "tariff_9"): (0.40889, 20, 1.6835),
    ("tariff_7", "MID", "tariff_1"): (-0.03592, 14, 0.7577),
    ("tariff_7", "MID", "tariff_10"): (0.01682, 31, 1.2229),
    ("tariff_7", "MID", "tariff_11"): (0.02085, 6, 1.0773),
    ("tariff_7", "MID", "tariff_12"): (0.01359, 2, 0.8433),
    ("tariff_7", "MID", "tariff_13"): (-0.02107, 18, 0.7326),
    ("tariff_7", "MID", "tariff_21"): (0.00215, 2, 0.8437),
    ("tariff_7", "MID", "tariff_4"): (0.05106, 15, 1.2446),
    ("tariff_7", "MID", "tariff_8"): (0.01185, 13, 1.0585),
    ("tariff_7", "MID", "tariff_9"): (0.00104, 3, 0.8519),
    ("tariff_8", "HIGH", "tariff_1"): (-0.02964, 49, 0.4431),
    ("tariff_8", "HIGH", "tariff_10"): (0.00498, 283, 0.4904),
    ("tariff_8", "HIGH", "tariff_11"): (0.00310, 133, 0.5619),
    ("tariff_8", "HIGH", "tariff_12"): (0.00373, 113, 0.6155),
    ("tariff_8", "HIGH", "tariff_13"): (-0.05390, 113, 0.2758),
    ("tariff_8", "HIGH", "tariff_21"): (-0.00013, 1, 0.4589),
    ("tariff_8", "HIGH", "tariff_4"): (-0.04075, 204, 0.4106),
    ("tariff_8", "LOW", "tariff_1"): (0.02334, 7, 1.3508),
    ("tariff_8", "LOW", "tariff_10"): (0.12400, 30, 1.7251),
    ("tariff_8", "LOW", "tariff_11"): (0.09676, 16, 1.7432),
    ("tariff_8", "LOW", "tariff_12"): (0.02899, 10, 1.4361),
    ("tariff_8", "LOW", "tariff_13"): (0.13412, 15, 1.6179),
    ("tariff_8", "LOW", "tariff_21"): (0.05084, 6, 1.8210),
    ("tariff_8", "LOW", "tariff_4"): (0.12487, 20, 1.8628),
    ("tariff_8", "LOW", "tariff_9"): (0.42896, 26, 1.4281),
    ("tariff_8", "MID", "tariff_1"): (-0.03493, 63, 0.6745),
    ("tariff_8", "MID", "tariff_10"): (0.09455, 182, 1.1654),
    ("tariff_8", "MID", "tariff_11"): (0.03475, 99, 1.1861),
    ("tariff_8", "MID", "tariff_12"): (0.01531, 55, 1.2137),
    ("tariff_8", "MID", "tariff_13"): (-0.02221, 164, 0.5392),
    ("tariff_8", "MID", "tariff_21"): (0.00030, 10, 0.9893),
    ("tariff_8", "MID", "tariff_4"): (-0.01961, 178, 0.7924),
    ("tariff_8", "MID", "tariff_9"): (0.00510, 19, 1.1929),
    ("tariff_9", "HIGH", "tariff_1"): (-0.03787, 4, 0.3946),
    ("tariff_9", "HIGH", "tariff_10"): (-0.02523, 4, 0.4254),
    ("tariff_9", "HIGH", "tariff_11"): (0.01937, 7, 0.7277),
    ("tariff_9", "HIGH", "tariff_13"): (-0.01512, 2, 0.4346),
    ("tariff_9", "HIGH", "tariff_4"): (-0.00931, 2, 0.4193),
    ("tariff_9", "HIGH", "tariff_8"): (-0.03733, 12, 0.4669),
    ("tariff_9", "LOW", "tariff_1"): (0.04745, 44, 1.1549),
    ("tariff_9", "LOW", "tariff_10"): (0.03241, 3, 1.7506),
    ("tariff_9", "LOW", "tariff_11"): (0.04486, 4, 1.7644),
    ("tariff_9", "LOW", "tariff_12"): (0.01592, 2, 1.4728),
    ("tariff_9", "LOW", "tariff_13"): (0.21000, 18, 1.4750),
    ("tariff_9", "LOW", "tariff_4"): (0.10018, 12, 1.8073),
    ("tariff_9", "LOW", "tariff_8"): (0.36336, 37, 1.6800),
    ("tariff_9", "MID", "tariff_1"): (-0.05188, 44, 0.7227),
    ("tariff_9", "MID", "tariff_10"): (0.01548, 11, 0.8709),
    ("tariff_9", "MID", "tariff_11"): (0.01782, 11, 1.0671),
    ("tariff_9", "MID", "tariff_12"): (0.01478, 10, 0.8032),
    ("tariff_9", "MID", "tariff_13"): (0.03495, 49, 0.7887),
    ("tariff_9", "MID", "tariff_4"): (0.06141, 37, 1.2476),
    ("tariff_9", "MID", "tariff_8"): (0.12014, 71, 1.2356),
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
