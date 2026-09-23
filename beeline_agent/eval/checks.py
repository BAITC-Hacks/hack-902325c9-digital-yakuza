"""
Проверки тренажёра и правил организаторов. Запуск: cd beeline_agent && python -m eval.checks

Тренажёр (ему можно верить, только если всё зелёное):
  T1  в мок-мире тренажёр даёт ровно то же число, что local_eval.py организаторов
  T2  потолок не ниже результата любой стратегии ни в одном мире
  T3  одинаковый seed → одинаковый мир, план и результат
  T4  калибровка: «только история» против потолка (в ТЗ: без разведки ~в 15 раз меньше)

Правила ТЗ для agent.py:
  R1  честная игра: нет доступа к внутренностям среды, чтения файлов и ключей в коде
  R2  время работы далеко от лимита 10 минут
  R3  запасной путь: агент не падает, если пилоты или план ломаются
  R4  без OPENAI_API_KEY всё работает
  R5  submission.csv в репо совпадает с тем, что сейчас выдаёт агент (сверка организаторов)
  R6  таблицы априора в agent.py: формат (q, q_se, n_obs) и каждое значение (конечное, q_se ≥ 0, n_obs — целое)
"""

import ast
import os
import sys
from pathlib import Path

import numpy as np

import local_eval
from agent import Agent
from eval.core import load_inputs, run_strategy, upper_bound
from sim.worlds import make_world
from eval.strategies import STRATEGIES

ROOT = Path(__file__).resolve().parents[1]
ENV_API = {"customer_profile", "tariffs", "channels", "remaining_budget", "remaining_contacts",
           "pilots_left", "run_pilot", "pilot_history", "total_budget", "max_total_contacts"}
FORBIDDEN_IMPORTS = {"environment", "mock_environment", "scoring_core", "local_eval", "make_submission",
                     "gc", "inspect", "ctypes", "importlib", "pickle", "subprocess"}
FORBIDDEN_ATTRS = {"__closure__", "__globals__", "__code__", "__dict__", "f_back", "f_locals", "cell_contents"}
FORBIDDEN_CALLS = {"open", "read_csv", "read_parquet", "read_excel", "read_json", "eval", "exec",
                   "compile", "__import__", "getattr", "vars", "globals"}

results = []


def report(code, ok, detail):
    status = {True: "OK", False: "FAIL", None: "ИНФО"}[ok]
    results.append((code, status, detail))
    print(f"{code:4} {status:5} {detail}", flush=True)


def t1_matches_local_eval():
    got = {}
    for name in ("agent", "template"):
        ours = run_strategy(make_world(0, "mock"), STRATEGIES[name], env_seed=42)["net_arpu_gain"]
        theirs = local_eval.evaluate_agent(STRATEGIES[name](), seed=42, verbose=False)["net_arpu_gain"]
        if abs(ours - theirs) > 1e-6:
            return report("T1", False, f"{name}: тренажёр {ours:,.0f} ≠ local_eval {theirs:,.0f}")
        got[name] = ours
    report("T1", True, "мок-мир, seed 42, совпадает с local_eval: " +
           ", ".join(f"{k} {v:,.0f}" for k, v in got.items()))


def t2_ceiling_is_ceiling(worlds):
    worst_gap = np.inf
    for world in worlds:
        ceiling = upper_bound(world)
        for name, make in STRATEGIES.items():
            net = run_strategy(world, make, env_seed=world.seed)["net_arpu_gain"]
            worst_gap = min(worst_gap, ceiling - net)
            if net > ceiling + 1e-6:
                return report("T2", False, f"{world.name}: {name} {net:,.0f} > потолок {ceiling:,.0f}")
    report("T2", True, f"{len(worlds)} миров × {len(STRATEGIES)} стратегии: потолок всегда выше "
                       f"(минимальный запас {worst_gap:,.0f})")


def t3_deterministic():
    w1, w2 = make_world(7, "random"), make_world(7, "random")
    if not w1.impact_model.equals(w2.impact_model):
        return report("T3", False, "make_world(7) дал разные миры")
    a = run_strategy(w1, STRATEGIES["agent"], env_seed=7)
    b = run_strategy(w2, STRATEGIES["agent"], env_seed=7)
    ok = a["net_arpu_gain"] == b["net_arpu_gain"] and a["plan"] == b["plan"]
    report("T3", ok, "тот же seed → тот же мир, план и результат" if ok
           else f"расхождение: {a['net_arpu_gain']:,.0f} vs {b['net_arpu_gain']:,.0f}")


def t4_calibration(worlds):
    shares = []
    for world in worlds:
        ceiling = upper_bound(world)
        net = run_strategy(world, STRATEGIES["history_only"], env_seed=world.seed)["net_arpu_gain"]
        shares.append(net / ceiling)
    med = float(np.median(shares))
    report("T4", None, f"«только история» берёт медиану {med:.0%} потолка (min {min(shares):.0%}); "
                       f"по ТЗ без разведки ≈ 1/15 ≈ 7% — если сильно больше, миры слишком похожи на историю")


def r1_honest_play():
    tree = ast.parse((ROOT / "agent.py").read_text(encoding="utf-8"))
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            problems += [f"import {n}" for n in names if n.split(".")[0] in FORBIDDEN_IMPORTS]
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRS:
                problems.append(f".{node.attr}")
            if isinstance(node.value, ast.Name) and node.value.id == "env" and node.attr not in ENV_API:
                problems.append(f"env.{node.attr} (нет в публичном API)")
        elif isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else ""
            if name in FORBIDDEN_CALLS:
                problems.append(f"вызов {name}()")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith(("sk-", "nvapi-")):
            problems.append("похоже на ключ API в коде")
    report("R1", not problems, "нет импортов среды, __closure__/gc, чтения файлов и ключей" if not problems
           else "; ".join(sorted(set(problems))))


def r2_time():
    seconds = max(run_strategy(make_world(s, "random"), STRATEGIES["agent"], env_seed=s)["seconds"] for s in range(3))
    report("R2", seconds < 120, f"агент работает {seconds:.1f} с (лимит ТЗ 600 с, наш запас — до 120 с)")


class _PlanCrashesOnce(Agent):
    calls = 0

    def _build_plan(self, *args, **kwargs):
        type(self).calls += 1
        if type(self).calls == 1:
            raise RuntimeError("искусственная ошибка в плане")
        return super()._build_plan(*args, **kwargs)


def r3_fallback():
    world = make_world(0, "mock")
    # а) все пилоты падают
    from environment import make_environment
    from mock_environment import CHANNELS, MAX_TOTAL_CONTACTS, TOTAL_BUDGET
    profile, dict_tariff = load_inputs()
    env, _ = make_environment(profile, world.impact_model, dict_tariff, CHANNELS, TOTAL_BUDGET,
                              MAX_TOTAL_CONTACTS, world.fallback_predict, seed=1)

    def broken_pilot(**kwargs):
        raise RuntimeError("пилот недоступен")
    env.run_pilot = broken_pilot
    try:
        plan_a = Agent(verbose=False).act(env)
    except Exception as exc:
        return report("R3", False, f"агент упал, когда пилоты недоступны: {exc}")
    # б) построение плана падает один раз → запасной план
    res = run_strategy(world, lambda: _PlanCrashesOnce(verbose=False), env_seed=42)
    if res["error"]:
        return report("R3", False, f"агент упал при ошибке в плане: {res['error']}")
    ok = isinstance(plan_a, list) and len(res["plan"]) > 0 and res["net_arpu_gain"] > 0
    report("R3", ok, f"пилоты сломаны → план из {len(plan_a)} кампаний без падения; "
                     f"ошибка в плане → запасной план {len(res['plan'])} кампаний, net {res['net_arpu_gain']:,.0f}")


def r4_no_key():
    saved = os.environ.pop("OPENAI_API_KEY", None)
    try:
        res = run_strategy(make_world(0, "mock"), STRATEGIES["agent"], env_seed=42)
    finally:
        if saved is not None:
            os.environ["OPENAI_API_KEY"] = saved
    report("R4", res["error"] is None and len(res["plan"]) > 0, "без OPENAI_API_KEY агент работает и выдаёт план")


def r5_submission_fresh():
    import io
    import make_submission
    fresh = make_submission.build_submission(Agent(verbose=False))
    buf = io.StringIO()
    fresh.to_csv(buf, index=False)
    committed = (ROOT / "submission.csv").read_text(encoding="utf-8")
    ok = buf.getvalue().splitlines() == committed.splitlines()
    report("R5", ok, "submission.csv совпадает с текущим агентом" if ok
           else "submission.csv устарел: запусти python make_submission.py и закоммить")


def r6_prior_tables():
    from eval.prior_tables import read_tables, validate
    problems = validate(read_tables(ROOT / "agent.py"))
    report("R6", not problems, "PRIOR и PRIOR_UNSEEN в формате (q, q_se, n_obs), значения корректны" if not problems
           else "; ".join(problems))


def main():
    worlds = [make_world(0, "mock")] + [make_world(s, sc) for s in range(3)
                                         for sc in ("random", "flip", "shift", "stingy", "unknown_rich", "high_rich")]
    r6_prior_tables()               # первой: при несовместимой таблице остальные проверки бессмысленны
    t1_matches_local_eval()
    t2_ceiling_is_ceiling(worlds)
    t3_deterministic()
    t4_calibration([make_world(s, "random") for s in range(10)])
    r1_honest_play()
    r2_time()
    r3_fallback()
    r4_no_key()
    r5_submission_fresh()
    failed = [r for r in results if r[1] == "FAIL"]
    print(f"\nитог: {len(results) - len(failed)} из {len(results)} без ошибок" + (" — ЕСТЬ ПРОВАЛЫ" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
