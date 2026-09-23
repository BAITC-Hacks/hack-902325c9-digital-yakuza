"""
Проверка вшитых таблиц априора: формат, содержимое и сравнение двух версий agent.py.

    python -m eval.prior_tables                 # проверить таблицы в текущем agent.py
    python -m eval.prior_tables --diff OLD.py   # сравнить с другой версией agent.py

Таблицы читаются из текста agent.py (ast.literal_eval), без импорта агента,
поэтому можно сравнивать версии из git: git show HEAD:beeline_agent/agent.py > /tmp/old.py
"""

import argparse
import ast
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FORMAT = ("q", "q_se", "n_obs")
Q_SE_MAX = 1.0          # q — доля ARPU; стандартная ошибка больше 1 означает не те единицы


def read_tables(agent_path):
    """{'PRIOR_FORMAT': ..., 'PRIOR': {...}, 'PRIOR_UNSEEN': {...}} из текста agent.py (последнее присваивание)."""
    tree = ast.parse(Path(agent_path).read_text(encoding="utf-8"))
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("PRIOR_FORMAT", "PRIOR", "PRIOR_UNSEEN"):
                found[name] = ast.literal_eval(node.value)
    return found


def validate(tables):
    """Список проблем (пусто = всё хорошо). Проверяет подпись формата и каждое значение."""
    problems = []
    if tables.get("PRIOR_FORMAT") != EXPECTED_FORMAT:
        problems.append(f"PRIOR_FORMAT = {tables.get('PRIOR_FORMAT')!r}, ожидается {EXPECTED_FORMAT}")
    for name, min_n in (("PRIOR", 1), ("PRIOR_UNSEEN", 0)):
        table = tables.get(name)
        if not table:
            problems.append(f"{name} пустая или отсутствует")
            continue
        bad = []
        for key, row in table.items():
            ok = (isinstance(row, tuple) and len(row) == 3
                  and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in row)
                  and math.isfinite(row[0]) and math.isfinite(row[1])
                  and 0 <= row[1] <= Q_SE_MAX
                  and isinstance(row[2], int) and row[2] >= min_n)
            if not ok:
                bad.append((key, row))
        if bad:
            problems.append(f"{name}: {len(bad)} из {len(table)} строк не в формате (q, q_se, n_obs), "
                            f"пример {bad[0][0]} → {bad[0][1]}")
    return problems


def diff(old, new):
    """Что поменялось между версиями: набор ключей, q и n_obs (q_se может появиться/измениться)."""
    lines = []
    for name in ("PRIOR", "PRIOR_UNSEEN"):
        a, b = old.get(name, {}), new.get(name, {})
        if set(a) != set(b):
            lines.append(f"{name}: ключи отличаются (+{len(set(b) - set(a))} / −{len(set(a) - set(b))})")
            continue
        old_n = 1 if old.get("PRIOR_FORMAT") is None else 2      # старый формат: (q, n_obs, pct_std)
        dq = [k for k in a if abs(a[k][0] - b[k][0]) > 1e-9]
        dn = [k for k in a if a[k][old_n] != b[k][2]]
        lines.append(f"{name}: {len(a)} ключей совпадают; q изменился у {len(dq)}, n_obs у {len(dn)}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default=str(ROOT / "agent.py"))
    ap.add_argument("--diff", default=None, help="путь к другой версии agent.py для сравнения")
    args = ap.parse_args()
    new = read_tables(args.agent)
    problems = validate(new)
    print("формат и содержимое: " + ("OK" if not problems else "ПРОБЛЕМЫ"))
    for p in problems:
        print("  -", p)
    if args.diff:
        for line in diff(read_tables(args.diff), new):
            print("  diff:", line)


if __name__ == "__main__":
    main()
