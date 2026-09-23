"""
Стратегии для сравнения в тренажёре.

  template      — шаблон организаторов (agent_template.py), нижняя планка
  history_only  — верим истории без проверки пилотами: берём всё, что история считает прибыльным
  agent         — наш агент (agent.py)

Потолок (знает истинные эффекты) считается отдельно: eval.core.upper_bound.
"""

import time

from agent import Agent
from agent_template import Agent as TemplateAgent


class HistoryOnly:
    """План только по истории, без единого пилота. Показывает, сколько стоит «не проверять»."""

    def act(self, env):
        agent = Agent(verbose=False)
        agent.trace, agent._t0 = [], time.monotonic()
        candidates = agent._make_candidates(env.customer_profile, env)
        for c in candidates:
            c.sd = 1e-9                  # полная уверенность в истории: нижняя граница = оценка
        return agent._build_plan(candidates, env, require_pilot=False)


STRATEGIES = {
    "template": TemplateAgent,
    "history_only": HistoryOnly,
    "agent": lambda: Agent(verbose=False),
}
