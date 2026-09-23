import argparse
import inspect
import json
import math
import time

import numpy as np
import pandas as pd

from app.core.beeline import agent_directory, agent_sha256, RUN_TIMEOUT_SECONDS


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def emit(kind, data):
    print(json.dumps(
        {"type": kind, "data": json_safe(data)}, ensure_ascii=False, allow_nan=False
    ), flush=True)


def observed_agent(agent_class, env):
    class ObservedAgent(agent_class):
        def _log(self, kind, **data):
            super()._log(kind, **data)
            emit("decision", self.trace[-1])
            if kind == "pilot" and env.pilot_history:
                update = {
                    key: data[key] for key in ("mu", "sd", "lcb") if key in data
                }
                if "candidate" in data:
                    update["candidate_id"] = data["candidate"]
                emit("pilot_update", {"sequence": len(env.pilot_history), **update})

    return ObservedAgent(verbose=False)


def decision_info(trace):
    warnings = []
    stop_reason = None
    fallback_reason = None
    fallback = None
    recovered = False
    for event in trace:
        kind = event.get("kind")
        if kind == "explore_stop":
            stop_reason = event.get("reason")
            warnings.append({"code": "partial_exploration", "event": event})
        elif kind in {"error", "pilot_error", "fallback_error", "minimal_error"}:
            warnings.append({"code": "recoverable_error", "event": event})
            if kind == "error":
                recovered = True
                fallback_reason = event.get("error")
        elif kind in {"plan_drop", "minimal_skip"}:
            warnings.append({"code": "strategy_warning", "event": event})
        elif kind == "minimal_campaign":
            fallback = event
            fallback_reason = event.get("reason")
            warnings.append({"code": "fallback_used", "event": event})
    if recovered and fallback is None:
        warnings.append({"code": "fallback_used", "reason": fallback_reason})
    info = {
        "warnings": warnings,
        "stop_reason": stop_reason,
        "is_fallback": recovered or fallback is not None,
        "fallback_reason": fallback_reason,
    }
    if fallback is not None:
        info["estimate_source"] = fallback.get("estimate")
        info["risk_info"] = {
            key: fallback[key]
            for key in ("mu", "sd", "downside", "exposure_arpu", "expected_gain")
            if key in fallback
        }
    return info


def run(seed):
    started = time.monotonic()
    directory = agent_directory()
    version = agent_sha256()
    from agent import Agent
    from environment import MAX_PILOTS, MIN_PILOT_CUSTOMERS, MAX_PILOT_CUSTOMERS
    from make_submission import CAMPAIGN_COLUMNS
    from mock_environment import make_mock_env, _mock_impact_model, _mock_fallback
    from scoring_core import MAX_CAMPAIGNS, MAX_CUSTOMERS_PER_CAMPAIGN
    from scoring_core import score_campaigns, validate_strategy

    env, internals = make_mock_env(
        seed=seed, data_dir=str(directory / "data"),
        profile_path=str(directory / "customer_profile.csv"),
    )
    original_pilot = env.run_pilot
    signature = inspect.signature(original_pilot)

    def record_pilot(*args, **kwargs):
        arguments = signature.bind(*args, **kwargs)
        arguments.apply_defaults()
        result = original_pilot(*args, **kwargs)
        filters = {
            key: value for key, value in arguments.arguments.items()
            if key.startswith("filter_") and value is not None
        }
        emit("pilot", {
            **result,
            "sequence": len(env.pilot_history),
            "hypothesis": (
                f"Проверка прироста ARPU при переходе на {result['target_tariff']} "
                f"через {result['channel']} для выбранного сегмента."
            ),
            "segment": filters,
            "pilot_size": result["n_customers"],
            "observed_lift": result["observed_lift_ratio"],
        })
        return result

    env.run_pilot = record_pilot
    agent = observed_agent(Agent, env)
    campaigns = agent.act(env)
    emit("trace", agent.trace)
    if not isinstance(campaigns, list) or not 1 <= len(campaigns) <= MAX_CAMPAIGNS:
        raise ValueError("Agent must return 1–10 final campaigns")
    validate_strategy(pd.DataFrame(campaigns), env.tariffs)
    if not 1 <= len(env.pilot_history) <= MAX_PILOTS:
        raise ValueError("Agent must execute 1–20 pilots")
    if any(
        not MIN_PILOT_CUSTOMERS <= row["n_customers"] <= MAX_PILOT_CUSTOMERS
        for row in env.pilot_history
    ):
        raise ValueError("Invalid pilot size")
    pilots = internals.executed_pilot_campaigns()
    strategy = pd.DataFrame(pilots + campaigns)
    for column in list(CAMPAIGN_COLUMNS) + ["explicit_ids"]:
        if column not in strategy:
            strategy[column] = None
    model = _mock_impact_model(pd.read_csv(directory / "data" / "change_tariff.csv"))
    score = score_campaigns(
        strategy, env.customer_profile, model, env.tariffs,
        float(env.customer_profile["predicted_arpu"].sum()), _mock_fallback,
        team_id="backend",
    )
    if score["total_cost"] > env.total_budget or score["total_contacts"] > env.max_total_contacts:
        raise ValueError("Budget or contact limit exceeded")
    if any(row["n_contacts"] > MAX_CUSTOMERS_PER_CAMPAIGN for row in score["campaigns_detail"]):
        raise ValueError("Campaign size exceeded")
    elapsed = time.monotonic() - started
    if elapsed > RUN_TIMEOUT_SECONDS:
        raise ValueError("Agent runtime exceeded")
    details = score.pop("campaigns_detail")
    finals = [
        {**campaign, **detail}
        for campaign, detail in zip(campaigns, details[len(pilots):])
    ]
    info = decision_info(agent.trace)
    if info["is_fallback"]:
        for campaign in finals:
            campaign.update({
                key: info[key]
                for key in ("is_fallback", "fallback_reason", "estimate_source", "risk_info")
                if key in info
            })
    csv = pd.DataFrame(campaigns).reindex(columns=CAMPAIGN_COLUMNS).to_csv(index=False)
    emit("result", {
        "campaigns": finals,
        "submission_csv": csv,
        "metrics": {
            **score,
            **info,
            "environment": "official_mock",
            "score_source": "official_mock_scoring_not_judging_score",
            "final_campaign_count": len(campaigns),
            "pilot_count": len(pilots),
            "pilot_cost": sum(row["cost"] for row in env.pilot_history),
            "pilot_contacts": sum(row["n_customers"] for row in env.pilot_history),
            "remaining_budget": env.total_budget - score["total_cost"],
            "remaining_contacts": env.max_total_contacts - score["total_contacts"],
            "elapsed_seconds": elapsed,
            "agent_sha256": version,
        },
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    run(arguments.seed)
