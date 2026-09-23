import asyncio
import hashlib
import io
import json
import os
import time
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request, urlopen
from uuid import UUID

import pandas as pd
from sqlalchemy import select

from app.core.beeline import agent_directory
from app.core.database import engine, get_database_url, session_factory
from app.models.agent_run import AgentRun, CampaignResult, PilotResult
from app.services.agent_worker import decision_info, observed_agent
from app.services.agent_runs import redact_secrets


def request(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = Request(
        "http://127.0.0.1:8000" + path, data=data,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=15) as response:
        body = response.read()
        return response.status, json.loads(body) if "application/json" in response.headers.get("Content-Type", "") else body


def check_contract(run, pilots):
    for field in (
        "run_id", "status", "seed", "started_at", "finished_at", "error",
        "environment", "metrics", "trace", "campaigns",
    ):
        assert field in run, field
    assert run["status"] == "completed", run
    assert run["environment"] == "official_mock"
    assert isinstance(run["trace"], list) and run["trace"]
    assert isinstance(run["metrics"], dict)
    assert isinstance(run["warnings"], list)
    assert run["warnings"] == run["metrics"]["warnings"]
    assert run["explanation"] is not None
    report = run["explanation"]["explanation"]
    assert run["explanation"]["source"] == report["source"]
    assert run["explanation"]["rendered"] == report["rendered"]
    assert report["source"] in {"llm", "template", "unavailable"}
    assert run["warnings"] == (report.get("rendered") or {}).get("warnings", [])
    for pilot in pilots:
        for field in (
            "sequence", "hypothesis", "segment", "target_tariff", "channel",
            "pilot_size", "observed_lift", "cost", "mu", "sd", "lcb", "candidate_id", "candidate",
        ):
            assert field in pilot, field
        assert isinstance(pilot["sequence"], int)
        assert isinstance(pilot["segment"], dict)
        assert isinstance(pilot["hypothesis"], str)
        assert 10 <= pilot["pilot_size"] <= 200
    events = [event for event in run["trace"] if event["kind"] == "pilot"]
    assert len(events) == len(pilots)
    for event, pilot in zip(events, pilots):
        assert pilot["candidate_id"] == event["candidate"]
        assert pilot["candidate"] == event["candidate"]
        for field in ("mu", "sd", "lcb"):
            assert pilot[field] == event[field]
    for campaign in run["campaigns"]:
        for field in ("campaign_name", "target_tariff", "channel", "n_contacts", "cost", "gross_lift"):
            assert field in campaign, field
        assert campaign["n_contacts"] <= 5000
    assert 1 <= len(run["campaigns"]) <= 10
    assert 1 <= len(pilots) <= 20
    assert run["metrics"]["total_cost"] <= 100000
    assert run["metrics"]["total_contacts"] <= 15000


async def check_database(run, pilots):
    async with session_factory() as db:
        run_id = UUID(run["run_id"])
        saved = await db.get(AgentRun, run_id)
        assert saved.status == "completed"
        assert saved.trace == run["trace"]
        assert saved.metrics == run["metrics"]
        assert saved.submission_csv
        assert saved.explanation == run["explanation"]
        for model, expected in ((PilotResult, pilots), (CampaignResult, run["campaigns"])):
            rows = await db.scalars(select(model).where(model.run_id == run_id).order_by(model.sequence))
            assert [row.payload for row in rows] == expected
    await engine.dispose()


def check_fallback(agent_class, make_env):
    class FailingPlanAgent(agent_class):
        def _build_plan(self, *args, **kwargs):
            raise ValueError("local validation: unavailable normal plan")

    env, _ = make_env(seed=42)
    emitted = []
    with patch("app.services.agent_worker.emit", side_effect=lambda kind, data: emitted.append((kind, data))):
        adapter = observed_agent(FailingPlanAgent, env)
        campaigns = adapter.act(env)
    info = decision_info(adapter.trace)
    assert len(campaigns) == 1 and campaigns[0]["channel"] == "push"
    assert info["is_fallback"] and info["fallback_reason"]
    assert info["warnings"] == []
    event = next(row for row in adapter.trace if row["kind"] == "minimal_campaign")
    assert info["estimate_source"] == event["estimate"]
    assert all(value == event[key] for key, value in info["risk_info"].items())
    assert [data for kind, data in emitted if kind == "decision"] == adapter.trace
    assert len([1 for kind, _ in emitted if kind == "pilot_update"]) == len(env.pilot_history)


def check_explanation_failure(run, pilots):
    import workflow
    token = "local-check-secret-do-not-store"
    result = {
        "campaigns": run["campaigns"], "pilots": pilots, "trace": run["trace"],
        "metrics": run["metrics"],
        "total_budget": run["metrics"]["total_cost"] + run["metrics"]["remaining_budget"],
    }
    before = json.dumps(result, sort_keys=True)
    with patch.dict(os.environ, {"AGENT_LLM": "1", "OPENAI_API_KEY": token}), \
            patch.object(workflow, "_make_client", side_effect=RuntimeError(token)), \
            patch.object(workflow.Agent, "act", side_effect=AssertionError("Agent must not run again")) as act:
        explanation = workflow.explain_result(result, use_llm=True)
        assert act.call_count == 0
    assert explanation["explanation"]["source"] == "template"
    assert before == json.dumps(result, sort_keys=True)
    safe = redact_secrets(explanation, [token])
    assert token not in json.dumps(safe)
    assert safe["explanation"]["rendered"]["warnings"] == explanation["explanation"]["rendered"]["warnings"]


def main():
    if get_database_url().host != "postgres":
        raise RuntimeError("Run this check only in local Compose with DATABASE_URL host postgres")
    assert request("/")[0] == 200
    assert request("/health") == (200, {"status": "ok", "database": "ok"})
    assert request("/docs")[0] == 200
    _, spec = request("/openapi.json")
    for path, method in (
        ("/api/case/summary", "get"), ("/api/agent/run", "post"),
        ("/api/agent/result", "get"), ("/api/agent/pilots", "get"),
        ("/api/agent/submission", "get"),
    ):
        assert method in spec["paths"][path]
    _, summary = request("/api/case/summary")
    for field in ("customers", "tariffs", "segments", "channels", "constraints"):
        assert field in summary, field
    directory = agent_directory()
    import agent
    from local_eval import evaluate_agent
    from make_submission import build_submission, CAMPAIGN_COLUMNS, SUBMISSION_SEED
    from mock_environment import make_mock_env

    assert Path(agent.__file__).resolve() == directory / "agent.py"
    assert summary["agent_sha256"] == hashlib.sha256((directory / "agent.py").read_bytes()).hexdigest()
    strategy = summary["strategy"]
    assert strategy["max_pilot_budget_fraction"] == agent.EXPLORE_BUDGET_SHARE
    assert strategy["max_pilot_contacts_fraction"] == agent.EXPLORE_CONTACT_SHARE
    assert strategy["exploration_timeout_seconds"] == agent.TIME_BUDGET_S
    assert strategy["channel_economics"] == agent.CHANNEL_ECONOMICS
    assert strategy["risk_k"] == agent.RISK_K
    assert summary["agent_status"] == "CURRENT"
    assert summary["agent_sha256_lf"] == hashlib.sha256((directory / "agent.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    status, run = request("/api/agent/run", {"seed": SUBMISSION_SEED})
    assert status == 202
    run_id = run["run_id"]
    deadline = time.monotonic() + 600
    incremental = False
    while run["status"] == "running":
        assert time.monotonic() < deadline, "Agent timed out"
        time.sleep(0.05)
        _, run = request(f"/api/agent/result?run_id={run_id}")
        incremental |= run["status"] == "running" and bool(run["trace"])
    explanation_deadline = time.monotonic() + 180
    while run["status"] == "completed" and run.get("explanation") is None:
        assert time.monotonic() < explanation_deadline, "Explanation timed out"
        time.sleep(0.1)
        _, run = request(f"/api/agent/result?run_id={run_id}")
    _, pilot_response = request(f"/api/agent/pilots?run_id={run_id}")
    pilots = pilot_response["pilots"]
    check_contract(run, pilots)
    status, csv = request(f"/api/agent/submission?run_id={run_id}")
    assert status == 200
    os.chdir(directory)
    official = evaluate_agent(agent.Agent(verbose=False), seed=SUBMISSION_SEED, verbose=False)
    assert official["status"] == "PASS", official
    for key in ("total_cost", "total_contacts", "net_arpu_gain"):
        assert run["metrics"][key] == official[key], key
    expected = build_submission(agent.Agent(verbose=False)).to_csv(index=False)
    assert csv.decode() == expected
    assert csv.decode().replace("\r\n", "\n") == (directory / "submission.csv").read_text(encoding="utf-8")
    assert list(pd.read_csv(io.BytesIO(csv)).columns) == CAMPAIGN_COLUMNS
    assert [c["channel"] for c in run["campaigns"]] == ["sms", "digital_ads", "sms", "sms", "digital_ads"]
    assert len(run["campaigns"]) == 5 and len(pilots) == 20
    assert run["metrics"]["total_cost"] == 95094
    assert run["metrics"]["total_contacts"] == 13320
    assert round(run["metrics"]["net_arpu_gain"]) == 5165453
    channels = [event for event in run["trace"] if event["kind"] == "channels"]
    assert channels and "budget_left" in channels[-1]
    for campaign in run["campaigns"]:
        assert channels[-1]["assigned"][campaign["candidate_id"]] == campaign["channel"]
    assert run["explanation"]["campaign_pilot_links"] == {
        "C1": ["P5", "P6"], "C2": ["P13", "P14"], "C3": ["P15", "P16"],
        "C4": ["P7", "P8"], "C5": ["P9", "P10"],
    }
    check_fallback(agent.Agent, make_mock_env)
    check_explanation_failure(run, pilots)
    asyncio.run(check_database(run, pilots))
    print(json.dumps({
        "check": "PASS", "run_id": run_id, "database": "OK",
        "official_evaluator": official["status"], "csv_matches_official": True,
        "fallback_adapter": "OK", "incremental_trace_observed": incremental,
        "campaigns": len(run["campaigns"]), "pilots": len(pilots),
        "cost": run["metrics"]["total_cost"], "contacts": run["metrics"]["total_contacts"],
        "net": run["metrics"]["net_arpu_gain"],
        "channels": [c["channel"] for c in run["campaigns"]],
        "explanation_source": run["explanation"]["explanation"]["source"],
        "explanation_failure_safe": True,
        "max_campaign_size": max(row["n_contacts"] for row in run["campaigns"]),
    }, indent=2))


if __name__ == "__main__":
    main()
