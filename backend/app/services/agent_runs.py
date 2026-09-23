import asyncio
import json
import logging
import sys
import tempfile
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select

from app.core.beeline import RUN_TIMEOUT_SECONDS
from app.core.database import session_factory
from app.models.agent_run import AgentRun, CampaignResult, PilotResult
from app.services.agent_worker import decision_info

logger = logging.getLogger(__name__)


async def save_event(run_id: UUID, kind: str, data) -> None:
    async with session_factory() as db:
        run = await db.get(AgentRun, run_id)
        if run is None or run.status != "running":
            raise RuntimeError("Run is no longer active")
        if kind == "pilot":
            db.add(PilotResult(run_id=run_id, sequence=data["sequence"], payload=data))
        elif kind == "pilot_update":
            pilot = await db.scalar(select(PilotResult).where(
                PilotResult.run_id == run_id, PilotResult.sequence == data["sequence"],
            ))
            if pilot is None:
                raise ValueError("Pilot update received before pilot result")
            pilot.payload = {**pilot.payload, **data}
        elif kind == "decision":
            run.trace = [*(run.trace or []), data]
            run.metrics = {**(run.metrics or {}), **decision_info(run.trace)}
        elif kind == "trace":
            run.trace = data
            run.metrics = {**(run.metrics or {}), **decision_info(data)}
        await db.commit()


async def complete_run(run_id: UUID, result: dict) -> None:
    async with session_factory() as db:
        run = await db.get(AgentRun, run_id)
        if run is None or run.status != "running":
            return
        for sequence, campaign in enumerate(result["campaigns"], 1):
            db.add(CampaignResult(run_id=run_id, sequence=sequence, payload=campaign))
        run.metrics = result["metrics"]
        run.submission_csv = result["submission_csv"]
        run.status = "completed"
        run.finished_at = datetime.now(timezone.utc)
        await db.commit()


async def fail_run(run_id: UUID, error: str) -> None:
    async with session_factory() as db:
        run = await db.get(AgentRun, run_id)
        if run and run.status == "running":
            run.status = "failed"
            run.error = error
            run.finished_at = datetime.now(timezone.utc)
            await db.commit()


async def execute_run(run_id: UUID, seed: int) -> None:
    process = None
    with tempfile.TemporaryFile() as errors:
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-u", "-m", "app.services.agent_worker",
                "--seed", str(seed),
                stdout=asyncio.subprocess.PIPE, stderr=errors,
                limit=1024 * 1024,
            )

            async def consume():
                result = None
                async for line in process.stdout:
                    event = json.loads(line)
                    if event["type"] in {"pilot", "pilot_update", "decision", "trace"}:
                        await save_event(run_id, event["type"], event["data"])
                    elif event["type"] == "result":
                        result = event["data"]
                code = await process.wait()
                if code or result is None:
                    raise RuntimeError("Agent process failed or returned no result")
                return result

            result = await asyncio.wait_for(consume(), timeout=RUN_TIMEOUT_SECONDS)
            await complete_run(run_id, result)
        except TimeoutError:
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()
            await fail_run(run_id, "Agent exceeded the 600 second limit")
        except asyncio.CancelledError:
            await asyncio.shield(fail_run(run_id, "Backend stopped during agent execution"))
            raise
        except Exception:
            errors.seek(0)
            details = errors.read().decode("utf-8", errors="replace")[-4000:]
            logger.exception("Agent run %s failed: %s", run_id, details)
            await fail_run(run_id, "Agent execution failed; see backend logs")
        finally:
            if process is not None and process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()
