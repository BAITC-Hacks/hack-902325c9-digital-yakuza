from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.beeline import RUN_TIMEOUT_SECONDS
from app.models.agent_run import AgentRun, CampaignResult, PilotResult


class RunInProgress(Exception):
    pass


async def expire_stale_runs(db: AsyncSession) -> None:
    now = datetime.now(timezone.utc)
    await db.execute(
        update(AgentRun)
        .where(
            AgentRun.status == "running",
            AgentRun.started_at < now - timedelta(seconds=RUN_TIMEOUT_SECONDS + 30),
        )
        .values(status="failed", finished_at=now, error="Run interrupted or timed out")
    )


async def create_run(db: AsyncSession, seed: int) -> AgentRun:
    locked = await db.scalar(select(func.pg_try_advisory_xact_lock(714230901)))
    if not locked:
        raise RunInProgress
    await expire_stale_runs(db)
    active = await db.scalar(select(AgentRun.id).where(AgentRun.status == "running").limit(1))
    if active:
        raise RunInProgress
    run = AgentRun(status="running", seed=seed)
    db.add(run)
    await db.commit()
    await db.refresh(run)
    return run


async def get_run(db: AsyncSession, run_id: UUID | None = None) -> AgentRun | None:
    await expire_stale_runs(db)
    await db.commit()
    query = select(AgentRun)
    if run_id:
        query = query.where(AgentRun.id == run_id)
    else:
        query = query.order_by(AgentRun.started_at.desc(), AgentRun.id.desc())
    return await db.scalar(query.limit(1))


async def get_pilots(db: AsyncSession, run_id: UUID) -> list[dict]:
    rows = await db.scalars(
        select(PilotResult).where(PilotResult.run_id == run_id).order_by(PilotResult.sequence)
    )
    return [row.payload for row in rows]


async def get_campaigns(db: AsyncSession, run_id: UUID) -> list[dict]:
    rows = await db.scalars(
        select(CampaignResult).where(CampaignResult.run_id == run_id).order_by(CampaignResult.sequence)
    )
    return [row.payload for row in rows]
