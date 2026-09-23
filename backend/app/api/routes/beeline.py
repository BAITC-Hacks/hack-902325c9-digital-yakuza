import asyncio
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.beeline import agent_directory
from app.core.database import get_db
from app.repositories.agent_runs import (
    RunInProgress, create_run, get_run, get_pilots, get_campaigns,
)
from app.schemas.agent import RunRequest, RunResponse, PilotsResponse
from app.services.agent_runs import execute_run
from app.services.case_summary import get_case_summary

router = APIRouter(prefix="/api", tags=["beeline"])
Database = Annotated[AsyncSession, Depends(get_db)]


async def require_run(db, run_id):
    run = await get_run(db, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Agent run not found")
    return run


async def serialize_run(db, run):
    return RunResponse(
        run_id=run.id, status=run.status, seed=run.seed,
        started_at=run.started_at, finished_at=run.finished_at,
        error=run.error, metrics=run.metrics, trace=run.trace or [],
        campaigns=await get_campaigns(db, run.id), explanation=run.explanation,
    )


@router.get("/case/summary")
async def case_summary() -> dict:
    try:
        return await asyncio.to_thread(get_case_summary)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Beeline data package unavailable") from exc


@router.post("/agent/run", status_code=202, response_model=RunResponse)
async def start_run(
    background_tasks: BackgroundTasks,
    db: Database,
    request: RunRequest | None = None,
):
    try:
        await asyncio.to_thread(agent_directory)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Beeline data package unavailable") from exc
    try:
        run = await create_run(db, (request or RunRequest()).seed)
    except RunInProgress as exc:
        raise HTTPException(status_code=409, detail="An agent run is already in progress") from exc
    background_tasks.add_task(execute_run, run.id, run.seed)
    return await serialize_run(db, run)


@router.get("/agent/result", response_model=RunResponse)
async def agent_result(db: Database, run_id: UUID | None = None):
    return await serialize_run(db, await require_run(db, run_id))


@router.get("/agent/pilots", response_model=PilotsResponse)
async def agent_pilots(db: Database, run_id: UUID | None = None):
    run = await require_run(db, run_id)
    return PilotsResponse(
        run_id=run.id, status=run.status, pilots=await get_pilots(db, run.id)
    )


@router.get("/agent/submission")
async def agent_submission(db: Database, run_id: UUID | None = None):
    run = await require_run(db, run_id)
    if run.status != "completed" or run.submission_csv is None:
        raise HTTPException(status_code=409, detail="Submission is not ready")
    return Response(
        content=run.submission_csv, media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="submission.csv"'},
    )
