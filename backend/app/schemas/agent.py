from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    seed: int = Field(default=42, ge=0, le=2147483647)


class RunResponse(BaseModel):
    run_id: UUID
    status: Literal["running", "completed", "failed"]
    seed: int
    started_at: datetime
    finished_at: datetime | None = None
    error: str | None = None
    environment: Literal["official_mock"] = "official_mock"
    metrics: dict[str, Any] | None = None
    trace: list[dict[str, Any]] = Field(default_factory=list)
    campaigns: list[dict[str, Any]] = Field(default_factory=list)


class PilotsResponse(BaseModel):
    run_id: UUID
    status: Literal["running", "completed", "failed"]
    pilots: list[dict[str, Any]]
