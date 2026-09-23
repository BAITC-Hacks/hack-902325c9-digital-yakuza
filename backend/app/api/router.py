from fastapi import APIRouter

from app.api.routes.health import router as health_router
from app.api.routes.beeline import router as beeline_router
from app.core.config import get_settings

router = APIRouter()
router.include_router(health_router)
router.include_router(beeline_router)


@router.get("/", tags=["info"])
async def root() -> dict[str, str]:
    return {"app": get_settings().app_name, "docs": "/docs"}
