from fastapi import APIRouter

from src.db.sqlite_db import sqlite_ready

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "safe-code-api", "sqlite_ready": sqlite_ready()}
