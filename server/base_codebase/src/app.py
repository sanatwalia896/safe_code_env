from fastapi import FastAPI

from src.api.health import router as health_router
from src.api.users import router as users_router
from src.db.sqlite_db import init_db

app = FastAPI(title="Safe Code Base")
app.include_router(health_router)
app.include_router(users_router, prefix="/users", tags=["users"])


@app.on_event("startup")
def on_startup() -> None:
    init_db()
