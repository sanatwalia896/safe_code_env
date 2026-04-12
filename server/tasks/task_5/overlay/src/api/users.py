from fastapi import APIRouter
from pydantic import BaseModel

from src.services.user_service import register_user

router = APIRouter()


class UserCreateRequest(BaseModel):
    email: str
    display_name: str


@router.post("")
def create_user(payload: UserCreateRequest) -> dict:
    user = register_user(payload.email, payload.display_name)
    return {"ok": True, "user": user}
