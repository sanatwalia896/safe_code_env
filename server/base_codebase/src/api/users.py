from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr

from src.services.user_service import register_user

router = APIRouter()


class UserCreateRequest(BaseModel):
    email: EmailStr
    display_name: str


@router.post("")
def create_user(payload: UserCreateRequest) -> dict:
    user = register_user(payload.email, payload.display_name)
    if not user:
        raise HTTPException(status_code=400, detail="User already exists")
    return user
