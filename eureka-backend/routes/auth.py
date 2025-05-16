import os
import bcrypt
from jose import jwt
from dotenv import load_dotenv
from db.database import user_collection
from fastapi import APIRouter, HTTPException
from schemas.user import UserCreate, UserLogin
from datetime import datetime, timedelta, timezone
from authentication.rbac import get_user_access

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")

router = APIRouter(prefix="/auth", tags=["Auth"])

@router.post("/signup")
def signup(user: UserCreate):
    existing_user = user_collection.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")

    hashed_pw = bcrypt.hashpw(user.password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    user_dict = {
        "name": user.name,
        "email": user.email,
        "hashed_password": hashed_pw
    }

    user_collection.insert_one(user_dict)
    return {"message": "User registered successfully"}

@router.post("/login")
def login(user: UserLogin):
    db_user = user_collection.find_one({"email": user.email})
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    if not bcrypt.checkpw(user.password.encode("utf-8"), db_user["hashed_password"].encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid password")

    try:
        scopes = get_user_access(user.email)
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))

    payload = {
        "sub": user.email,
        "scopes": scopes,
        "exp": datetime.now(timezone.utc) + timedelta(hours=2)
    }

    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

    return {
        "access_token": token,
        "token_type": "bearer",
        "scopes": scopes,
        "name": db_user["name"]
    }
