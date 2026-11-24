"""Logging in and out. Accounts themselves are made with scripts.accounts."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from apps.api import auth
from apps.api.config import get_settings
from apps.api.db import get_db
from apps.api.student import TEST_STUDENT_ID, Student, current_or_none

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    email: str = Field(min_length=1, max_length=320)
    password: str = Field(min_length=1, max_length=200)


class SignupRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    email: str = Field(min_length=1, max_length=320)
    password: str = Field(min_length=1, max_length=200)


def _me(student: Student) -> dict:
    return {
        "logged_in": True,
        "id": str(student.id),
        "name": student.display_name,
        "email": student.email,
        # The page says so: a progress page full of test clicks looks
        # exactly like a real one.
        "is_test": student.id == TEST_STUDENT_ID,
    }


def _start(db: Session, response: Response, student: Student) -> None:
    token = auth.start_session(db, student.id)
    response.set_cookie(
        auth.COOKIE, token,
        max_age=auth.SESSION_DAYS * 24 * 3600,
        httponly=True,  # page scripts cannot read it
        samesite="lax",  # not sent with another site's form posts
        secure=get_settings().environment != "local",  # https only, once deployed
        path="/",
    )


@router.post("/login")
def login(payload: LoginRequest, response: Response, db: Session = Depends(get_db)):
    try:
        student = auth.log_in(db, payload.email, payload.password)
    except auth.LoginError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
    _start(db, response, student)
    return _me(student)


@router.post("/signup")
def signup(payload: SignupRequest, request: Request, response: Response,
           db: Session = Depends(get_db)):
    """A student makes their own account and is logged in straight away."""
    if not get_settings().open_signup:
        raise HTTPException(status_code=403, detail="New accounts are made by your teacher.")
    address = request.client.host if request.client else ""
    try:
        student = auth.sign_up(db, payload.name, payload.email, payload.password, address)
    except auth.LoginError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
    _start(db, response, student)
    return _me(student)


@router.post("/logout")
def logout(request: Request, response: Response, db: Session = Depends(get_db)):
    token = request.cookies.get(auth.COOKIE)
    if token:
        auth.end_session(db, token)
    response.delete_cookie(auth.COOKIE, path="/")
    return {"logged_in": False}


@router.get("/me")
def me(db: Session = Depends(get_db)):
    """Who is logged in. Public, so the page can ask before showing itself."""
    student_id = current_or_none()
    student = db.get(Student, student_id) if student_id else None
    if student is None:
        # Whether to offer "Create an account" on the login screen.
        return {"logged_in": False, "signup": get_settings().open_signup}
    return _me(student)
