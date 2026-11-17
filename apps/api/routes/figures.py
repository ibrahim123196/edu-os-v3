"""Textbook figures, as pictures, for the lesson page."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from apps.api.db import get_db
from apps.api.services import figure_service

router = APIRouter(prefix="/figures", tags=["figures"])


@router.get("/{document_id}/{label}.png")
def figure(document_id: uuid.UUID, label: str, page: int | None = None,
           db: Session = Depends(get_db)):
    path = figure_service.picture(db, document_id, label, page)
    if path is None:
        raise HTTPException(status_code=404, detail="No such figure.")
    return FileResponse(path, media_type="image/png",
                        headers={"Cache-Control": "public, max-age=86400"})
