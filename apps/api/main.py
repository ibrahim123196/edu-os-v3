"""EduOS API — Milestone 1: Grounded Study Chat."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from apps.api.config import settings
from apps.api.db import init_db
from apps.api.routes import chat, documents, health

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
log = logging.getLogger("eduos")

WEB_DIR = Path(__file__).resolve().parents[2] / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting %s (%s)", settings.app_name, settings.environment)
    init_db()
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    yield
    log.info("Shutting down")


app = FastAPI(
    title="EduOS API",
    version="0.1.0",
    description=(
        "Grounded study chat: upload your own material, ask questions, get "
        "answers that cite the page they came from."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(chat.router)

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(str(WEB_DIR / "index.html"))
