from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from synopsis import SynopsisConfig, VideoSynopsisProcessor
from synopsis.utils import ensure_dir

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = ensure_dir(BASE_DIR / "uploads")
OUTPUTS_DIR = ensure_dir(BASE_DIR / "outputs")

app = FastAPI(title="Video Synopsis Demo", version="1.0.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.mount("/media", StaticFiles(directory=OUTPUTS_DIR), name="media")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
processor = VideoSynopsisProcessor(SynopsisConfig())


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "title": "Video Synopsis Demo",
        },
    )


@app.post("/process", response_class=HTMLResponse)
async def process_video(request: Request, file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="File video tidak valid.")

    job_id = uuid.uuid4().hex
    job_upload_dir = ensure_dir(UPLOADS_DIR / job_id)
    job_output_dir = ensure_dir(OUTPUTS_DIR / job_id)

    input_path = job_upload_dir / Path(file.filename).name
    with input_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = processor.process(input_path, job_output_dir)
    except Exception as exc:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "request": request,
                "title": "Video Synopsis Demo",
                "error": str(exc),
            },
            status_code=400,
        )

    preview_url = f"/media/{job_id}/synopsis.mp4"
    background_url = f"/media/{job_id}/background.jpg"

    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={
            "request": request,
            "title": "Hasil Video Synopsis",
            "result": result,
            "preview_url": preview_url,
            "background_url": background_url,
            "compression_percent": round(result.compression_ratio * 100.0, 1),
        },
    )
