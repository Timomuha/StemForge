"""
Stem splitter backend (FastAPI + Demucs).

Flow:
  1. POST /api/upload            -> saves file, queues a job, returns job_id
  2. GET  /api/jobs/{job_id}     -> status: queued | processing | done | error
  3. GET  /api/jobs/{job_id}/stems/{stem}   -> download one stem (lossless WAV)
  4. GET  /api/jobs/{job_id}/download       -> download all stems as a zip

Requires: ffmpeg + ffprobe installed on the server.
Run on Windows: start_backend.bat
Alternative:    uvicorn main:app --host 127.0.0.1 --port 8000
"""
import asyncio
import json
import logging
import math
import re
import shutil
import subprocess
import sys
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# ----------------------------- settings -----------------------------
BASE = Path(__file__).resolve().parent / "storage"
BASE.mkdir(exist_ok=True)

MAX_FILE_MB = 200
MAX_DURATION_SEC = 20 * 60          # 20 minutes
RETENTION_SECONDS = 24 * 3600       # auto-delete after 24 hours
ALLOWED_EXT = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}

MODELS = {"4stem": "htdemucs", "6stem": "htdemucs_6s"}
STEMS = {
    "4stem": ["vocals", "drums", "bass", "other"],
    "6stem": ["vocals", "drums", "bass", "guitar", "piano", "other"],
}

JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")

# Only one heavy job at a time (CPU/GPU bound). Raise if you have a big GPU.
executor = ThreadPoolExecutor(max_workers=1)

app = FastAPI(title="Stem Splitter API")
logger = logging.getLogger(__name__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # TODO: restrict to your frontend domain
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "ffmpeg_available": shutil.which("ffmpeg") is not None,
        "ffprobe_available": shutil.which("ffprobe") is not None,
    }


# ----------------------------- helpers ------------------------------
def job_dir(job_id: str) -> Path:
    if not JOB_ID_RE.match(job_id):
        raise HTTPException(400, "Invalid job id")
    return BASE / job_id


def write_status(job_id: str, **fields) -> None:
    path = BASE / job_id / "status.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data.update(fields)
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(data))
    temporary_path.replace(path)


def read_status(job_id: str) -> dict:
    path = job_dir(job_id) / "status.json"
    if not path.exists():
        raise HTTPException(404, "Job not found (it may have expired)")
    return json.loads(path.read_text())


def stem_path(job_id: str, stem: str) -> Path:
    directory = job_dir(job_id) / "stems"
    for extension in (".wav", ".mp3"):
        path = directory / f"{stem}{extension}"
        if path.is_file():
            return path
    raise HTTPException(404, "Stem file not found (it may have expired)")


def audio_duration(path: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise HTTPException(503, "Audio inspection requires ffprobe to be installed and available on PATH")
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, check=True, timeout=30,
        )
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or "").strip().splitlines()
        logger.info("ffprobe rejected uploaded audio %s: %s", path.name, details[-1] if details else "unknown error")
        raise HTTPException(400, "Could not read audio file; ensure it is a valid audio file") from exc
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(400, "Audio inspection timed out; the file may be invalid") from exc
    except OSError as exc:
        raise HTTPException(503, "Could not start ffprobe; verify the FFmpeg installation") from exc
    try:
        duration = float(out.stdout.strip())
    except ValueError as exc:
        raise HTTPException(400, "Could not read audio file") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise HTTPException(400, "Could not read audio file")
    return duration


# --------------------------- background job -------------------------
def process_job(job_id: str, model_key: str) -> None:
    folder = BASE / job_id
    try:
        write_status(job_id, status="processing")
        src = next(folder.glob("input.*"))

        work = folder / "work"
        subprocess.run(
            [sys.executable, "-m", "demucs.separate",
             "-n", MODELS[model_key], "--int24", "-o", str(work), str(src)],
            check=True, capture_output=True, text=True,
        )

        # Demucs writes 24-bit PCM WAV files under work/<model>/<trackname>.
        result_dir = next((work / MODELS[model_key]).iterdir())
        stems_dir = folder / "stems"
        stems_dir.mkdir(exist_ok=True)
        produced_stems = {path.stem: path for path in result_dir.glob("*.wav")}
        missing_stems = set(STEMS[model_key]) - produced_stems.keys()
        if missing_stems:
            raise RuntimeError(f"Demucs did not produce expected stems: {', '.join(sorted(missing_stems))}")
        for stem, path in produced_stems.items():
            shutil.move(str(path), stems_dir / path.name)
        shutil.rmtree(work, ignore_errors=True)

        write_status(job_id, status="done", stems=STEMS[model_key])
    except subprocess.CalledProcessError as e:
        details = (e.stderr or e.stdout or "").strip()
        message = details[-1200:] if details else f"Demucs exited with code {e.returncode}"
        logger.error("Audio processing failed for job %s: %s", job_id, message)
        write_status(job_id, status="error", message=message)
    except Exception as e:  # noqa: BLE001
        logger.exception("Audio processing failed for job %s", job_id)
        write_status(job_id, status="error", message=str(e)[:300])


# ------------------------------ routes ------------------------------
@app.post("/api/upload")
async def upload(
    file: UploadFile = File(...),
    model: str = Form("4stem"),          # "4stem" or "6stem"
):
    if model not in MODELS:
        raise HTTPException(400, "model must be '4stem' or '6stem'")

    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file type. Use: {', '.join(sorted(ALLOWED_EXT))}")

    job_id = uuid.uuid4().hex            # unguessable id = private link
    folder = BASE / job_id
    folder.mkdir(parents=True)
    dest = folder / f"input{ext}"

    # stream to disk with a size limit
    size = 0
    try:
        with dest.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_FILE_MB * 1024 * 1024:
                    raise HTTPException(413, f"File too large (max {MAX_FILE_MB} MB)")
                out.write(chunk)

        if size == 0:
            raise HTTPException(400, "Uploaded file is empty")
        if audio_duration(dest) > MAX_DURATION_SEC:
            raise HTTPException(400, "Song too long (max 20 minutes)")
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise

    write_status(job_id, status="queued", created=time.time(), model=model)
    loop = asyncio.get_running_loop()
    loop.run_in_executor(executor, process_job, job_id, model)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    return read_status(job_id)


@app.get("/api/jobs/{job_id}/stems/{stem}")
def get_stem(job_id: str, stem: str):
    status = read_status(job_id)
    if status.get("status") != "done":
        raise HTTPException(409, "Job not finished")
    if stem not in status["stems"]:     # whitelist also blocks path traversal
        raise HTTPException(404, "Unknown stem")
    path = stem_path(job_id, stem)
    media_type = "audio/wav" if path.suffix == ".wav" else "audio/mpeg"
    return FileResponse(path, media_type=media_type, filename=path.name)


@app.get("/api/jobs/{job_id}/download")
def download_all(job_id: str):
    status = read_status(job_id)
    if status.get("status") != "done":
        raise HTTPException(409, "Job not finished")
    zip_path = job_dir(job_id) / "stems.zip"
    if not zip_path.exists():
        with zipfile.ZipFile(zip_path, "w") as z:
            for stem in status["stems"]:
                path = stem_path(job_id, stem)
                z.write(path, path.name)
    return FileResponse(zip_path, media_type="application/zip", filename="stems.zip")


# --------------------- auto-delete old files (privacy) ---------------------
async def cleanup_loop():
    while True:
        now = time.time()
        for folder in BASE.iterdir():
            if not folder.is_dir() or not JOB_ID_RE.fullmatch(folder.name):
                continue
            status_file = folder / "status.json"
            try:
                status = json.loads(status_file.read_text()) if status_file.exists() else {}
                created = float(status.get("created", folder.stat().st_mtime))
                if not math.isfinite(created):
                    created = folder.stat().st_mtime
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                try:
                    created = folder.stat().st_mtime
                except OSError:
                    logger.exception("Could not inspect stored job folder %s", folder)
                    continue
            if now - created > RETENTION_SECONDS:
                try:
                    shutil.rmtree(folder)
                except OSError:
                    logger.exception("Could not delete expired job folder %s", folder)
        await asyncio.sleep(60)


@app.on_event("startup")
async def start_cleanup():
    asyncio.create_task(cleanup_loop())
