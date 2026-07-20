"""Web UI for the VN scene generator.

A single background worker processes one generation job at a time (the
pipeline is GPU-heavy — SDXL + MusicGen + TTS all on one pinned GPU — so we
deliberately don't run jobs concurrently). run_scene.py is invoked as a
subprocess exactly as from the CLI; the server just queues, tracks, and
serves the result.
"""
import asyncio
import random
import string
import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vn_creator import config  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"
WEB_OUTPUTS_DIR = config.DATA_ROOT / "outputs" / "web"
WEB_LOGS_DIR = config.TMP_DIR / "web_jobs"
WEB_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
WEB_LOGS_DIR.mkdir(parents=True, exist_ok=True)

RUN_SCENE_SCRIPT = REPO_ROOT / "scripts" / "run_scene.py"
VENV_PYTHON = Path("/data/shasegawa/vn/venv/bin/python")


def _load_dotenv() -> None:
    import os

    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()

app = FastAPI(title="VN Scene Generator")

jobs: dict[str, dict] = {}
job_queue: "asyncio.Queue[str]" = asyncio.Queue()


class GenerateRequest(BaseModel):
    character_name: str
    character_description: str
    background_description: str
    persona: str
    context: str
    genre: str
    retro: bool = False


def _random_name(prefix: str) -> str:
    suffix = "".join(random.choices(string.digits, k=6))
    return f"{prefix}_{suffix}"


async def _worker() -> None:
    while True:
        job_id = await job_queue.get()
        job = jobs[job_id]
        job["status"] = "running"
        try:
            out_path = WEB_OUTPUTS_DIR / f"{job_id}.mp4"
            log_path = WEB_LOGS_DIR / f"{job_id}.log"
            cmd = [
                str(VENV_PYTHON), str(RUN_SCENE_SCRIPT),
                "--character-name", job["params"]["character_name"],
                "--character-description", job["params"]["character_description"],
                "--background-description", job["params"]["background_description"],
                "--persona", job["params"]["persona"],
                "--context", job["params"]["context"],
                "--genre", job["params"]["genre"],
                "--character-seed", str(random.randint(0, 2**31 - 1)),
                "--out", str(out_path),
            ]
            if job["params"]["retro"]:
                cmd += ["--force-shot-type", "illustration", "--retro-style"]

            with open(log_path, "w") as log_f:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=log_f, stderr=asyncio.subprocess.STDOUT,
                    cwd=str(REPO_ROOT),
                )
                job["pid"] = proc.pid
                returncode = await proc.wait()

            if returncode == 0 and out_path.exists():
                job["status"] = "done"
                job["video_url"] = f"/api/video/{job_id}"
            else:
                job["status"] = "error"
                tail = log_path.read_text()[-2000:] if log_path.exists() else ""
                job["error"] = f"generation failed (exit {returncode})\n{tail}"
        except Exception as e:  # noqa: BLE001
            job["status"] = "error"
            job["error"] = str(e)
        finally:
            job_queue.task_done()


@app.on_event("startup")
async def _start_worker():
    asyncio.create_task(_worker())


@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/retro", response_class=HTMLResponse)
async def retro_page():
    return (STATIC_DIR / "retro.html").read_text(encoding="utf-8")


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "status": "queued",
        "params": req.model_dump(),
        "queue_position": job_queue.qsize(),
    }
    await job_queue.put(job_id)
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
async def status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job_id")
    return {
        "status": job["status"],
        "video_url": job.get("video_url"),
        "error": job.get("error"),
        "queue_position": job_queue.qsize() if job["status"] == "queued" else 0,
    }


@app.get("/api/video/{job_id}")
async def video(job_id: str):
    job = jobs.get(job_id)
    if job is None or job.get("status") != "done":
        raise HTTPException(404, "video not ready")
    return FileResponse(WEB_OUTPUTS_DIR / f"{job_id}.mp4", media_type="video/mp4")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
