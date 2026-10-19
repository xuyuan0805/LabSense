from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .engine import ClassroomEngine


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "app" / "static"
engine = ClassroomEngine(ROOT / "data" / "cbc_teaching_raw.csv")
app = FastAPI(title="LabSense 地中海贫血提示模型教学版", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class DecisionRequest(BaseModel):
    issue_id: str
    action: str


class ThresholdRequest(BaseModel):
    threshold: float = Field(default=0.5, ge=0.05, le=0.95)


class PatientRequest(BaseModel):
    patient_id: str


def call(method, *args, **kwargs):
    try:
        return method(*args, **kwargs)
    except (ValueError, RuntimeError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": "local", "network_required": "false"}


@app.get("/api/session")
def session() -> dict[str, Any]:
    return engine.state()


@app.post("/api/session/reset")
def reset() -> dict[str, Any]:
    return engine.reset()


@app.get("/api/dataset/raw")
def raw(page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100)) -> dict[str, Any]:
    return engine.raw_page(page, page_size)


@app.post("/api/checks/run")
def checks() -> dict[str, Any]:
    return call(engine.run_checks)


@app.post("/api/decisions")
def decisions(request: DecisionRequest) -> dict[str, Any]:
    return call(engine.answer_decision, request.issue_id, request.action)


@app.get("/api/features/preview")
def features() -> dict[str, Any]:
    return call(engine.feature_preview)


@app.post("/api/imputation/start")
def imputation() -> dict[str, Any]:
    return call(engine.start_job, "imputation")


@app.post("/api/models/feature-selection/start")
def selection() -> dict[str, Any]:
    return call(engine.start_job, "selection")


@app.post("/api/models/compare/start")
def comparison() -> dict[str, Any]:
    return call(engine.start_job, "compare")


@app.get("/api/jobs/{job_id}")
def job(job_id: str) -> dict[str, Any]:
    return call(engine.job_status, job_id)


@app.get("/api/models/explanation/global")
def global_explanation() -> dict[str, Any]:
    return call(engine.global_explanation)


@app.post("/api/models/explanation/patient")
def patient_explanation(request: PatientRequest) -> dict[str, Any]:
    return call(engine.patient_explanation, request.patient_id)


@app.post("/api/models/threshold")
def threshold(request: ThresholdRequest) -> dict[str, Any]:
    return call(engine.evaluate_threshold, request.threshold)


@app.post("/api/models/unsupervised/run")
def unsupervised() -> dict[str, Any]:
    return call(engine.run_unsupervised)
