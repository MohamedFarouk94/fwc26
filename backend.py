"""
WC 2026 Simulator – FastAPI backend
Run locally with:
    uvicorn backend:app --reload --port 8000
"""

import matplotlib
matplotlib.use('Agg')

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from simulation import SM, wc26_builder

# ──────────────────────────────────────────────────────────────────
app = FastAPI(title="WC 2026 Simulator", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────
class SimParams(BaseModel):
    rho: float = Field(default=0.5, ge=0.0, le=1.0,
                       description="Form Change Factor (0–1)")
    mu:  float = Field(default=0.5, ge=0.0, le=1.0,
                       description="Shock Factor (0–1)")


# ──────────────────────────────────────────────────────────────────
@app.post("/simulate") 
@app.post("/api/simulate")
def simulate(params: SimParams):
    """Run one simulation and return the group-stage image + bracket HTML."""
    try:
        sm = SM(n=1, year=2026, wc_builder=wc26_builder,
                rho=params.rho, mu=params.mu)
        sm.run()
        wc = sm.trs_[0].wc
        return wc.get_plot()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/health") 
@app.get("/api/health")
def health():
    return {"status": "ok"}


# ── Serve static HTML pages ────────────────────────────────────────
@app.get("/")
def index():
    return FileResponse("index.html")

@app.get("/simulate")
def simulate_page():
    return FileResponse("simulate.html")

# Serve flag images and other static assets
app.mount("/data", StaticFiles(directory="data"), name="data")
