
import matplotlib
matplotlib.use('Agg')   # must come before any pyplot import

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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


@app.get("/health")
def health():
    return {"status": "ok"}
