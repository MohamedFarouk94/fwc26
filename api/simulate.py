import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from mangum import Mangum
from simulation import SM, wc26_builder

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class SimParams(BaseModel):
    rho: float = Field(default=0.5, ge=0.0, le=1.0)
    mu:  float = Field(default=0.5, ge=0.0, le=1.0)

@app.post("/api/simulate")
def simulate(params: SimParams):
    try:
        sm = SM(n=1, year=2026, wc_builder=wc26_builder, rho=params.rho, mu=params.mu)
        sm.run()
        return sm.trs_[0].wc.get_plot()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/api/health")
def health():
    return {"status": "ok"}

handler = Mangum(app, lifespan="off")