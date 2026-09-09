"""OptiRoute API — QPSO vehicle routing for SIH26137."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from optimizer import DEPOT, HOTSPOTS, build_cost_matrix, congestion_snapshot, generate_customers, solve

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("OptiRoute API ready — graph-based QPSO CVRP solver")
    yield


app = FastAPI(title="OptiRoute — Quantum-Inspired Route Optimization", description="SIH26137 QPSO vehicle routing API", version="1.1.0", docs_url="/api/docs", redoc_url="/api/redoc", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])
static_dir = os.path.join(ROOT, "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


class Customer(BaseModel):
    id: int
    lat: float
    lng: float
    demand: int = Field(ge=1)


# PHASE 2 CHANGE: bound list sizes so iterations * particles * n can't be
# driven arbitrarily high by a careless or hostile payload.
class OptimizeRequest(BaseModel):
    customers: List[Customer] = Field(min_length=1, max_length=60)
    capacity: int = Field(default=32, ge=1)
    algorithm: Literal["qpso", "pso", "nearest_neighbor", "exact"] = "qpso"
    iterations: int = Field(default=70, ge=5, le=400)
    particles: int = Field(default=24, ge=4, le=80)
    seed: Optional[int] = None
    congestion: Optional[List[dict]] = None


class BenchmarkRequest(BaseModel):
    customers: List[Customer] = Field(min_length=1, max_length=50)
    capacity: int = Field(default=32, ge=1)
    iterations: int = Field(default=70, ge=5, le=400)
    particles: int = Field(default=24, ge=4, le=80)
    seed: Optional[int] = 42
    congestion: Optional[List[dict]] = None
    include_exact: bool = False


@app.get("/api/health")
async def health():
    return {"ok": True, "service": "OptiRoute", "problem": "SIH26137"}


@app.get("/api/formulation")
async def formulation():
    # PHASE 1 CHANGE: Document the graph-based cost model exposed by the API.
    return {
        "problem": "Capacitated Vehicle Routing Problem (CVRP) on a dynamic weighted road graph",
        "objective": "Minimize all-pairs Dijkstra shortest-path travel cost over all vehicle tours",
        "network": "A connected local road graph is generated from the depot and delivery locations; road-edge weights include simulated congestion.",
        "constraints": ["Every customer visited exactly once", "Each tour starts and ends at the depot", "Vehicle load never exceeds capacity Q"],
        "algorithm": "Quantum Particle Swarm Optimization (random-key encoding + capacity-feasible decode)",
        "baselines": ["Classical PSO", "Nearest-neighbour greedy", "Exact permutation search (n<=8)"],
    }


@app.get("/api/instance")
async def new_instance(n: int = 14, capacity: int = 32, seed: Optional[int] = None):
    n = max(4, min(40, n))
    customers = generate_customers(n, seed)
    return {"depot": DEPOT, "customers": customers, "capacity": capacity, "hotspots": HOTSPOTS, "congestion": congestion_snapshot(), "note": "Congestion and the road graph are simulated for this instance, not sourced from a live traffic feed."}


@app.get("/api/congestion")
async def congestion():
    return {"simulated": True, "zones": congestion_snapshot(), "hotspots": HOTSPOTS}


@app.post("/api/optimize")
async def optimize(body: OptimizeRequest):
    customers = [customer.model_dump() for customer in body.customers]
    if not customers:
        raise HTTPException(400, "customers required")
    try:
        result = solve(customers, body.capacity, snapshot=body.congestion, algorithm=body.algorithm, iterations=body.iterations, particles=body.particles, seed=body.seed)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"success": True, **result}


@app.post("/api/benchmark")
async def benchmark(body: BenchmarkRequest):
    customers = [customer.model_dump() for customer in body.customers]
    if not customers:
        raise HTTPException(400, "customers required")
    snapshot = body.congestion or congestion_snapshot()
    # PHASE 2 CHANGE: build the road graph + all-pairs Dijkstra matrix once and
    # reuse it across all algorithm variants below, instead of solve() rebuilding
    # it from scratch on each of the (up to 4) calls for identical customers+snapshot.
    matrix = build_cost_matrix(customers, snapshot)
    common = dict(customers=customers, capacity=body.capacity, snapshot=snapshot, iterations=body.iterations, particles=body.particles, seed=body.seed, matrix=matrix)
    qpso, pso, nn = solve(algorithm="qpso", **common), solve(algorithm="pso", **common), solve(algorithm="nearest_neighbor", **common)
    payload = {"success": True, "qpso": qpso, "pso": pso, "nearest_neighbor": nn, "improvement_vs_nn_pct": round((nn["cost"] - qpso["cost"]) / nn["cost"] * 100, 2) if nn["cost"] else 0, "improvement_vs_pso_pct": round((pso["cost"] - qpso["cost"]) / pso["cost"] * 100, 2) if pso["cost"] else 0, "congestion": snapshot, "note": "All algorithms use the same simulated traffic snapshot and Dijkstra road-distance matrix."}
    if body.include_exact and len(customers) <= 8:
        payload["exact"] = solve(algorithm="exact", **common)
    return payload


@app.get("/")
async def root():
    fleet, index = os.path.join(ROOT, "OptiRoute.html"), os.path.join(ROOT, "index.html")
    if os.path.isfile(fleet):
        return FileResponse(fleet)
    if os.path.isfile(index):
        return FileResponse(index)
    return {"message": "FleetPath API. Open /api/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
