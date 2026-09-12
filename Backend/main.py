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

import sys

try:
    from optimizer import CITIES, DEPOT, HOTSPOTS, build_cost_matrix, congestion_snapshot, generate_customers, get_city, solve
except ImportError:
    from Backend.optimizer import CITIES, DEPOT, HOTSPOTS, build_cost_matrix, congestion_snapshot, generate_customers, get_city, solve

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))



@asynccontextmanager
async def lifespan(app: FastAPI):
    print("OptiRoute API ready — multi-city graph-based QPSO CVRP solver")
    yield


app = FastAPI(title="OptiRoute — Quantum-Inspired Route Optimization", description="SIH26137 QPSO vehicle routing API with multi-city support", version="1.2.0", docs_url="/api/docs", redoc_url="/api/redoc", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])
static_dir = os.path.join(ROOT, "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


class Customer(BaseModel):
    id: int
    lat: float
    lng: float
    demand: int = Field(ge=1)


class OptimizeRequest(BaseModel):
    customers: List[Customer] = Field(min_length=1, max_length=60)
    capacity: int = Field(default=32, ge=1)
    algorithm: Literal["qpso", "pso", "nearest_neighbor", "exact"] = "qpso"
    iterations: int = Field(default=70, ge=5, le=400)
    particles: int = Field(default=24, ge=4, le=80)
    seed: Optional[int] = None
    congestion: Optional[List[dict]] = None
    city: Optional[str] = "delhi"


class BenchmarkRequest(BaseModel):
    customers: List[Customer] = Field(min_length=1, max_length=50)
    capacity: int = Field(default=32, ge=1)
    iterations: int = Field(default=70, ge=5, le=400)
    particles: int = Field(default=24, ge=4, le=80)
    seed: Optional[int] = 42
    congestion: Optional[List[dict]] = None
    include_exact: bool = False
    city: Optional[str] = "delhi"


@app.get("/api/health")
async def health():
    return {"ok": True, "service": "OptiRoute", "problem": "SIH26137", "cities": list(CITIES.keys())}


@app.get("/api/cities")
async def cities():
    return {
        "default": "delhi",
        "cities": [
            {
                "id": c["id"],
                "name": c["name"],
                "label": c["label"],
                "center": c["center"],
                "zoom": c["zoom"],
                "depot": c["depot"],
                "hotspots": c["hotspots"],
            }
            for c in CITIES.values()
        ],
    }


@app.get("/api/formulation")
async def formulation():
    return {
        "problem": "Capacitated Vehicle Routing Problem (CVRP) on a dynamic weighted road graph",
        "objective": "Minimize all-pairs Dijkstra shortest-path travel cost over all vehicle tours",
        "network": "A connected local road graph is generated from the depot and delivery locations; road-edge weights include simulated congestion.",
        "constraints": ["Every customer visited exactly once", "Each tour starts and ends at the depot", "Vehicle load never exceeds capacity Q"],
        "algorithm": "Quantum Particle Swarm Optimization (random-key encoding + capacity-feasible decode)",
        "baselines": ["Classical PSO", "Nearest-neighbour greedy", "Exact permutation search (n<=8)"],
    }


@app.get("/api/instance")
async def new_instance(n: int = 14, capacity: int = 32, seed: Optional[int] = None, city: str = "delhi"):
    n = max(4, min(40, n))
    city_data = get_city(city)
    customers = generate_customers(n, seed, city=city_data["id"])
    return {
        "city": city_data["id"],
        "cityName": city_data["name"],
        "depot": city_data["depot"],
        "customers": customers,
        "capacity": capacity,
        "hotspots": city_data["hotspots"],
        "congestion": congestion_snapshot(city=city_data["id"]),
        "note": f"Congestion and the road graph are simulated for {city_data['name']}, not sourced from a live traffic feed.",
    }


@app.get("/api/congestion")
async def congestion(city: str = "delhi"):
    city_data = get_city(city)
    return {
        "simulated": True,
        "city": city_data["id"],
        "zones": congestion_snapshot(city=city_data["id"]),
        "hotspots": city_data["hotspots"],
    }


@app.post("/api/optimize")
async def optimize(body: OptimizeRequest):
    customers = [customer.model_dump() for customer in body.customers]
    if not customers:
        raise HTTPException(400, "customers required")
    try:
        result = solve(
            customers,
            body.capacity,
            snapshot=body.congestion,
            algorithm=body.algorithm,
            iterations=body.iterations,
            particles=body.particles,
            seed=body.seed,
            city=body.city,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"success": True, **result}


@app.post("/api/benchmark")
async def benchmark(body: BenchmarkRequest):
    customers = [customer.model_dump() for customer in body.customers]
    if not customers:
        raise HTTPException(400, "customers required")
    city_data = get_city(body.city)
    snapshot = body.congestion or congestion_snapshot(city=city_data["id"])
    matrix = build_cost_matrix(customers, snapshot, depot=city_data["depot"], hotspots=city_data["hotspots"])
    common = dict(
        customers=customers,
        capacity=body.capacity,
        snapshot=snapshot,
        iterations=body.iterations,
        particles=body.particles,
        seed=body.seed,
        matrix=matrix,
        city=city_data["id"],
    )
    qpso, pso, nn = (
        solve(algorithm="qpso", **common),
        solve(algorithm="pso", **common),
        solve(algorithm="nearest_neighbor", **common),
    )
    payload = {
        "success": True,
        "city": city_data["id"],
        "cityName": city_data["name"],
        "qpso": qpso,
        "pso": pso,
        "nearest_neighbor": nn,
        "improvement_vs_nn_pct": round((nn["cost"] - qpso["cost"]) / nn["cost"] * 100, 2) if nn["cost"] else 0,
        "improvement_vs_pso_pct": round((pso["cost"] - qpso["cost"]) / pso["cost"] * 100, 2) if pso["cost"] else 0,
        "congestion": snapshot,
        "note": f"All algorithms use the same simulated traffic snapshot and Dijkstra road-distance matrix for {city_data['name']}.",
    }
    if body.include_exact and len(customers) <= 8:
        payload["exact"] = solve(algorithm="exact", **common)
    return payload


@app.get("/")
@app.get("/OptiRoute.html")
async def root():
    fleet = os.path.join(ROOT, "OptiRoute.html")
    index = os.path.join(ROOT, "index.html")
    if os.path.isfile(fleet):
        return FileResponse(fleet, media_type="text/html")
    if os.path.isfile(index):
        return FileResponse(index, media_type="text/html")
    return {"message": "OptiRoute API. Open /api/docs"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
