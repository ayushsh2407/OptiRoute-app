"""Graph-based CVRP decoder plus QPSO, PSO, greedy, and exact solvers."""

from __future__ import annotations

import heapq
import math
import random
import time
from typing import Dict, Hashable, List, Mapping, Optional, Sequence, Tuple


CITIES: Dict[str, Dict] = {
    "delhi": {
        "id": "delhi",
        "name": "Delhi NCR",
        "label": "Delhi Metro",
        "center": [28.6315, 77.2167],
        "zoom": 12,
        "depot": {"id": 0, "lat": 28.6315, "lng": 77.2167, "name": "Central Depot — Connaught Place, Delhi"},
        "hotspots": [
            {"name": "Downtown Core", "lat": 28.6280, "lng": 77.2200, "radius": 0.020, "phase": 0.0},
            {"name": "Airport Junction", "lat": 28.6450, "lng": 77.1900, "radius": 0.018, "phase": 1.4},
            {"name": "University District", "lat": 28.6150, "lng": 77.2350, "radius": 0.016, "phase": 2.8},
        ],
        "lat_range": 0.045,
        "lng_range": 0.055,
    },
    "mumbai": {
        "id": "mumbai",
        "name": "Mumbai Metropolitan",
        "label": "Mumbai Region",
        "center": [19.0662, 72.8687],
        "zoom": 12,
        "depot": {"id": 0, "lat": 19.0662, "lng": 72.8687, "name": "Central Depot — BKC (Bandra Kurla Complex), Mumbai"},
        "hotspots": [
            {"name": "Lower Parel Hub", "lat": 18.9950, "lng": 72.8300, "radius": 0.022, "phase": 0.0},
            {"name": "Western Express / Andheri", "lat": 19.1190, "lng": 72.8465, "radius": 0.025, "phase": 1.2},
            {"name": "Navi Mumbai Vashi Corridor", "lat": 19.0770, "lng": 72.9980, "radius": 0.025, "phase": 2.6},
        ],
        "lat_range": 0.050,
        "lng_range": 0.045,
    },
    "kolkata": {
        "id": "kolkata",
        "name": "Kolkata Metro",
        "label": "Kolkata Region",
        "center": [22.5726, 88.3639],
        "zoom": 12,
        "depot": {"id": 0, "lat": 22.5726, "lng": 88.3639, "name": "Central Depot — BBD Bagh, Kolkata"},
        "hotspots": [
            {"name": "Salt Lake Sector V", "lat": 22.5800, "lng": 88.4350, "radius": 0.022, "phase": 0.0},
            {"name": "Howrah Interchange", "lat": 22.5850, "lng": 88.3470, "radius": 0.018, "phase": 1.5},
            {"name": "South Kolkata Gariahat", "lat": 22.5180, "lng": 88.3650, "radius": 0.020, "phase": 2.7},
        ],
        "lat_range": 0.045,
        "lng_range": 0.050,
    },
}

DEFAULT_CITY = "delhi"
DEPOT = CITIES[DEFAULT_CITY]["depot"]
HOTSPOTS = CITIES[DEFAULT_CITY]["hotspots"]


def get_city(city_id: Optional[str] = None) -> Dict:
    """Retrieve city parameters with fallback to Delhi."""
    if not city_id:
        return CITIES[DEFAULT_CITY]
    return CITIES.get(str(city_id).lower().strip(), CITIES[DEFAULT_CITY])


def auto_detect_city(customers: Sequence[Dict]) -> str:
    """Detect city based on centroid of customer coordinates."""
    if not customers:
        return DEFAULT_CITY
    avg_lat = sum(c["lat"] for c in customers) / len(customers)
    avg_lng = sum(c["lng"] for c in customers) / len(customers)
    best_city = DEFAULT_CITY
    best_dist = math.inf
    for cid, cdata in CITIES.items():
        d = math.hypot(avg_lat - cdata["center"][0], avg_lng - cdata["center"][1])
        if d < best_dist:
            best_dist = d
            best_city = cid
    return best_city


# PHASE 1 CHANGE: This helper is used only to assign lengths while constructing
# a simulated road network. It is no longer used as the route/edge cost.
def _geographic_distance_km(a: Mapping, b: Mapping) -> float:
    """Return a local geographic estimate used to create road-edge weights."""
    mean_lat = math.radians((a["lat"] + b["lat"]) / 2)
    lat_km = (a["lat"] - b["lat"]) * 110.574
    lng_km = (a["lng"] - b["lng"]) * 111.320 * math.cos(mean_lat)
    return math.hypot(lat_km, lng_km)


def congestion_snapshot(t: Optional[float] = None, city: Optional[str] = None, hotspots: Optional[Sequence[Dict]] = None) -> List[Dict]:
    now = time.time() if t is None else t
    target_hotspots = hotspots if hotspots is not None else get_city(city)["hotspots"]
    return [
        {
            "name": h["name"],
            "intensity": 40 + 45 * abs(math.sin(now * 0.15 + h["phase"])),
        }
        for h in target_hotspots
    ]


def generate_customers(n: int, seed: Optional[int] = None, city: Optional[str] = None) -> List[Dict]:
    rng = random.Random(seed)
    city_data = get_city(city)
    depot = city_data["depot"]
    lat_r = city_data["lat_range"]
    lng_r = city_data["lng_range"]
    return [
        {
            "id": i + 1,
            "lat": depot["lat"] + rng.uniform(-lat_r, lat_r),
            "lng": depot["lng"] + rng.uniform(-lng_r, lng_r),
            "demand": rng.randint(4, 12),
        }
        for i in range(n)
    ]


# PHASE 1 CHANGE: Namespaced graph node keys prevent a customer with id 0
# from colliding with the depot, while API routes can still use the original ids.
DEPOT_KEY = "__depot__"


def _customer_key(customer_id: int) -> str:
    return f"customer:{customer_id}"


def node_map(customers: Sequence[Dict], depot: Optional[Dict] = None) -> Dict[Hashable, Dict]:
    nodes: Dict[Hashable, Dict] = {DEPOT_KEY: depot if depot is not None else DEPOT}
    for customer in customers:
        nodes[_customer_key(customer["id"])] = customer
    return nodes


def _hotspot_penalty(a: Mapping, b: Mapping, snapshot: Sequence[Dict], gamma: float, hotspots: Optional[Sequence[Dict]] = None) -> float:
    """Traffic multiplier for one road segment, based on its midpoint."""
    mid_lat = (a["lat"] + b["lat"]) / 2
    mid_lng = (a["lng"] + b["lng"]) / 2
    penalty = 0.0
    target_hotspots = hotspots if hotspots is not None else HOTSPOTS
    for hotspot, state in zip(target_hotspots, snapshot):
        distance = math.hypot(mid_lat - hotspot["lat"], mid_lng - hotspot["lng"])
        if distance < hotspot["radius"]:
            penalty = max(
                penalty,
                state["intensity"] / 100.0 * (1.0 - distance / hotspot["radius"]),
            )
    return 1.0 + gamma * penalty


# PHASE 1 CHANGE: Build a connected, weighted road graph from the depot and
# delivery locations. An MST guarantees connectivity; nearby extra links model
# alternative roads. The 1.15 factor approximates road distance versus a direct line.
def build_road_graph(
    customers: Sequence[Dict],
    snapshot: Sequence[Dict],
    gamma: float = 0.85,
    depot: Optional[Dict] = None,
    hotspots: Optional[Sequence[Dict]] = None,
) -> Dict[Hashable, List[Tuple[Hashable, float]]]:
    nodes = node_map(customers, depot=depot)
    keys = list(nodes)
    graph: Dict[Hashable, List[Tuple[Hashable, float]]] = {key: [] for key in keys}
    if len(keys) < 2:
        return graph

    distances = {
        (left, right): _geographic_distance_km(nodes[left], nodes[right])
        for index, left in enumerate(keys)
        for right in keys[index + 1 :]
    }

    def base_distance(left: Hashable, right: Hashable) -> float:
        direct = distances.get((left, right))
        return direct if direct is not None else distances[(right, left)]

    edges = set()
    connected = {DEPOT_KEY}
    # Prim's algorithm: at least one road path always exists between every pair.
    while len(connected) < len(keys):
        left, right = min(
            (
                (source, target)
                for source in connected
                for target in keys
                if target not in connected
            ),
            key=lambda pair: base_distance(*pair),
        )
        edges.add(frozenset((left, right)))
        connected.add(right)

    neighbour_count = min(3, len(keys) - 1)
    for source in keys:
        nearby = sorted(
            (target for target in keys if target != source), key=lambda target: base_distance(source, target)
        )[:neighbour_count]
        edges.update(frozenset((source, target)) for target in nearby)

    for edge in edges:
        left, right = tuple(edge)
        road_length = base_distance(left, right) * 1.15
        traffic_weight = _hotspot_penalty(nodes[left], nodes[right], snapshot, gamma, hotspots=hotspots)
        weight = road_length * traffic_weight
        graph[left].append((right, weight))
        graph[right].append((left, weight))
    return graph


# PHASE 1 CHANGE: Dijkstra finds shortest travel cost through the weighted road graph.
def dijkstra(graph: Mapping[Hashable, Sequence[Tuple[Hashable, float]]], start: Hashable) -> Dict[Hashable, float]:
    distances = {node: math.inf for node in graph}
    distances[start] = 0.0
    # The sequence number also makes heap entries safe when two costs match.
    queue: List[Tuple[float, int, Hashable]] = [(0.0, 0, start)]
    sequence = 1
    while queue:
        current_cost, _, node = heapq.heappop(queue)
        if current_cost != distances[node]:
            continue
        for neighbour, weight in graph[node]:
            candidate = current_cost + weight
            if candidate < distances[neighbour]:
                distances[neighbour] = candidate
                heapq.heappush(queue, (candidate, sequence, neighbour))
                sequence += 1
    return distances


# PHASE 1 CHANGE: All-pairs values are precomputed once per optimization run.
def build_distance_matrix(
    graph: Mapping[Hashable, Sequence[Tuple[Hashable, float]]]
) -> Dict[Hashable, Dict[Hashable, float]]:
    return {node: dijkstra(graph, node) for node in graph}


def build_cost_matrix(
    customers: Sequence[Dict],
    snapshot: Sequence[Dict],
    depot: Optional[Dict] = None,
    hotspots: Optional[Sequence[Dict]] = None,
) -> Dict[Hashable, Dict[Hashable, float]]:
    """Public entry point so callers (e.g. /api/benchmark) can build the
    road-graph distance matrix once and reuse it across several solve() calls
    instead of rebuilding the graph + running Dijkstra from every node per algorithm."""
    return build_distance_matrix(build_road_graph(customers, snapshot, depot=depot, hotspots=hotspots))


# PHASE 2 CHANGE: The frontend's own congestion snapshot (produced by
# takeCongestionSnapshot() in fleetpath.html) has a different shape —
# {lat, lng, radiusKm, intensity} — than what this module expects internally
# — {name, intensity}, zipped positionally against HOTSPOTS. Normalize here
# instead of trusting the caller, so a malformed/foreign-shaped snapshot falls
# back to a freshly generated one rather than silently mis-pairing zones.
def normalize_congestion(snapshot: Optional[Sequence[dict]], hotspots: Optional[Sequence[Dict]] = None) -> List[Dict]:
    target_hotspots = hotspots if hotspots is not None else HOTSPOTS
    if not snapshot or len(snapshot) != len(target_hotspots):
        return congestion_snapshot(hotspots=target_hotspots)
    normalized = []
    for hotspot, entry in zip(target_hotspots, snapshot):
        intensity = entry.get("intensity")
        if not isinstance(intensity, (int, float)):
            return congestion_snapshot(hotspots=target_hotspots)
        normalized.append({"name": hotspot["name"], "intensity": intensity})
    return normalized


def _edge_cost(matrix: Mapping[Hashable, Mapping[Hashable, float]], source: Hashable, target: Hashable) -> float:
    """Look up the graph shortest-path cost, failing safely for an unreachable node."""
    return matrix.get(source, {}).get(target, math.inf)


# PHASE 2 CHANGE: This used to be a greedy "cut a new route the moment
# capacity overflows" split. That is provably suboptimal for a random-key
# giant tour: the frontend's JS decoder (splitTour) already used the correct
# approach — Prins' split algorithm, a DP over the fixed visiting order that
# finds the capacity-feasible partition minimizing total route cost. Python
# now runs the same DP so both engines score a given permutation identically
# (mirrors splitTour() in fleetpath.html; keep the two in sync if either changes).
def decode_random_key(
    keys: Sequence[float],
    customers: Sequence[Dict],
    capacity: int,
    matrix: Mapping[Hashable, Mapping[Hashable, float]],
    depot: Optional[Dict] = None,
) -> Tuple[List[List[int]], float]:
    order = [customer["id"] for _, customer in sorted(zip(keys, customers), key=lambda pair: pair[0])]
    nodes = node_map(customers, depot=depot)
    n = len(order)

    # V[j] = minimum cost to serve the first j customers in `order` using
    # capacity-feasible routes; P[j] = the split point that achieves it.
    best_cost = [math.inf] * (n + 1)
    split_at = [-1] * (n + 1)
    best_cost[0] = 0.0

    for i in range(1, n + 1):
        load = 0
        segment_dist = 0.0
        first_key = _customer_key(order[i - 1])
        for j in range(i, n + 1):
            customer_j = nodes[_customer_key(order[j - 1])]
            load += customer_j["demand"]
            if load > capacity:
                break
            if j > i:
                prev_key = _customer_key(order[j - 2])
                segment_dist += _edge_cost(matrix, prev_key, _customer_key(order[j - 1]))
            last_key = _customer_key(order[j - 1])
            route_cost = _edge_cost(matrix, DEPOT_KEY, first_key) + segment_dist + _edge_cost(matrix, last_key, DEPOT_KEY)
            candidate = best_cost[i - 1] + route_cost
            if candidate < best_cost[j]:
                best_cost[j] = candidate
                split_at[j] = i - 1

    routes: List[List[int]] = []
    j = n
    while j > 0:
        i = split_at[j]
        routes.insert(0, order[i:j])
        j = i
    return routes, best_cost[n]


def nearest_neighbor(
    customers: Sequence[Dict],
    capacity: int,
    matrix: Mapping[Hashable, Mapping[Hashable, float]],
    depot: Optional[Dict] = None,
) -> Tuple[List[List[int]], float]:
    nodes = node_map(customers, depot=depot)
    unvisited = {customer["id"] for customer in customers}
    routes: List[List[int]] = []
    total = 0.0
    while unvisited:
        route: List[int] = []
        load = 0
        current = DEPOT_KEY
        while True:
            candidates = [
                customer_id
                for customer_id in unvisited
                if load + nodes[_customer_key(customer_id)]["demand"] <= capacity
            ]
            if not candidates:
                break
            best_id = min(candidates, key=lambda cid: _edge_cost(matrix, current, _customer_key(cid)))
            total += _edge_cost(matrix, current, _customer_key(best_id))
            route.append(best_id)
            load += nodes[_customer_key(best_id)]["demand"]
            current = _customer_key(best_id)
            unvisited.remove(best_id)
        if not route:
            raise ValueError("A customer demand exceeds the vehicle capacity")
        total += _edge_cost(matrix, current, DEPOT_KEY)
        routes.append(route)
    return routes, total


def compute_routes_cost(
    routes: Sequence[Sequence[int]],
    matrix: Mapping[Hashable, Mapping[Hashable, float]],
) -> float:
    """Calculate the total Dijkstra road graph cost of a given list of vehicle routes."""
    total = 0.0
    for route in routes:
        if not route:
            continue
        first_key = _customer_key(route[0])
        total += _edge_cost(matrix, DEPOT_KEY, first_key)
        for i in range(len(route) - 1):
            total += _edge_cost(matrix, _customer_key(route[i]), _customer_key(route[i + 1]))
        last_key = _customer_key(route[-1])
        total += _edge_cost(matrix, last_key, DEPOT_KEY)
    return total


def _run_swarm(
    mode: str, customers: Sequence[Dict], capacity: int, matrix: Mapping[Hashable, Mapping[Hashable, float]],
    iterations: int, particles: int, seed: Optional[int] = 42, depot: Optional[Dict] = None
) -> Dict:
    effective_seed = 42 if seed is None else seed
    rng = random.Random(effective_seed)
    n = len(customers)
    swarm = [{"x": [rng.random() for _ in range(n)], "v": [(rng.random() - 0.5) * 0.2 for _ in range(n)], "pbest": None, "pbest_cost": math.inf} for _ in range(particles)]
    gbest = None
    gbest_cost = math.inf
    history: List[float] = []
    for iteration in range(iterations):
        for particle in swarm:
            _, cost = decode_random_key(particle["x"], customers, capacity, matrix, depot=depot)
            if cost < particle["pbest_cost"]:
                particle["pbest_cost"], particle["pbest"] = cost, particle["x"][:]
            if cost < gbest_cost:
                gbest_cost, gbest = cost, particle["x"][:]
        history.append(round(gbest_cost, 4))
        if mode == "qpso":
            mbest = [sum(particle["pbest"][dimension] for particle in swarm) / particles for dimension in range(n)]
            beta = 1.0 - 0.5 * (iteration / max(1, iterations))
            for particle in swarm:
                for dimension in range(n):
                    phi = rng.random()
                    attractor = phi * particle["pbest"][dimension] + (1 - phi) * gbest[dimension]
                    u = max(1e-6, rng.random())
                    sign = 1 if rng.random() > 0.5 else -1
                    value = attractor + sign * beta * abs(mbest[dimension] - particle["x"][dimension]) * math.log(1 / u)
                    particle["x"][dimension] = min(1.0, max(0.0, value))
        else:
            for particle in swarm:
                for dimension in range(n):
                    r1, r2 = rng.random(), rng.random()
                    velocity = 0.7 * particle["v"][dimension] + 1.5 * r1 * (particle["pbest"][dimension] - particle["x"][dimension]) + 1.5 * r2 * (gbest[dimension] - particle["x"][dimension])
                    particle["v"][dimension] = max(-0.25, min(0.25, velocity))
                    particle["x"][dimension] = min(1.0, max(0.0, particle["x"][dimension] + particle["v"][dimension]))
    routes, cost = decode_random_key(gbest, customers, capacity, matrix, depot=depot)
    return {"routes": routes, "cost": cost, "history": history}


def exact_small(
    customers: Sequence[Dict],
    capacity: int,
    matrix: Mapping[Hashable, Mapping[Hashable, float]],
    max_n: int = 8,
    depot: Optional[Dict] = None,
) -> Optional[Dict]:
    """Brute-force visit order, then capacity split. Only for tiny instances."""
    if len(customers) > max_n:
        return None
    from itertools import permutations
    ids = [customer["id"] for customer in customers]
    id_to_index = {customer["id"]: index for index, customer in enumerate(customers)}
    best_routes, best_cost = None, math.inf
    for permutation in permutations(ids):
        keys = [0.0] * len(customers)
        for rank, customer_id in enumerate(permutation):
            keys[id_to_index[customer_id]] = rank / max(1, len(customers))
        routes, cost = decode_random_key(keys, customers, capacity, matrix, depot=depot)
        if cost < best_cost:
            best_routes, best_cost = routes, cost
    return {"routes": best_routes, "cost": best_cost}


def solve(
    customers: Sequence[Dict],
    capacity: int,
    snapshot: Optional[Sequence[Dict]] = None,
    algorithm: str = "qpso",
    iterations: int = 70,
    particles: int = 24,
    seed: Optional[int] = 42,
    matrix: Optional[Mapping[Hashable, Mapping[Hashable, float]]] = None,
    city: Optional[str] = None,
) -> Dict:
    if len({customer["id"] for customer in customers}) != len(customers):
        raise ValueError("Customer ids must be unique")
    if any(customer["demand"] > capacity for customer in customers):
        raise ValueError("A customer demand exceeds the vehicle capacity")

    # Resolve target city, its central depot and hotspots
    detected_city = city or auto_detect_city(customers)
    city_data = get_city(detected_city)
    city_depot = city_data["depot"]
    city_hotspots = city_data["hotspots"]

    snap = normalize_congestion(snapshot, hotspots=city_hotspots)
    if matrix is None:
        matrix = build_cost_matrix(customers, snap, depot=city_depot, hotspots=city_hotspots)
    t0 = time.perf_counter()
    effective_seed = 42 if seed is None else seed
    if algorithm == "nearest_neighbor":
        routes, cost = nearest_neighbor(customers, capacity, matrix, depot=city_depot)
        result = {"routes": routes, "cost": cost, "history": [cost]}
    elif algorithm == "pso":
        result = _run_swarm("pso", customers, capacity, matrix, iterations, particles, effective_seed, depot=city_depot)
    elif algorithm == "exact":
        exact = exact_small(customers, capacity, matrix, depot=city_depot)
        if exact is None:
            raise ValueError("Exact solver only supports n <= 8 customers")
        result = {**exact, "history": [exact["cost"]]}
    else:
        result = _run_swarm("qpso", customers, capacity, matrix, iterations, particles, effective_seed, depot=city_depot)
    result["runtime_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    result["algorithm"] = algorithm
    result["vehicles"] = len(result["routes"])
    result["city"] = city_data["id"]
    result["city_name"] = city_data["name"]
    result["cityName"] = city_data["name"]
    result["depot"] = city_depot
    result["network_model"] = f"dynamic weighted road graph with Dijkstra shortest paths ({city_data['name']})"
    return result
