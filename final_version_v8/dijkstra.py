"""
dijkstra.py — Mode-aware Dijkstra over the Road+Bus intermodal graph
=====================================================================
Four semiring weight functions, now multimodal:
    Each function checks edge['mode'] in {road, bus, transfer}
    and applies the appropriate cost model.

CT Framing (ACT4E §14.3):
    The intermodal graph G is a category enriched over the
    tropical semiring (R, min, +). Each weight function
    defines a different enrichment of the same graph.
    Same Dijkstra algorithm; four semirings.
    Proved correct once; works across all enrichments for free.
    (ACT4E §14.6: CT avoids redundant re-implementation)
"""

import heapq
import math
import time

from semirings import SafetySemiring, LexicographicSemiring, TransferSemiring

# Road speed by highway type (km/h)
ROAD_SPEED_KMH = {
    "motorway": 80,       "motorway_link": 60,
    "trunk": 65,          "trunk_link": 50,
    "primary": 50,        "primary_link": 40,
    "secondary": 40,      "secondary_link": 35,
    "tertiary": 35,       "tertiary_link": 30,
    "residential": 25,    "living_street": 15,
    "unclassified": 30,   "service": 20,
}
ROAD_DEFAULT_SPEED = 25

BUS_SPEED_KMH  = 18.0
BUS_WAIT_SEC   = 600.0   # 10-minute average bus wait
WALK_SPEED_KMH = 5.0

ROAD_FARE_PER_M  = 15.0  / 1000   # Rs 15/km auto
BUS_FARE_PER_M   = 1.5   / 1000   # Rs 1.5/km BMTC (10x cheaper)

ROAD_SAFETY = {
    "motorway": 0.45,     "motorway_link": 0.50,
    "trunk": 0.55,        "trunk_link": 0.60,
    "primary": 0.65,      "primary_link": 0.68,
    "secondary": 0.75,    "secondary_link": 0.78,
    "tertiary": 0.82,     "tertiary_link": 0.83,
    "residential": 0.92,  "living_street": 0.95,
    "unclassified": 0.80, "service": 0.85,
}
ROAD_DEFAULT_SAFETY = 0.80
BUS_SAFETY  = 0.90
WALK_SAFETY = 0.88


def _get_highway(edge_data):
    hw = edge_data.get("highway", "unclassified")
    return hw[0] if isinstance(hw, list) else str(hw)

def _get_mode(edge_data):
    return edge_data.get("mode", "road")


# ── 4 Semiring weight functions ─────────────────────────────────────

def w_distance(u, v, edge_data):
    """Semiring 1: Distance (R, min, +) — metres, all modes equal."""
    return float(edge_data.get("length", 50.0))


def w_time(u, v, edge_data):
    """
    Semiring 2: Time (R, min, +) — seconds.
    Mode matters: road uses car speed, bus adds wait penalty, walk is slow.
    Bus wait (10 min) often makes road faster even though bus is cheaper.
    """
    mode   = _get_mode(edge_data)
    length = float(edge_data.get("length", 50.0))

    if mode == "road":
        speed = ROAD_SPEED_KMH.get(_get_highway(edge_data), ROAD_DEFAULT_SPEED)
        return length / (speed * 1000 / 3600)
    elif mode == "bus":
        travel = length / (BUS_SPEED_KMH * 1000 / 3600)
        return travel + BUS_WAIT_SEC
    elif mode == "transfer":
        return length / (WALK_SPEED_KMH * 1000 / 3600)
    return length / (ROAD_DEFAULT_SPEED * 1000 / 3600)


def w_fare(u, v, edge_data):
    """
    Semiring 3: Fare (R, min, +) — rupees.
    Road (auto): Rs 15/km. Bus (BMTC): Rs 1.5/km. Walk: free.
    This semiring most strongly prefers bus routes.
    """
    mode   = _get_mode(edge_data)
    length = float(edge_data.get("length", 50.0))

    if mode == "road":     return length * ROAD_FARE_PER_M
    elif mode == "bus":    return length * BUS_FARE_PER_M
    elif mode == "transfer": return 0.0
    return length * ROAD_FARE_PER_M


def w_safety(u, v, edge_data):
    """
    Semiring 4: Safety — (R, max, min) semantics.
    Stored as (1 - score) for min-heap. Bus is safer than road.
    """
    mode = _get_mode(edge_data)
    if mode == "road":
        safety = ROAD_SAFETY.get(_get_highway(edge_data), ROAD_DEFAULT_SAFETY)
    elif mode == "bus":      safety = BUS_SAFETY
    elif mode == "transfer": safety = WALK_SAFETY
    else:                    safety = ROAD_DEFAULT_SAFETY
    return 1.0 - safety


# ── Semiring weight functions (class-based) ─────────────────────────

def w_safety_score(u, v, edge_data):
    """
    SafetySemiring weight — raw score in [0, 1] (NOT inverted).
    SafetySemiring.combine = min; SafetySemiring.better handles maximise.
    """
    mode = _get_mode(edge_data)
    if mode == "road":
        safety = ROAD_SAFETY.get(_get_highway(edge_data), ROAD_DEFAULT_SAFETY)
    elif mode == "bus":
        safety = BUS_SAFETY
    elif mode == "transfer":
        safety = WALK_SAFETY
    else:
        safety = ROAD_DEFAULT_SAFETY
    return safety   # positive; higher = safer


def w_time_fare(u, v, edge_data):
    """
    LexicographicSemiring weight — (time_seconds, fare_rupees) pair.
    Minimise time first; break ties by minimising fare.
    """
    return (w_time(u, v, edge_data), w_fare(u, v, edge_data))


def w_time_fare_transfers(u, v, edge_data):
    """
    TransferSemiring weight — (time_seconds, fare_rupees, transfer_count) triple.
    A 'transfer' mode edge (road↔bus switch) contributes 1 to the hop count;
    all other edges contribute 0. The semiring then penalises total hops
    heavily so the algorithm avoids unnecessary mode switches.
    """
    mode = _get_mode(edge_data)
    hops = 1 if mode == 'transfer' else 0
    return (w_time(u, v, edge_data), w_fare(u, v, edge_data), hops)


# ── Semiring-aware Dijkstra ──────────────────────────────────────────

def dijkstra_semiring(G, source, semiring, weight_fn):
    """
    Single-source shortest paths for an arbitrary Semiring.

    Uses the O(V^2) select-best variant so it works generically for
    any (combine, better, zero, one) without requiring a heap ordering.

    Parameters
    ----------
    G          : NetworkX graph
    source     : source node
    semiring   : Semiring instance  (combine / better / zero / one)
    weight_fn  : (u, v, edge_data) -> cost in the semiring's carrier set

    Returns
    -------
    dist : {node: best cost from source}
    prev : {node: predecessor on best path}
    """
    dist = {n: semiring.zero() for n in G.nodes}
    prev = {n: None for n in G.nodes}
    dist[source] = semiring.one()
    unvisited = set(G.nodes)

    while unvisited:
        # Pick the unvisited node with the best (non-zero) dist value.
        best = None
        for node in unvisited:
            d = dist[node]
            if d == semiring.zero():
                continue
            if best is None or semiring.better(d, dist[best]):
                best = node
        if best is None:
            break   # remaining nodes unreachable

        u = best
        unvisited.remove(u)

        for v, edge_data in G[u].items():
            if isinstance(edge_data, dict) and 0 in edge_data:
                edge_data = edge_data[0]
            w       = weight_fn(u, v, edge_data)
            relaxed = semiring.combine(dist[u], w)
            if semiring.better(relaxed, dist[v]):
                dist[v] = relaxed
                prev[v] = u

    return dist, prev


# ── Core Dijkstra (CLRS §24.3) ──────────────────────────────────────

def dijkstra(G, source, weight_fn):
    """
    Single-source shortest paths. O((V+E) log V).
    Works on road-only, bus-only, and intermodal graphs.
    weight_fn determines the semiring enrichment.
    """
    dist = {n: math.inf for n in G.nodes}
    prev = {n: None     for n in G.nodes}
    dist[source] = 0.0
    pq = [(0.0, source)]

    while pq:
        d_u, u = heapq.heappop(pq)
        if d_u > dist[u]:
            continue
        for v, edge_data in G[u].items():
            if isinstance(edge_data, dict) and 0 in edge_data:
                edge_data = edge_data[0]
            w = weight_fn(u, v, edge_data)
            if w < 0:
                raise ValueError(
                    f"Dijkstra: negative weight on edge ({u}->{v}): {w:.4f}. "
                    "Use Bellman-Ford instead."
                )
            relaxed = dist[u] + w
            if relaxed < dist[v]:
                dist[v] = relaxed
                prev[v] = u
                heapq.heappush(pq, (relaxed, v))

    return dist, prev


def reconstruct_path(prev, source, target):
    """Trace prev dict back to source. Returns [] if unreachable."""
    if prev.get(target) is None and target != source:
        return []
    path, node = [], target
    while node is not None:
        path.append(node)
        node = prev[node]
    return list(reversed(path))


def path_mode_breakdown(G, path):
    """Return {mode: {count, length_m}} for a path — useful for report."""
    breakdown = {}
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        if not G.has_edge(u, v):
            continue
        data = G[u][v]
        if isinstance(data, dict) and 0 in data:
            data = data[0]
        mode   = data.get("mode", "road")
        length = float(data.get("length", 0))
        if mode not in breakdown:
            breakdown[mode] = {"count": 0, "length_m": 0.0}
        breakdown[mode]["count"]    += 1
        breakdown[mode]["length_m"] += length
    return breakdown


# ── Run all 4 semirings ─────────────────────────────────────────────

_safety_sr   = SafetySemiring()
_lexico_sr   = LexicographicSemiring()
_transfer_sr = TransferSemiring()

# Each entry: (weight_fn, semiring_instance_or_None, unit_label, colour)
#   semiring=None  → standard tropical Dijkstra  (dijkstra)
#   semiring=<obj> → class-based Dijkstra        (dijkstra_semiring)
SEMIRINGS = {
    "distance":      (w_distance,           None,          "m",            "#1A56A0"),
    "time":          (w_time,               None,          "sec",          "#1A6E3C"),
    "fare":          (w_fare,               None,          "Rs",           "#B45309"),
    "safety":        (w_safety_score,       _safety_sr,    "safety score", "#A32D2D"),
    "lexicographic": (w_time_fare,          _lexico_sr,    "sec+Rs",       "#6B3FA0"),
    "transfers":     (w_time_fare_transfers, _transfer_sr, "sec+Rs+hops",  "#2E7D32"),
}


def run_all_semirings(G, source, target):
    results = {}
    for name, (wfn, sr, unit, colour) in SEMIRINGS.items():
        t0 = time.perf_counter()
        try:
            if sr is None:
                # Tropical (R, min, +) — standard Dijkstra
                dist, prev = dijkstra(G, source, wfn)
                t1   = time.perf_counter()
                cost = dist[target]
                display, display_unit = cost, unit
            else:
                # Class-based semiring — generic Dijkstra
                dist, prev = dijkstra_semiring(G, source, sr, wfn)
                t1   = time.perf_counter()
                cost = dist[target]
                if name == "lexicographic":
                    # Unpack tuple for display
                    display      = cost          # (time, fare) pair
                    display_unit = unit
                else:
                    display      = cost
                    display_unit = unit

            path = reconstruct_path(prev, source, target)
            results[name] = {
                "cost": display, "raw_cost": cost, "path": path,
                "path_len": len(path), "runtime_ms": (t1-t0)*1000,
                "unit": display_unit, "colour": colour,
                "breakdown": path_mode_breakdown(G, path),
                "weight_fn": wfn,
            }
        except Exception as e:
            results[name] = {
                "cost": None, "path": [], "path_len": 0,
                "runtime_ms": 0, "unit": unit, "colour": colour,
                "breakdown": {}, "error": str(e), "weight_fn": wfn,
            }
    return results


def print_results_table(results, origin, dest):
    print(f"\n  {'─'*65}")
    print(f"  {origin}  →  {dest}")
    print(f"  {'─'*65}")
    print(f"  {'Semiring':<12} {'Cost':>16}  {'Nodes':>6}  "
          f"{'Road km':>8}  {'Bus km':>7}  {'Walk m':>7}")
    print(f"  {'.'*63}")
    for name, r in results.items():
        if not r.get("path_len"):
            print(f"  {name:<12}  {'unreachable':>16}")
            continue
        bd = r.get("breakdown", {})
        road_km  = bd.get("road",     {}).get("length_m", 0) / 1000
        bus_km   = bd.get("bus",      {}).get("length_m", 0) / 1000
        walk_m   = bd.get("transfer", {}).get("length_m", 0)
        cost     = r["cost"]
        if isinstance(cost, tuple):
            cost_str = f"{cost[0]:.0f}s/Rs{cost[1]:.2f}"
        elif cost is not None:
            cost_str = f"{cost:.2f} {r['unit']}"
        else:
            cost_str = "N/A"
        print(f"  {name:<12}  {cost_str:>16}  {r['path_len']:>6}  "
              f"{road_km:>7.2f}  {bus_km:>7.2f}  {walk_m:>6.0f}")
