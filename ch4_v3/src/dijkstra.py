"""
dijkstra.py
===========
Dijkstra's algorithm (CLRS §24.3) parameterised over a semiring.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHERE THE SEMIRING STRUCTURE LIVES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A semiring (S, ⊕, ⊗) has two operations:
    ⊕  "addition"       — combines alternatives   (we want the best)
    ⊗  "multiplication" — combines costs in series (we add them up)

The tropical semiring T = (R∪{∞}, min, +) uses:
    ⊕ = min    →  "take the cheaper option"
    ⊗ = +      →  "add costs along a path"

This file implements T four times, once per routing criterion:

    SEMIRING 1 — Distance  (R, min, +)   weight = metres
    SEMIRING 2 — Time      (R, min, +)   weight = seconds
    SEMIRING 3 — Fare      (R, min, +)   weight = rupees
    SEMIRING 4 — Safety    (R, max, min) weight = 1 − safety_score

The ⊕ = min operation lives inside Dijkstra's relaxation step:
    if dist[u] + w(u→v)  <  dist[v]:   ← this "<" IS the ⊕ = min
        dist[v] = dist[u] + w(u→v)      ← this "+" IS the ⊗ = +

The weight_fn parameter selects which semiring enrichment to use.
Swapping weight_fn = swapping the semiring. The algorithm is identical.

CT Reference: ACT4E §14.3 — the city graph is a category enriched
over T. Dijkstra computes the optimal morphism (cheapest path) between
two objects in this enriched category.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import heapq, math, time

# ── Road speeds by OSM highway type (km/h) ───────────────────────────
ROAD_SPEED = {
    "motorway":80, "motorway_link":60, "trunk":65,    "trunk_link":50,
    "primary":50,  "primary_link":40,  "secondary":40, "secondary_link":35,
    "tertiary":35, "tertiary_link":30, "residential":25,"living_street":15,
    "unclassified":30, "service":20,
}
ROAD_SPEED_DEFAULT = 25

# ── Mode travel constants ─────────────────────────────────────────────
BUS_SPEED_KMH   = 18.0;  BUS_WAIT_SEC   = 600.0   # 10-min avg wait
METRO_SPEED_KMH = 32.0;  METRO_WAIT_SEC = 300.0   # 5-min avg wait
WALK_SPEED_KMH  = 5.0

# ── Fare rates ────────────────────────────────────────────────────────
ROAD_FARE_PER_M  = 15.0 / 1000    # Rs 15/km  (auto)
BUS_FARE_PER_M   = 1.5  / 1000    # Rs 1.5/km (BMTC)
METRO_BASE_FARE  = 45.0            # Rs 45 flat minimum
METRO_FARE_PER_M = 4.0  / 1000    # Rs 4/km   (above base)

# ── Safety scores (0 = dangerous, 1 = safe) ──────────────────────────
ROAD_SAFETY = {
    "motorway":0.45, "motorway_link":0.50, "trunk":0.55,   "trunk_link":0.60,
    "primary":0.65,  "primary_link":0.68,  "secondary":0.75,"secondary_link":0.78,
    "tertiary":0.82, "tertiary_link":0.83, "residential":0.92,"living_street":0.95,
    "unclassified":0.80, "service":0.85,
}
ROAD_SAFETY_DEFAULT = 0.80
BUS_SAFETY   = 0.90
METRO_SAFETY = 0.95   # grade-separated, enclosed — safest mode
WALK_SAFETY  = 0.88


def _hw(ed):
    hw = ed.get("highway", "unclassified")
    return hw[0] if isinstance(hw, list) else str(hw)

def _mode(ed):
    return ed.get("mode", "road")


# ════════════════════════════════════════════════════════════════════
#  THE SEMIRING WEIGHT FUNCTIONS
#  ─────────────────────────────────────────────────────────────────
#  Each function defines one edge cost: w(u, v, edge_data) → float
#  This IS the semiring enrichment of the city graph.
#  The function maps each edge to its cost in the chosen semiring.
#
#  Dijkstra then computes:
#      dist[v]  =  min over all paths P from source to v  of  Σ w(e) for e in P
#                  ^^^                                         ^^^^^^^^^^^^^^^
#                  ⊕ = min (the semiring "addition")          ⊗ = + (the semiring "multiplication")
# ════════════════════════════════════════════════════════════════════

# ── SEMIRING 1: Distance  (R≥0, min, +) ─────────────────────────────
def w_distance(u, v, ed):
    """
    ⊗ element: length in metres.
    Same for all modes — purely geometric.
    Dijkstra minimises total path length.
    """
    return float(ed.get("length", 50.0))


# ── SEMIRING 2: Time  (R≥0, min, +) ─────────────────────────────────
def w_time(u, v, ed):
    """
    ⊗ element: travel time in seconds.
    Mode changes the speed and adds a one-time wait penalty:
        road     → distance / speed(highway_type)          no wait
        bus      → distance / 18 km/h  +  600s wait        10 min
        metro    → distance / 32 km/h  +  300s wait         5 min
        transfer → distance / 5 km/h                  walking only

    The wait penalty is the key reason the time semiring often
    prefers road over transit even when transit is much cheaper.
    """
    m, l = _mode(ed), float(ed.get("length", 50.0))
    if m == "road":
        spd = ROAD_SPEED.get(_hw(ed), ROAD_SPEED_DEFAULT)
        return l / (spd * 1000 / 3600)
    elif m == "bus":
        return l / (BUS_SPEED_KMH * 1000 / 3600) + BUS_WAIT_SEC
    elif m == "metro":
        return l / (METRO_SPEED_KMH * 1000 / 3600) + METRO_WAIT_SEC
    elif m == "transfer":
        return l / (WALK_SPEED_KMH * 1000 / 3600)
    return l / (ROAD_SPEED_DEFAULT * 1000 / 3600)


# ── SEMIRING 3: Fare  (R≥0, min, +) ─────────────────────────────────
def w_fare(u, v, ed):
    """
    ⊗ element: cost in rupees.
    Mode dramatically changes the cost:
        road     → Rs 15/km  (auto)     — most expensive
        bus      → Rs 1.5/km (BMTC)    — cheapest per km
        metro    → Rs 45 base + Rs 4/km — cheaper than auto for >3km
        transfer → Rs 0                 — walking is free

    The 10× gap between road and bus is what makes the fare semiring
    route through transit even when it means a longer path.
    """
    m, l = _mode(ed), float(ed.get("length", 50.0))
    if m == "road":      return l * ROAD_FARE_PER_M
    elif m == "bus":     return l * BUS_FARE_PER_M
    elif m == "metro":   return METRO_BASE_FARE + l * METRO_FARE_PER_M
    elif m == "transfer":return 0.0
    return l * ROAD_FARE_PER_M


# ── SEMIRING 4: Safety  (R, max, min) ────────────────────────────────
def w_safety(u, v, ed):
    """
    ⊗ element: 1 − safety_score  (inverted for min-heap compatibility).

    The safety semiring is (R, max, min):
        ⊕ = max   →  "take the safer alternative"
        ⊗ = min   →  "the path safety is its worst segment"

    To use standard min-heap Dijkstra, we invert: store (1 − score).
    Minimising (1 − score) = maximising score.

    Safety ranking:  metro (0.95) > bus (0.90) > walk (0.88) > road (varies)
    """
    m = _mode(ed)
    if m == "road":      s = ROAD_SAFETY.get(_hw(ed), ROAD_SAFETY_DEFAULT)
    elif m == "bus":     s = BUS_SAFETY
    elif m == "metro":   s = METRO_SAFETY
    elif m == "transfer":s = WALK_SAFETY
    else:                s = ROAD_SAFETY_DEFAULT
    return 1.0 - s   # ← inversion: min(1-s) = max(s)


# ── SEMIRING 5: Comfort  (R≥0, min, +) ──────────────────────────────
# Comfort scores: how unpleasant is each edge?
COMFORT_COST = {
    "road":     0.1,   # auto: door-to-door, private, comfortable
    "metro":    0.2,   # metro: clean, AC, predictable
    "bus":      0.5,   # bus: crowded, standing, unpredictable
    "transfer": 1.0,   # walking transfer: inconvenient, weather-exposed
}

def w_comfort(u, v, ed):
    """
    ⊗ element: discomfort score × distance (km).
    Minimising total discomfort finds the route that involves
    least inconvenience — prefers auto and metro over bus,
    and strongly penalises mode switches (walking transfers).

    NEW SEMIRING — zero algorithm changes needed.
    This demonstrates the CT advantage: adding a 5th routing
    criterion required only this 1 function. Dijkstra, the graph,
    and all correctness proofs are completely unchanged.
    (ACT4E §14.6: one algorithm over any enrichment)
    """
    m = _mode(ed)
    l_km = float(ed.get("length", 50.0)) / 1000   # convert to km
    return COMFORT_COST.get(m, 0.3) * l_km


# ── SEMIRING 6: Congestion-aware time  (R≥0, min, +) ─────────────────
# Peak-hour congestion multipliers by road type (speeds drop)
# Based on Bengaluru traffic survey data (approximate)
CONGESTION_PEAK = {
    "motorway": 0.55,   "motorway_link": 0.50,
    "trunk": 0.45,      "trunk_link": 0.50,
    "primary": 0.40,    "primary_link": 0.45,   # Outer Ring Road: worst
    "secondary": 0.50,  "secondary_link": 0.55,
    "tertiary": 0.65,   "tertiary_link": 0.70,
    "residential": 0.85,"living_street": 0.90,
    "unclassified": 0.75,"service": 0.85,
}
CONGESTION_OFFPEAK = {k: 1.0 for k in CONGESTION_PEAK}  # no congestion

# Global flag — set in main.py before running
IS_PEAK_HOUR = True

def w_congestion(u, v, ed):
    """
    ⊗ element: congested travel time in seconds.
    Same as w_time but road speeds are multiplied by a congestion
    factor that drops to 0.40 on primary roads during peak hour.

    Bus and metro are unaffected by road congestion (dedicated lanes/tracks).
    This makes transit options relatively MORE attractive at peak hour —
    exactly matching real-world Bengaluru commute patterns.

    Peak hour:  7–10am, 5–8pm (set IS_PEAK_HOUR=True)
    Off peak:   all other times (set IS_PEAK_HOUR=False)
    """
    m, l = _mode(ed), float(ed.get("length", 50.0))
    prev_m = ed.get("prev_mode", "")

    if m == "road":
        hw  = _hw(ed)
        spd = ROAD_SPEED.get(hw, ROAD_SPEED_DEFAULT)
        cong = (CONGESTION_PEAK if IS_PEAK_HOUR else CONGESTION_OFFPEAK).get(hw, 0.7)
        effective_spd = spd * cong
        return l / (effective_spd * 1000 / 3600)

    elif m == "bus":
        # Buses also slow down in traffic (not on dedicated lanes)
        cong = 0.60 if IS_PEAK_HOUR else 1.0
        travel = l / (BUS_SPEED_KMH * cong * 1000 / 3600)
        wait = BUS_WAIT_SEC if prev_m in ("transfer", "road", "") else 0.0
        return travel + wait

    elif m == "metro":
        # Metro unaffected — grade separated
        travel = l / (METRO_SPEED_KMH * 1000 / 3600)
        wait = METRO_WAIT_SEC if prev_m in ("transfer", "road", "") else 0.0
        return travel + wait

    elif m == "transfer":
        return l / (WALK_SPEED_KMH * 1000 / 3600)

    return l / (ROAD_SPEED_DEFAULT * 1000 / 3600)


# ════════════════════════════════════════════════════════════════════
#  DIJKSTRA'S ALGORITHM  (CLRS §24.3)
#  ─────────────────────────────────────────────────────────────────
#  This function is IDENTICAL for all four semirings.
#  The only input that changes is weight_fn.
#
#  The semiring operations appear at two points:
#      Line marked [⊗]: r = dist[u] + w   ← tropical multiplication
#      Line marked [⊕]: if r < dist[v]    ← tropical addition (min)
# ════════════════════════════════════════════════════════════════════

def dijkstra(G, source, weight_fn):
    """
    Single-source shortest paths over the semiring defined by weight_fn.
    O((V + E) log V) — same complexity for all four semirings.

    weight_fn(u, v, edge_data) → float
        Defines the ⊗ operation: cost of traversing one edge.
        The ⊕ = min operation is hardcoded in the relaxation step.
    """
    dist = {n: math.inf for n in G.nodes}
    prev = {n: None     for n in G.nodes}
    dist[source] = 0.0

    # Tuple: (cost, counter, node)
    # Counter breaks ties without comparing mixed int/str node IDs.
    counter = 0
    pq = [(0.0, counter, source)]

    while pq:
        d_u, _, u = heapq.heappop(pq)
        if d_u > dist[u]:
            continue   # stale entry — already processed

        for v, ed in G[u].items():
            if isinstance(ed, dict) and 0 in ed:
                ed = ed[0]

            w = weight_fn(u, v, ed)             # ← ⊗ : edge cost in this semiring

            r = dist[u] + w                      # ← ⊗ : path cost = accumulate costs

            if r < dist[v]:                      # ← ⊕ : min over alternatives
                dist[v] = r
                prev[v] = u
                counter += 1
                heapq.heappush(pq, (r, counter, v))

    return dist, prev


def reconstruct_path(prev, source, target):
    """Trace prev pointers back to source. Returns [] if unreachable."""
    if prev.get(target) is None and target != source:
        return []
    path, n = [], target
    while n is not None:
        path.append(n); n = prev[n]
    return list(reversed(path))


def path_mode_breakdown(G, path):
    """Return {mode: {count, length_m}} for a reconstructed path."""
    bd = {}
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if not G.has_edge(u, v): continue
        ed = G[u][v]
        if isinstance(ed, dict) and 0 in ed: ed = ed[0]
        m = ed.get("mode", "road"); l = float(ed.get("length", 0))
        bd.setdefault(m, {"count": 0, "length_m": 0.0})
        bd[m]["count"] += 1; bd[m]["length_m"] += l
    return bd


# ── Semiring registry ─────────────────────────────────────────────────
# Maps semiring name → (weight_fn, unit_label, chart_colour)
# To add a new semiring: add one entry here and write its weight_fn above.

SEMIRINGS = {
    "distance":   (w_distance,   "m",        "#334155"),
    "time":       (w_time,       "sec",      "#1A6E3C"),
    "fare":       (w_fare,       "Rs",       "#B45309"),
    "safety":     (w_safety,     "score",    "#7C3AED"),
    "comfort":    (w_comfort,    "discomfort","#DC2626"),
    "congestion": (w_congestion, "sec",      "#0891B2"),
}


def run_all_semirings(G, source, target):
    """Run Dijkstra once per semiring. Returns dict keyed by semiring name."""
    results = {}
    for name, (wfn, unit, colour) in SEMIRINGS.items():
        t0 = time.perf_counter()
        try:
            dist, prev = dijkstra(G, source, wfn)
            ms   = (time.perf_counter() - t0) * 1000
            path = reconstruct_path(prev, source, target)
            cost = dist[target]
            disp = (1.0 - cost) if name == "safety" else cost
            results[name] = {
                "cost": disp, "raw_cost": cost, "path": path,
                "path_len": len(path), "runtime_ms": ms,
                "unit": "safety score" if name == "safety" else unit,
                "colour": colour,
                "breakdown": path_mode_breakdown(G, path),
                "weight_fn": wfn,
            }
        except Exception as e:
            results[name] = {
                "cost": None, "path": [], "path_len": 0, "runtime_ms": 0,
                "unit": unit, "colour": colour, "breakdown": {},
                "error": str(e), "weight_fn": wfn,
            }
    return results


def print_results_table(results, origin, dest):
    print(f"\n  {'─'*70}")
    print(f"  {origin}  →  {dest}")
    print(f"  {'─'*70}")
    print(f"  {'Semiring':<12} {'Cost':>16}  {'Nodes':>6}  "
          f"{'Road':>7}  {'Bus':>7}  {'Metro':>7}  {'Walk':>6}")
    print(f"  {'.'*68}")
    for name, r in results.items():
        if not r.get("path_len"):
            print(f"  {name:<12}  {'unreachable':>16}"); continue
        bd = r.get("breakdown", {})
        rk = bd.get("road",    {}).get("length_m", 0) / 1000
        bk = bd.get("bus",     {}).get("length_m", 0) / 1000
        mk = bd.get("metro",   {}).get("length_m", 0) / 1000
        wm = bd.get("transfer",{}).get("length_m", 0)
        c  = r["cost"]
        cs = f"{c:.2f} {r['unit']}" if c is not None else "N/A"
        print(f"  {name:<12}  {cs:>16}  {r['path_len']:>6}  "
              f"{rk:>6.2f}  {bk:>6.2f}  {mk:>6.2f}  {wm:>5.0f}m")
