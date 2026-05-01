"""
bellman_ford.py — Bellman-Ford (CLRS §24.1) + negative weight demo
===================================================================
Demonstrates where Dijkstra fails (negative weights) and
provides runtime comparison to validate O(VE) vs O((V+E)logV).
"""

import math, time, copy
import networkx as nx
from dijkstra import dijkstra, w_fare, ROAD_FARE_PER_M


def bellman_ford(G, source, weight_fn):
    """
    Bellman-Ford single-source shortest paths. CLRS §24.1.
    O(V * E) — much slower than Dijkstra but handles negative weights.
    Raises ValueError on negative-weight cycle.
    """
    nodes = list(G.nodes)
    dist  = {n: math.inf for n in nodes}
    prev  = {n: None     for n in nodes}
    dist[source] = 0.0

    edges = []
    for u in G.nodes:
        for v, ed in G[u].items():
            if isinstance(ed, dict) and 0 in ed:
                ed = ed[0]
            edges.append((u, v, ed))

    for _ in range(len(nodes) - 1):
        updated = False
        for u, v, ed in edges:
            w = weight_fn(u, v, ed)
            if dist[u] != math.inf and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
                updated = True
        if not updated:
            break

    # Detect negative-weight cycle
    for u, v, ed in edges:
        if dist[u] != math.inf and dist[u] + weight_fn(u, v, ed) < dist[v]:
            raise ValueError(
                f"Negative-weight cycle reachable from {source}")

    return dist, prev


def reconstruct_path(prev, source, target):
    if prev.get(target) is None and target != source:
        return []
    path, node = [], target
    while node is not None:
        path.append(node)
        node = prev[node]
    return list(reversed(path))


# ── Negative-weight demonstration ──────────────────────────────────

def w_fare_with_subsidy(u, v, edge_data):
    """Fare weight including an optional government subsidy (negative)."""
    length  = float(edge_data.get("length", 50.0))
    mode    = edge_data.get("mode", "road")
    base    = length * (ROAD_FARE_PER_M if mode == "road" else 1.5/1000)
    subsidy = float(edge_data.get("subsidy", 0.0))
    return base + subsidy   # subsidy is negative → reduces cost


def run_negative_weight_demo(G, source, target):
    """
    Add a subsidised (negative-fare) edge to a copy of the graph.
    Show Dijkstra fails / gets wrong answer; Bellman-Ford is correct.
    """
    # Pick two connected road nodes near source
    nbrs = [v for v in G.successors(source)
        if G[source][v].get("mode", "road") == "road"]
    if len(nbrs) < 1:
        print("  Demo: no road neighbours found — skipping")
        return {}

    u_sub = source
    v_sub = nbrs[0]

    G_demo = copy.deepcopy(G)
    # Inject a Rs 20 subsidy on this one edge
    # Subsidy injection — same, no [0] needed
    if G_demo.has_edge(u_sub, v_sub):
        G_demo[u_sub][v_sub]["subsidy"] = -20.0
    if G_demo.has_edge(v_sub, u_sub):
        G_demo[v_sub][u_sub]["subsidy"] = -20.0
    

    results = {}

    # Dijkstra
    print("  Dijkstra on graph with negative-fare edge...")
    try:
        t0 = time.perf_counter()
        dist, prev = dijkstra(G_demo, source, w_fare_with_subsidy)
        t1 = time.perf_counter()
        path = reconstruct_path(prev, source, target)
        results["dijkstra"] = {
            "status": "completed (may be incorrect — negative edge ignored)",
            "cost":   dist[target],
            "ms":     (t1-t0)*1000,
        }
        print(f"    Cost: Rs {dist[target]:.2f} (WARNING: Dijkstra "
              f"cannot guarantee correctness with negative weights)")
    except ValueError as e:
        results["dijkstra"] = {"status": f"FAILED: {e}", "cost": None}
        print(f"    FAILED: {e}")

    # Bellman-Ford
    print("  Bellman-Ford on same graph...")
    try:
        t0 = time.perf_counter()
        dist_bf, prev_bf = bellman_ford(G_demo, source, w_fare_with_subsidy)
        t1 = time.perf_counter()
        results["bellman_ford"] = {
            "status": "succeeded",
            "cost":   dist_bf[target],
            "ms":     (t1-t0)*1000,
        }
        print(f"    Cost: Rs {dist_bf[target]:.2f} "
              f"(correct — uses Rs 20 subsidised segment)")
    except ValueError as e:
        results["bellman_ford"] = {
            "status": f"FAILED (negative cycle): {e}", "cost": None}
        print(f"    FAILED: {e}")

    return results


# ── Runtime scaling ─────────────────────────────────────────────────

def compare_runtimes(G, source, target, sizes=None):
    """
    Time both algorithms on BFS subgraphs of increasing size.
    Returns {n: {dijkstra_ms, bellman_ford_ms}}.
    """
    if sizes is None:
        sizes = [50, 100, 200, 300, 500]

    bfs_nodes = list(nx.bfs_tree(G, source).nodes)
    results   = {}

    for n in sizes:
        if n > len(bfs_nodes):
            break
        H   = G.subgraph(bfs_nodes[:n]).copy()
        tgt = target if target in H.nodes else bfs_nodes[min(n-1, len(bfs_nodes)-1)]

        t0 = time.perf_counter()
        dijkstra(H, source, w_fare)
        d_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        bellman_ford(H, source, w_fare)
        bf_ms = (time.perf_counter() - t0) * 1000

        results[n] = {"dijkstra_ms": d_ms, "bellman_ford_ms": bf_ms}
        print(f"  n={n:4d}:  Dijkstra {d_ms:7.2f}ms  |  "
              f"Bellman-Ford {bf_ms:7.2f}ms")

    return results
