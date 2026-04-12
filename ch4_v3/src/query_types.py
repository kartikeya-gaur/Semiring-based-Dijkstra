"""
query_types.py — 5 structured routing queries
==============================================

(a) road_only       — G_r only, w_time
(b) bus_walk        — G_b ∪ G_t, w_fare
(c) metro_walk      — G_m ∪ G_t, w_fare
(d) bus_metro_walk  — G_b ∪ G_m ∪ G_t, w_fare
(e) full_multimodal — G_r ∪ G_b ∪ G_m ∪ G_t, w_fare

Running (b), (c), (d) with BOTH w_fare and w_time gives the
time-vs-fare trade-off matrix used in the scatter plot.

CT Framing:
    Each query is Dijkstra on a different sub-category of the full
    intermodal category G. The sub-categories differ only in which
    morphisms (edge modes) are included — not in the algorithm.
"""

import math, time
import networkx as nx
from dijkstra import dijkstra, reconstruct_path, path_mode_breakdown, w_fare, w_time


def subgraph_by_modes(G, allowed_modes: set) -> nx.DiGraph:
    """Extract subgraph keeping only edges whose mode is in allowed_modes."""
    H = nx.DiGraph()
    for u, v, d in G.edges(data=True):
        if d.get("mode","road") in allowed_modes:
            for n in (u, v):
                if n not in H and n in G.nodes:
                    H.add_node(n, **G.nodes[n])
            H.add_edge(u, v, **d)
    return H


def _run(G, source, target, wfn, label):
    """Run Dijkstra and return standardised result dict."""
    if source not in G.nodes or target not in G.nodes:
        return _inf(f"source or target not in subgraph")
    if not nx.has_path(G, source, target):
        return _inf("no path in subgraph")
    t0 = time.perf_counter()
    dist, prev = dijkstra(G, source, wfn)
    ms = (time.perf_counter()-t0)*1000
    cost = dist[target]
    if math.isinf(cost): return _inf("unreachable")
    path = reconstruct_path(prev, source, target)
    bd   = path_mode_breakdown(G, path)
    seq  = _mode_seq(G, path)
    return {
        "feasible":True,"query":label,"cost":cost,
        "path":path,"path_len":len(path),"runtime_ms":ms,
        "breakdown":bd,"mode_sequence":seq,
    }


def _inf(reason):
    return {"feasible":False,"reason":reason,"cost":math.inf,
            "path":[],"path_len":0,"runtime_ms":0,"breakdown":{},"mode_sequence":[]}


def _mode_seq(G, path):
    modes = []
    for i in range(len(path)-1):
        u,v = path[i],path[i+1]
        if not G.has_edge(u,v): continue
        ed = G[u][v]
        if isinstance(ed,dict) and 0 in ed: ed=ed[0]
        modes.append(ed.get("mode","road"))
    if not modes: return []
    seq = [modes[0]]
    for m in modes[1:]:
        if m != seq[-1]: seq.append(m)
    return seq


QUERY_DEFS = {
    "road_only":      ({"road"},                       w_time,  "(a) Road only"),
    "bus_walk":       ({"bus","transfer"},             w_fare,  "(b) Bus + walk"),
    "metro_walk":     ({"metro","transfer"},           w_fare,  "(c) Metro + walk"),
    "bus_metro_walk": ({"bus","metro","transfer"},     w_fare,  "(d) Bus+metro+walk"),
    "full_multimodal":({"road","bus","metro","transfer"},w_fare,"(e) Full multimodal"),
}


def run_all_queries(G, source, target, origin_name="", dest_name=""):
    """Run all 5 query types. Also run (b)(c)(d) with w_time for time comparison."""
    results = {}
    for key, (modes, wfn, label) in QUERY_DEFS.items():
        H = subgraph_by_modes(G, modes)
        results[key] = _run(H, source, target, wfn, label)

    # Also get time costs for transit-only options (b,c,d)
    for key in ("bus_walk","metro_walk","bus_metro_walk"):
        modes = QUERY_DEFS[key][0]
        H = subgraph_by_modes(G, modes)
        r = _run(H, source, target, w_time, QUERY_DEFS[key][2]+" [time]")
        results[key+"_time"] = r

    _print_table(results, origin_name, dest_name)
    return results


def _print_table(results, orig, dest):
    labels = {k:v[2] for k,v in QUERY_DEFS.items()}
    print(f"\n  {'─'*72}")
    print(f"  {orig}  →  {dest}")
    print(f"  {'─'*72}")
    print(f"  {'Query':<28} {'Feasible':>9}  {'Fare (Rs)':>10}  {'Time (min)':>11}  {'Sequence'}")
    print(f"  {'.'*70}")
    for key, label in labels.items():
        r  = results[key]
        rt = results.get(key+"_time", {})
        fare_str = f"{r['cost']:>8.0f}" if r["feasible"] and not math.isinf(r["cost"]) else "       —"
        time_min = rt.get("cost", math.inf)
        time_str = f"{time_min/60:>9.1f}" if rt.get("feasible") and not math.isinf(time_min) else "        —"
        seq_str  = " → ".join(r.get("mode_sequence",[]))[:30] if r["feasible"] else r.get("reason","N/A")[:30]
        print(f"  {label:<28}  {'Yes':>9}  {fare_str}  {time_str}  {seq_str}"
              if r["feasible"] else
              f"  {label:<28}  {'No':>9}  {fare_str}  {time_str}  {seq_str}")
    print()
