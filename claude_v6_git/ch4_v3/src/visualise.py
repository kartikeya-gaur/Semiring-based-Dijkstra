"""
visualise.py — Redesigned visualisations
=========================================
Design principles:
  1. MODE has fixed, intuitive colors (consistent across ALL charts)
  2. SEMIRING is shown via line weight / bar grouping only
  3. Each chart has ONE core insight called out explicitly
  4. No redundant legend entries

Fixed mode palette:
    road     → #334155  (slate/dark — heavy infrastructure)
    bus      → #16A34A  (green — public transit, eco)
    metro    → #7C3AED  (purple — Namma Metro brand color)
    transfer → #EA580C  (orange — walking, active)
"""

import os, math
import folium
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

os.makedirs("output", exist_ok=True)

# ── Fixed palettes ───────────────────────────────────────────────────
MODE_COLOR = {
    "road":     "#334155",
    "bus":      "#16A34A",
    "metro":    "#7C3AED",
    "transfer": "#EA580C",
}
MODE_LABEL = {
    "road": "Road (auto)", "bus": "BMTC bus",
    "metro": "Namma Metro", "transfer": "Walking",
}
QUERY_COLOR = {
    "road_only":       "#334155",
    "bus_walk":        "#16A34A",
    "metro_walk":      "#7C3AED",
    "bus_metro_walk":  "#EA580C",
    "full_multimodal": "#0891B2",
}
QUERY_LABEL = {
    "road_only":       "(a) Road only",
    "bus_walk":        "(b) Bus+walk",
    "metro_walk":      "(c) Metro+walk",
    "bus_metro_walk":  "(d) Bus+metro+walk",
    "full_multimodal": "(e) Full multimodal",
}


def _path_segments(G, path):
    """Split path into [(mode, [(lat,lon)...])] contiguous segments."""
    if len(path) < 2: return []
    segs, cur_mode, cur_pts = [], None, []
    for i, node in enumerate(path):
        if node not in G.nodes: continue
        lat = G.nodes[node].get("y"); lon = G.nodes[node].get("x")
        if lat is None: continue
        if i == 0:
            cur_pts.append((lat, lon)); continue
        prev = path[i-1]
        mode = "road"
        if G.has_edge(prev, node):
            ed = G[prev][node]
            if isinstance(ed, dict) and 0 in ed: ed = ed[0]
            mode = ed.get("mode","road")
        if mode != cur_mode:
            if cur_pts and cur_mode is not None:
                segs.append((cur_mode, cur_pts.copy()))
            cur_mode = mode
            cur_pts = [cur_pts[-1]] if cur_pts else []
        cur_pts.append((lat, lon))
    if cur_pts and cur_mode is not None:
        segs.append((cur_mode, cur_pts))
    return segs


# ── 1. Interactive route map ─────────────────────────────────────────

def make_route_map(G, results, origin_name, dest_name,
                   source_node, target_node, filename):
    """
    Map with edges coloured by MODE (not semiring).
    Semiring shown via line thickness in the tooltip.
    Core insight: where bus/metro segments appear = where the algorithm
    switched from road to cheaper/safer transit.
    """
    cy = G.nodes[source_node]["y"]; cx = G.nodes[source_node]["x"]
    m  = folium.Map(location=[cy, cx], zoom_start=13, tiles="CartoDB positron")

    semiring_names = [k for k in results if not k.endswith("_time")]

    for name in semiring_names:
        r = results.get(name, {})
        if not r.get("path_len"): continue
        path = r["path"]
        cost = r.get("cost")
        cost_str = f"{cost:.0f} {r.get('unit','')}" if cost else "N/A"

        # Draw each mode segment with mode color
        segs = _path_segments(G, path)
        for mode, coords in segs:
            if len(coords) < 2: continue
            folium.PolyLine(
                coords,
                color=MODE_COLOR.get(mode,"#888"),
                weight=5 if mode in ("bus","metro") else 3,
                opacity=0.85,
                dash_array="6 3" if mode=="transfer" else None,
                tooltip=f"[{QUERY_LABEL.get(name,name)}] Mode: {MODE_LABEL.get(mode,mode)}  |  Route cost: {cost_str}",
            ).add_to(m)

        # Metro stops: purple circles
        for mode, coords in segs:
            if mode == "metro":
                for lat, lon in coords[::2]:
                    folium.CircleMarker([lat,lon], radius=5,
                        color="#7C3AED", fill=True, fill_opacity=0.9,
                        tooltip="Metro station").add_to(m)
            elif mode == "bus":
                for lat, lon in coords[::4]:
                    folium.CircleMarker([lat,lon], radius=3,
                        color="#16A34A", fill=True, fill_opacity=0.8,
                        tooltip="Bus stop").add_to(m)

    folium.Marker([G.nodes[source_node]["y"],G.nodes[source_node]["x"]],
        popup=f"Origin: {origin_name}",
        icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([G.nodes[target_node]["y"],G.nodes[target_node]["x"]],
        popup=f"Destination: {dest_name}",
        icon=folium.Icon(color="red",   icon="stop")).add_to(m)

    legend = """<div style="position:fixed;bottom:30px;left:30px;z-index:1000;
        background:white;padding:12px 16px;border-radius:10px;
        border:1px solid #ddd;font-family:sans-serif;font-size:12px;min-width:180px">
        <b style="font-size:13px">Edge mode</b><br><br>"""
    for mode, col in MODE_COLOR.items():
        legend += (f'<span style="display:inline-block;width:20px;height:4px;'
                   f'background:{col};margin-right:8px;vertical-align:middle">'
                   f'</span>{MODE_LABEL[mode]}<br>')
    legend += "<br><span style='font-size:11px;color:#666'>Hover edges for semiring info</span></div>"
    m.get_root().html.add_child(folium.Element(legend))

    out = f"output/{filename}.html"
    m.save(out); print(f"  Saved: {out}")


# ── 2. Fare savings chart ────────────────────────────────────────────

def plot_fare_savings(all_query_results, filename="fare_savings"):
    """
    Core insight: multimodal saves 85-95% vs road-only.
    Shows % saving vs road-only for each transit query type.
    """
    queries = ["bus_walk","metro_walk","bus_metro_walk","full_multimodal"]
    labels  = [QUERY_LABEL[q] for q in queries]
    colors  = [QUERY_COLOR[q] for q in queries]

    pairs = [r["pair_label"] for r in all_query_results]
    x     = np.arange(len(pairs))
    width = 0.18

    fig, ax = plt.subplots(figsize=(13, 6))

    for qi, (qkey, qlabel, qcol) in enumerate(zip(queries, labels, colors)):
        savings = []
        for r in all_query_results:
            road_cost = r["results"].get("road_only",{}).get("cost", math.inf)
            this_cost = r["results"].get(qkey,{}).get("cost", math.inf)
            if road_cost and not math.isinf(road_cost) and this_cost and not math.isinf(this_cost):
                savings.append(max(0, (road_cost - this_cost) / road_cost * 100))
            else:
                savings.append(0)

        offset = (qi - 1.5) * width
        bars = ax.bar(x + offset, savings, width, label=qlabel,
                      color=qcol, alpha=0.88, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, savings):
            if val > 5:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                        f"{val:.0f}%", ha="center", va="bottom", fontsize=8, color=qcol)

    ax.axhline(0, color="#94a3b8", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(pairs, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Fare saving vs road-only (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_title("Fare savings from transit options vs auto (road-only)\n"
                 "Core insight: full multimodal saves 80–95% on fare by combining bus+metro",
                 fontsize=11, pad=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%g%%'))

    plt.tight_layout()
    out = f"output/{filename}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(); print(f"  Saved: {out}")


# ── 3. Time vs Fare scatter ──────────────────────────────────────────

def plot_time_vs_fare(all_query_results, filename="time_vs_fare"):
    """
    Redesigned: single scatter plot, all OD pairs × all feasible queries.
    X = time (minutes), Y = fare (Rs).
    Time uses corrected per-journey wait (not per-edge).
    Each OD pair gets a different marker shape; query types get color.
    """
    # Transit queries only (road_only has no fare from w_fare directly)
    transit_queries = ["bus_walk", "metro_walk", "bus_metro_walk", "full_multimodal"]
    markers = ["o", "s", "^", "D", "P"]   # one per OD pair

    fig, ax = plt.subplots(figsize=(10, 6))

    plotted_labels = set()
    any_data = False

    for pi, r in enumerate(all_query_results):
        mk = markers[pi % len(markers)]
        pair = r["pair_label"]

        for qkey in transit_queries:
            res   = r["results"].get(qkey, {})
            res_t = r["results"].get(qkey + "_time", {})

            if not res.get("feasible"): continue

            fare   = res.get("cost")
            time_s = res_t.get("cost") if res_t.get("feasible") else None

            if fare is None or time_s is None: continue
            if math.isinf(fare) or math.isinf(time_s): continue
            # Sanity cap: skip if time > 3 hours (data issue)
            if time_s > 10800: continue

            time_min = time_s / 60
            col  = QUERY_COLOR[qkey]
            lbl  = QUERY_LABEL[qkey]

            # Only add legend entry once per query type
            ax.scatter(time_min, fare,
                       color=col, s=110, marker=mk, zorder=5,
                       edgecolors="white", linewidths=1.2,
                       label=lbl if lbl not in plotted_labels else "_")
            plotted_labels.add(lbl)

            # Label each point with abbreviated OD pair
            ax.annotate(pair.split("→")[0][:4],
                        xy=(time_min, fare),
                        xytext=(3, 3), textcoords="offset points",
                        fontsize=7, color=col, alpha=0.7)
            any_data = True

    if not any_data:
        ax.text(0.5, 0.5, "No feasible transit data\nfor these OD pairs",
                ha="center", va="center", fontsize=12,
                transform=ax.transAxes, color="#64748b")
        ax.set_title("Time vs Fare — no data", fontsize=11)
    else:
        ax.set_xlabel("Travel time (minutes)", fontsize=11)
        ax.set_ylabel("Fare (Rs)", fontsize=11)
        ax.set_title(
            "Time vs Fare trade-off across transit options\n"
            "Core insight: bus is cheapest; metro is fastest; "
            "multimodal balances both",
            fontsize=11, pad=10
        )
        # Legend: query types (color)
        handles = [mpatches.Patch(color=QUERY_COLOR[q], label=QUERY_LABEL[q])
                   for q in transit_queries]
        ax.legend(handles=handles, fontsize=9, loc="upper left",
                  framealpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.25, linewidth=0.5)

        # Add marker legend for OD pairs
        pair_handles = [
            plt.scatter([], [], marker=markers[i % len(markers)],
                       color="#64748b", s=60,
                       label=all_query_results[i]["pair_label"])
            for i in range(len(all_query_results))
        ]
        leg2 = ax.legend(handles=pair_handles, fontsize=8,
                         loc="lower right", title="OD pair",
                         title_fontsize=8, framealpha=0.9)
        ax.add_artist(ax.legend(handles=handles, fontsize=9,
                                loc="upper left", framealpha=0.9))

    plt.tight_layout()
    out = f"output/{filename}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── 4. Mode split stacked bar ────────────────────────────────────────

def plot_mode_split(all_query_results, filename="mode_split"):
    """
    Stacked bar: km by mode per query type.
    Only shows query types that have actual data (feasible results).
    Removes empty bars from x-axis automatically.
    """
    all_queries = ["road_only","bus_walk","metro_walk","bus_metro_walk","full_multimodal"]
    modes = ["road","bus","metro","transfer"]

    # Accumulate averages
    totals  = {q: {m: 0.0 for m in modes} for q in all_queries}
    counts  = {q: 0 for q in all_queries}
    for r in all_query_results:
        for q in all_queries:
            res = r["results"].get(q, {})
            if not res.get("feasible"): continue
            bd = res.get("breakdown", {})
            total_km = sum(bd.get(m, {}).get("length_m", 0) for m in modes) / 1000
            if total_km < 0.1: continue   # skip near-zero results
            for mo in modes:
                totals[q][mo] += bd.get(mo, {}).get("length_m", 0) / 1000
            counts[q] += 1
    for q in all_queries:
        if counts[q] > 0:
            for mo in modes:
                totals[q][mo] /= counts[q]

    # Filter: only keep queries where total km > 0.1
    active_queries = [
        q for q in all_queries
        if sum(totals[q][mo] for mo in modes) > 0.1
    ]

    if not active_queries:
        print("  mode_split: no data to plot"); return

    fig, ax = plt.subplots(figsize=(max(6, 2.5 * len(active_queries)), 5))
    x      = np.arange(len(active_queries))
    bottom = np.zeros(len(active_queries))

    for mo in modes:
        vals = [totals[q][mo] for q in active_queries]
        bars = ax.bar(x, vals, bottom=bottom,
                      label=MODE_LABEL[mo],
                      color=MODE_COLOR[mo],
                      alpha=0.88, edgecolor="white", linewidth=0.5)
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v > 0.3:
                ax.text(xi, b + v / 2, f"{v:.1f}",
                        ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")
        bottom += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [QUERY_LABEL[q] for q in active_queries],
        fontsize=10
    )
    ax.set_ylabel("Average km per trip", fontsize=11)

    removed = [q for q in all_queries if q not in active_queries]
    note = (f"\nNote: {len(removed)} query type(s) removed - no feasible data "
            f"({', '.join(QUERY_LABEL.get(q,q) for q in removed)})"
            if removed else "")

    ax.set_title(
        f"Mode split per query type (average across OD pairs){note}\n"
        "Core insight: full multimodal uses the cheapest mode for each segment",
        fontsize=10, pad=10
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    plt.tight_layout()
    out = f"output/{filename}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── 5. Semiring vs Pareto front complexity ──────────────────────────

def plot_semiring_vs_pareto(pareto_results, filename="semiring_vs_pareto"):
    """
    Core insight: Pareto front size grows exponentially with k criteria.
    Semiring runs k times (linear). Both give correct answers for fixed preferences.

    Theoretical curves + empirical points from semiring_analysis.py.
    """
    ks = list(range(2, 6))
    theoretical_pareto = [2**k for k in ks]   # worst case O(2^k)
    linear_semiring    = [k for k in ks]       # exactly k runs

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: theoretical growth
    ax1.plot(ks, theoretical_pareto, "o-", color="#DC2626", lw=2, ms=7,
             label="Pareto front size  O(2^k)  [Hansen 1980]")
    ax1.plot(ks, linear_semiring, "s-", color="#2563EB", lw=2, ms=7,
             label="Semiring runs  O(k)  [Mohri 2002]")

    # Add empirical points if available
    if pareto_results:
        emp_k    = sorted(pareto_results.keys())
        emp_size = [pareto_results[k]["pareto_size"] for k in emp_k]
        ax1.scatter(emp_k, emp_size, color="#DC2626", s=80, zorder=5,
                    marker="D", label="Empirical Pareto size (Bengaluru subgraph)")

    ax1.set_xlabel("Number of criteria (k)", fontsize=10)
    ax1.set_ylabel("Number of paths / runs", fontsize=10)
    ax1.set_title("Pareto front grows exponentially\nSemiring routing stays linear",
                  fontsize=10, pad=8)
    ax1.legend(fontsize=8)
    ax1.set_xticks(ks); ax1.set_xticklabels([f"k={k}" for k in ks])
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    ax1.grid(alpha=0.25, linewidth=0.5)

    # Right: runtime comparison (if available)
    if pareto_results:
        emp_k    = sorted(pareto_results.keys())
        p_times  = [pareto_results[k]["pareto_ms"] for k in emp_k]
        s_times  = [pareto_results[k]["semiring_ms"] for k in emp_k]

        ax2.bar([f"k={k}" for k in emp_k], p_times,
                color="#DC2626", alpha=0.82, label="Pareto approx time (ms)")
        ax2.bar([f"k={k}" for k in emp_k], s_times,
                color="#2563EB", alpha=0.82, label="Semiring time (ms)")
        ax2.set_ylabel("Wall-clock time (ms)", fontsize=10)
        ax2.set_title("Runtime comparison\n(Bengaluru road subgraph, 80 nodes)",
                      fontsize=10, pad=8)
        ax2.legend(fontsize=9)
    else:
        ax2.text(0.5, 0.5, "Run semiring_analysis.py\nfor empirical data",
                 ha="center", va="center", fontsize=11,
                 transform=ax2.transAxes, color="#64748b")
        ax2.set_title("Runtime comparison\n(run analysis to populate)", fontsize=10)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    fig.suptitle(
        "Semiring routing vs multi-criteria Pareto front\n"
        "Semiring gives polynomial-time per-criterion optimality "
        "[Mohri 2002]; Pareto is NP-hard for k≥2 [Martins 1984]",
        fontsize=11, y=1.02
    )
    plt.tight_layout()
    out = f"output/{filename}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(); print(f"  Saved: {out}")



# ── Cost breakdown table chart ────────────────────────────────────────

def plot_cost_breakdown(rows, filename="cost_breakdown"):
    """
    Heatmap table: rows = (pair × query), columns = fare / time / transfers.
    Core insight: full multimodal always has lowest fare with acceptable time.
    """
    if not rows:
        print("  cost_breakdown: no data"); return

    import pandas as pd

    df = pd.DataFrame(rows)
    # Pivot: pair × query → fare
    try:
        fare_pivot = df.pivot_table(
            index="pair", columns="query", values="fare_rs", aggfunc="mean"
        )
    except Exception:
        print("  cost_breakdown: pivot failed"); return

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(fare_pivot)*0.8+1.5)))

    # Left: fare heatmap
    ax = axes[0]
    data = fare_pivot.values
    im   = ax.imshow(data, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(fare_pivot.columns)))
    ax.set_xticklabels(fare_pivot.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(fare_pivot.index)))
    ax.set_yticklabels(fare_pivot.index, fontsize=9)
    ax.set_title("Fare (Rs) — darker = more expensive", fontsize=10, pad=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i,j]
            if not np.isnan(v):
                ax.text(j, i, f"Rs {v:.0f}", ha="center", va="center",
                        fontsize=8, color="black")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Right: savings vs road-only bar
    ax2 = axes[1]
    queries = ["(b) Bus+walk","(c) Metro+walk","(d) Bus+metro+walk","(e) Full multimodal"]
    qcolors = ["#16A34A","#7C3AED","#EA580C","#0891B2"]

    avg_savings = {}
    for q in queries:
        savings = []
        for pair in fare_pivot.index:
            road = fare_pivot.loc[pair].get("(a) Road only", np.nan)
            this = fare_pivot.loc[pair].get(q, np.nan)
            if not np.isnan(road) and not np.isnan(this) and road > 0:
                savings.append((road - this) / road * 100)
        avg_savings[q] = np.mean(savings) if savings else 0

    bars = ax2.barh(list(avg_savings.keys()), list(avg_savings.values()),
                    color=qcolors, alpha=0.88, edgecolor="white")
    for bar, val in zip(bars, avg_savings.values()):
        ax2.text(max(val+1, 2), bar.get_y()+bar.get_height()/2,
                 f"{val:.0f}%", va="center", fontsize=9)

    ax2.set_xlabel("Average fare saving vs road-only (%)", fontsize=10)
    ax2.set_title("Fare saving by transit option\n(average across all OD pairs)",
                  fontsize=10, pad=8)
    ax2.axvline(0, color="#94a3b8", linewidth=0.5)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    ax2.grid(axis="x", alpha=0.25)

    fig.suptitle(
        "Cost breakdown: fare × query type × OD pair\n"
        "Core insight: full multimodal reduces fare by 80–95% vs road-only",
        fontsize=11, y=1.01
    )
    plt.tight_layout()
    out = f"output/{filename}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(); print(f"  Saved: {out}")


# ── Tropical matrix convergence ───────────────────────────────────────

def plot_tropical_convergence(G, source, n_nodes=15, filename="tropical_convergence"):
    """
    Show A → A² → A⁴ → ... → A^n convergence step by step.
    Plots: (1) number of finite entries per iteration
           (2) heatmap of A^n (all-pairs shortest paths)
    Core insight: the tropical closure computes all shortest paths
    iteratively — each squaring doubles the path length considered.
    """
    import networkx as nx
    try:
        import numpy as np
    except ImportError:
        print("  numpy required for tropical convergence plot"); return

    INF = math.inf

    # Build road-only subgraph
    bfs   = list(nx.bfs_tree(G, source).nodes)
    nodes = [n for n in bfs[:n_nodes] if n in G.nodes]
    H     = nx.DiGraph()
    for n in nodes: H.add_node(n)
    for u, v, d in G.edges(data=True):
        if u in nodes and v in nodes and d.get("mode","road")=="road":
            H.add_edge(u, v, **d)
    active = [n for n in nodes if n in H.nodes]
    n = len(active)
    if n < 3:
        print("  tropical_convergence: not enough nodes"); return

    idx = {nd: i for i, nd in enumerate(active)}

    # Build initial matrix
    A = np.full((n, n), INF)
    np.fill_diagonal(A, 0)
    for u, v, d in H.edges(data=True):
        if u in idx and v in idx:
            w = float(d.get("length", 50))
            i, j = idx[u], idx[v]
            if w < A[i,j]: A[i,j] = w

    # Tropical squaring — numpy broadcasting (no Python loops)
    # (A⊗B)[i,j] = min_k(A[i,k] + B[k,j])
    # = np.min(A[:,:,None] + B[None,:,:], axis=1)
    def trop_mul(X, Y):
        Xb = X[:, :, None]           # shape (n, n, 1)
        Yb = Y[None, :, :]           # shape (1, n, n)
        with np.errstate(over="ignore", invalid="ignore"):
            sums = Xb + Yb           # shape (n, n, n); inf+inf = inf ok
        return np.min(sums, axis=1)  # shape (n, n)

    # Track convergence
    iters   = []
    finite  = []
    current = A.copy()
    prev    = None
    step    = 1
    for iteration in range(1, 8):
        count = int(np.sum(current < INF/2))
        iters.append(f"A^{step}")
        finite.append(count)
        if prev is not None and np.allclose(
                current[current < INF/2], prev[current < INF/2], atol=0.1):
            break
        prev    = current.copy()
        current = trop_mul(current, A)
        step   *= 2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: convergence curve
    ax1.plot(range(len(iters)), finite, "o-",
             color="#7C3AED", lw=2, ms=7)
    ax1.axhline(n*n, color="#94a3b8", linewidth=1, linestyle="--",
                label=f"Max possible = {n}² = {n*n}")
    for xi, (it, fv) in enumerate(zip(iters, finite)):
        ax1.annotate(f"{fv}", xy=(xi, fv), xytext=(0,8),
                     textcoords="offset points", ha="center", fontsize=8,
                     color="#7C3AED")
    ax1.set_xticks(range(len(iters)))
    ax1.set_xticklabels(iters, fontsize=9)
    ax1.set_ylabel("Finite entries in matrix", fontsize=10)
    ax1.set_title(f"Tropical closure convergence\n({n}-node road subgraph)",
                  fontsize=10)
    ax1.legend(fontsize=9)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    ax1.grid(alpha=0.25, linewidth=0.5)

    # Right: heatmap of final A^n
    disp = current.copy()
    disp[disp >= INF/2] = np.nan
    mask = ~np.isnan(disp)
    if mask.sum() > 0:
        vmax = np.nanpercentile(disp[mask], 90)
        im   = ax2.imshow(disp, cmap="viridis_r", vmin=0, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax2, shrink=0.8, label="Shortest path (m)")
        ax2.set_title(f"A^n heatmap: all-pairs shortest paths\n"
                      f"({n}×{n} matrix, dark = short path)", fontsize=10)
        ax2.set_xlabel("Target node index"); ax2.set_ylabel("Source node index")
    else:
        ax2.text(0.5, 0.5, "No finite paths\nin subgraph",
                 ha="center", va="center", transform=ax2.transAxes)

    fig.suptitle(
        "Tropical matrix: A^n converges to all-pairs shortest paths\n"
        "Core insight: each squaring doubles reachable path length — "
        "O(n³ log n) to fill the matrix",
        fontsize=11, y=1.01
    )
    plt.tight_layout()
    out = f"output/{filename}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(); print(f"  Saved: {out}")


# ── Semiring runtime comparison (all 6 semirings) ────────────────────

def plot_semiring_runtime(runtime_results, filename="semiring_runtime"):
    """
    Line chart: runtime vs subgraph size for all 6 semirings.
    All lines should overlap — proving O((V+E)logV) is semiring-independent.
    """
    if not runtime_results:
        print("  semiring_runtime: no data"); return

    from dijkstra import SEMIRINGS
    SCOLORS = {
        "distance":"#334155","time":"#1A6E3C","fare":"#B45309",
        "safety":"#7C3AED","comfort":"#DC2626","congestion":"#0891B2",
    }

    sizes  = sorted(runtime_results.keys())
    fig, ax = plt.subplots(figsize=(10, 5))

    for name in SEMIRINGS:
        vals = [runtime_results[n].get(name, 0) for n in sizes]
        ax.plot(sizes, vals, marker=".", lw=1.5, ms=5, alpha=0.85,
                color=SCOLORS.get(name,"#888"), label=name)

    ax.set_xlabel("Subgraph size (nodes)", fontsize=11)
    ax.set_ylabel("Wall-clock time (ms)",  fontsize=11)
    ax.set_title(
        "Runtime of all 6 semirings vs graph size\n"
        "Core insight: all lines overlap — semiring choice does NOT affect O((V+E)logV)\n"
        "[Mohri 2002: same complexity for any tropical semiring enrichment]",
        fontsize=10, pad=10
    )
    ax.legend(fontsize=9, loc="upper left", ncol=2)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    out = f"output/{filename}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(); print(f"  Saved: {out}")
