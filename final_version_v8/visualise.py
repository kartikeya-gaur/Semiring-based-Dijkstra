"""
visualise.py — Maps and plots for Road+Bus multimodal routing
=============================================================
Three outputs:
  1. output/routes_{pair}.html  — Folium map, routes coloured by semiring,
                                   edges coloured by mode (road/bus/transfer)
  2. output/cost_comparison.png — bar chart, 4 semirings x 5 OD pairs
  3. output/runtime_scaling.png — Dijkstra vs Bellman-Ford runtime
"""

import os, math
import folium
import matplotlib.pyplot as plt
import networkx as nx

os.makedirs("output", exist_ok=True)

SEMIRING_COLOURS = {
    "distance":      "#1A56A0",
    "time":          "#1A6E3C",
    "fare":          "#B45309",
    "safety":        "#A32D2D",
    "lexicographic": "#6B3FA0",
    "transfers":     "#2E7D32",
}
SEMIRING_LABELS = {
    "distance":      "Shortest distance",
    "time":          "Fastest time",
    "fare":          "Cheapest fare (bus preferred)",
    "safety":        "Safest route",
    "lexicographic": "Time-first, fare tiebreak",
    "transfers":     "Fewest transfers (penalty 5×)",
}
MODE_COLOURS = {
    "road":     "#4A90D9",
    "bus":      "#27AE60",
    "transfer": "#E67E22",
}


def _path_coords(G, path):
    coords = []
    for n in path:
        if n in G.nodes and "y" in G.nodes[n] and "x" in G.nodes[n]:
            coords.append((G.nodes[n]["y"], G.nodes[n]["x"]))
    return coords


def _path_segments_by_mode(G, path):
    """Split a path into contiguous segments grouped by mode."""
    if len(path) < 2:
        return []
    segments, cur_mode, cur_coords = [], None, []

    for i in range(len(path)):
        node = path[i]
        if node not in G.nodes:
            continue
        lat = G.nodes[node].get("y")
        lon = G.nodes[node].get("x")
        if lat is None:
            continue

        if i == 0:
            cur_coords.append((lat, lon))
            continue

        prev = path[i - 1]
        mode = "road"
        if G.has_edge(prev, node):
            ed = G[prev][node]
            if isinstance(ed, dict) and 0 in ed:
                ed = ed[0]
            mode = ed.get("mode", "road")

        if mode != cur_mode:
            if cur_coords and cur_mode is not None:
                segments.append((cur_mode, cur_coords.copy()))
            cur_mode = mode
            cur_coords = [cur_coords[-1]] if cur_coords else []

        cur_coords.append((lat, lon))

    if cur_coords and cur_mode is not None:
        segments.append((cur_mode, cur_coords))
    return segments


def make_route_map(G, results, origin_name, dest_name,
                   source_node, target_node, filename):
    """
    Interactive Folium map showing:
      - Each semiring's route as a coloured polyline
      - Edge segments coloured by mode (road=blue, bus=green, walk=orange)
      - Bus stop markers along bus-mode segments
    """
    centre_lat = G.nodes[source_node]["y"]
    centre_lon = G.nodes[source_node]["x"]

    m = folium.Map(location=[centre_lat, centre_lon],
                   zoom_start=13, tiles="CartoDB positron")

    for name, r in results.items():
        path = r.get("path", [])
        if len(path) < 2:
            continue

        bd = r.get("breakdown", {})
        road_km = bd.get("road", {}).get("length_m", 0) / 1000
        bus_km  = bd.get("bus",  {}).get("length_m", 0) / 1000
        cost    = r.get("cost")
        if isinstance(cost, tuple):
            cost_str = "/".join(f"{c:.1f}" for c in cost) + f" {r.get('unit', '')}"
        elif cost is not None:
            cost_str = f"{cost:.1f} {r.get('unit', '')}"
        else:
            cost_str = "N/A"

        # Draw segments coloured by mode
        segments = _path_segments_by_mode(G, path)
        for mode, coords in segments:
            if len(coords) < 2:
                continue
            col = MODE_COLOURS.get(mode, "#888888")
            folium.PolyLine(
                coords,
                color=col,
                weight=5 if mode == "bus" else 3,
                opacity=0.80,
                tooltip=(
                    f"[{SEMIRING_LABELS[name]}] "
                    f"Mode: {mode}  Cost: {cost_str}"
                ),
                dash_array="8 4" if mode == "transfer" else None,
            ).add_to(m)

        # Outer highlight line in semiring colour (thin, underneath)
        all_coords = _path_coords(G, path)
        if all_coords:
            folium.PolyLine(
                all_coords,
                color=SEMIRING_COLOURS[name],
                weight=8,
                opacity=0.15,
            ).add_to(m)

        # Mark bus stop nodes along bus segments
        for mode, coords in segments:
            if mode == "bus":
                for lat, lon in coords[::3]:  # every 3rd point
                    folium.CircleMarker(
                        [lat, lon], radius=4,
                        color=MODE_COLOURS["bus"],
                        fill=True, fill_opacity=0.7,
                        tooltip="Bus stop",
                    ).add_to(m)

    # Origin / destination markers
    folium.Marker(
        [G.nodes[source_node]["y"], G.nodes[source_node]["x"]],
        popup=f"Origin: {origin_name}",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(m)
    folium.Marker(
        [G.nodes[target_node]["y"], G.nodes[target_node]["x"]],
        popup=f"Destination: {dest_name}",
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(m)

    # Legend
    legend = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:12px 16px;border-radius:8px;
                border:1px solid #ccc;font-family:sans-serif;font-size:12px">
      <b>Route (semiring)</b><br>
    """
    for name, col in SEMIRING_COLOURS.items():
        legend += (f'<span style="display:inline-block;width:20px;height:4px;'
                   f'background:{col};margin-right:6px;vertical-align:middle">'
                   f'</span>{SEMIRING_LABELS[name]}<br>')
    legend += "<br><b>Edge mode</b><br>"
    for mode, col in MODE_COLOURS.items():
        legend += (f'<span style="display:inline-block;width:12px;height:12px;'
                   f'background:{col};border-radius:2px;margin-right:6px;'
                   f'vertical-align:middle"></span>{mode}<br>')
    legend += "</div>"
    m.get_root().html.add_child(folium.Element(legend))

    out = f"output/{filename}.html"
    m.save(out)
    print(f"  Saved: {out}")
    return out


def plot_cost_comparison(all_results, filename="cost_comparison"):
    """
    Grid of bar charts — one per semiring (auto-sized).
    Key story: fare semiring dramatically lower for bus-using routes.
    """
    semirings  = list(SEMIRING_COLOURS.keys())
    pair_labels = [
        f"{r['origin_name'].split()[0]}→{r['dest_name'].split()[0]}"
        for r in all_results
    ]
    unit_labels = {
        "distance":      "metres",
        "time":          "seconds",
        "fare":          "Rs (approx)",
        "safety":        "safety score",
        "lexicographic": "seconds (primary)",
        "transfers":     "weighted score",
    }

    ncols = 3
    nrows = math.ceil(len(semirings) / ncols)
    fig, axes_2d = plt.subplots(nrows, ncols,
                                 figsize=(7 * ncols, 5 * nrows))
    fig.suptitle(
        "Route cost by semiring enrichment — Road+Bus intermodal graph, Bengaluru\n"
        "Same Dijkstra algorithm · Six semiring weight functions",
        fontsize=12, y=1.01
    )
    axes = axes_2d.flatten()

    for idx, name in enumerate(semirings):
        ax    = axes[idx]
        costs = []
        for r in all_results:
            c = r["results"].get(name, {}).get("cost")
            if c is None:
                costs.append(0)
            elif isinstance(c, tuple):
                costs.append(0 if any(math.isinf(x) for x in c) else c[0])
            else:
                costs.append(0 if math.isinf(c) else c)

        bars = ax.bar(range(len(pair_labels)), costs,
                      color=SEMIRING_COLOURS[name],
                      alpha=0.82, edgecolor="white", linewidth=0.5)
        ax.set_title(SEMIRING_LABELS[name], fontsize=11, pad=8)
        ax.set_xticks(range(len(pair_labels)))
        ax.set_xticklabels(pair_labels, rotation=20,
                           ha="right", fontsize=8)
        ax.set_ylabel(unit_labels.get(name, ""), fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)

        mx = max(costs) if costs else 1
        for bar, val in zip(bars, costs):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + mx*0.01,
                        f"{val:.1f}", ha="center",
                        va="bottom", fontsize=7)

    # Hide unused subplot cells
    for extra in axes[len(semirings):]:
        extra.set_visible(False)

    plt.tight_layout()
    out = f"output/{filename}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_runtime_scaling(scaling_results, filename="runtime_scaling"):
    """Runtime vs graph size: Dijkstra O((V+E)logV) vs Bellman-Ford O(VE)."""
    sizes   = sorted(scaling_results.keys())
    d_times = [scaling_results[n]["dijkstra_ms"] for n in sizes]
    bf_times= [scaling_results[n]["bellman_ford_ms"] for n in sizes]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sizes, d_times,  color=SEMIRING_COLOURS["distance"],
            marker="o", lw=2, ms=6, label="Dijkstra  O((V+E) log V)")
    ax.plot(sizes, bf_times, color=SEMIRING_COLOURS["safety"],
            marker="s", lw=2, ms=6, label="Bellman-Ford  O(V · E)")

    ax.set_xlabel("Graph size (nodes)", fontsize=11)
    ax.set_ylabel("Wall-clock time (ms)", fontsize=11)
    ax.set_title("Runtime scaling — Bengaluru intermodal subgraphs", fontsize=12)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.5)

    out = f"output/{filename}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_mode_breakdown(all_results, filename="mode_breakdown"):
    """
    Stacked bar chart: for each (OD pair × semiring),
    show km of road vs km of bus used.
    This is the clearest visual showing where Dijkstra
    switches from road to bus depending on semiring.
    """
    semirings = list(SEMIRING_COLOURS.keys())
    pair_labels = [
        f"{r['origin_name'].split()[0]}→{r['dest_name'].split()[0]}"
        for r in all_results
    ]

    fig, axes = plt.subplots(1, len(semirings),
                              figsize=(4 * len(semirings), 5),
                              sharey=True)
    fig.suptitle(
        "Road vs Bus km used — by semiring enrichment\n"
        "Fare semiring uses more bus; time semiring prefers road",
        fontsize=11
    )

    for ax, name in zip(axes, semirings):
        road_kms = []
        bus_kms  = []
        for r in all_results:
            bd = r["results"].get(name, {}).get("breakdown", {})
            road_kms.append(bd.get("road",     {}).get("length_m", 0) / 1000)
            bus_kms.append( bd.get("bus",      {}).get("length_m", 0) / 1000)

        x = range(len(pair_labels))
        ax.bar(x, road_kms, label="Road (auto)",
               color=MODE_COLOURS["road"], alpha=0.85)
        ax.bar(x, bus_kms, bottom=road_kms, label="Bus (BMTC)",
               color=MODE_COLOURS["bus"], alpha=0.85)

        ax.set_title(name, fontsize=10,
                     color=SEMIRING_COLOURS[name])
        ax.set_xticks(list(x))
        ax.set_xticklabels(pair_labels, rotation=30,
                           ha="right", fontsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ax == axes[0]:
            ax.set_ylabel("Distance (km)", fontsize=9)
            ax.legend(fontsize=8)

    plt.tight_layout()
    out = f"output/{filename}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
