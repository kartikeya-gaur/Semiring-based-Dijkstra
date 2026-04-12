"""
main.py — Chapter 4 v3 (final): All components
================================================
[1] Build G = G_r ∪ G_b ∪ G_m ∪ G_t
[2] Dijkstra x 6 semirings (+ comfort, + congestion)
[3] 5 query types
[4] Category demo: road graph as free category
[5] Semiring analysis: Pareto + runtime table + trip breakdown
[6] Tropical matrix convergence
[7] Visualisations
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import dijkstra as dijk_module
from graph_setup       import load_road_graph, load_or_build_multimodal, resolve_od_pairs
from dijkstra          import run_all_semirings, print_results_table
from query_types       import run_all_queries
from category_demo     import run_category_demo
from semiring_analysis import (run_semiring_analysis, empirical_runtime_table,
                                compute_trip_breakdown, print_trip_breakdown_table)
from visualise         import (make_route_map, plot_fare_savings, plot_time_vs_fare,
                                plot_mode_split, plot_semiring_vs_pareto,
                                plot_cost_breakdown, plot_tropical_convergence,
                                plot_semiring_runtime)


def main():
    print("\n" + "="*68)
    print("  Chapter 4 v3: Road + Bus + Metro Multimodal Routing")
    print("  Bengaluru  |  6 Semirings  |  5 Query Types  |  CT Deepening")
    print("="*68)

    # [1] Graphs
    print("\n[1/7] Loading graphs...")
    G_road = load_road_graph()
    G      = load_or_build_multimodal(G_road)
    pairs  = resolve_od_pairs(G, G_road)
    if not pairs:
        print("ERROR: No OD pairs resolved."); return
    first = pairs[0]

    # [2] Dijkstra x 6 semirings
    print("\n[2/7] Dijkstra -- 6 semiring weight functions...")
    dijk_module.IS_PEAK_HOUR = True
    print("  Peak-hour congestion: ON")
    all_semiring_results = []
    for p in pairs:
        r = run_all_semirings(G, p["source"], p["target"])
        print_results_table(r, p["origin_name"], p["dest_name"])
        all_semiring_results.append({**p, "results": r})

    # [3] 5 query types
    print("\n[3/7] 5 query types...")
    all_query_results = []
    for p in pairs:
        print(f"\n  -- {p['origin_name']} -> {p['dest_name']} --")
        qr = run_all_queries(G, p["source"], p["target"],
                             p["origin_name"], p["dest_name"])
        all_query_results.append({
            "pair_label": f"{p['origin_name'].split()[0]}->{p['dest_name'].split()[0]}",
            "results": qr,
        })

    # [4] Category demo
    print("\n[4/7] Category demo: road graph as free category...")
    run_category_demo(G, first["source"], first["target"])

    # [5] Semiring analysis
    print("\n[5/7] Semiring analysis...")
    sem_analysis = run_semiring_analysis(G, first["source"], first["target"])

    print("\n  Empirical runtime table:")
    runtime_data = empirical_runtime_table(G, first["source"],
                                            sizes=[100, 300, 500, 1000])

    print("\n  Trip breakdown:")
    breakdown_rows = compute_trip_breakdown(all_query_results)
    print_trip_breakdown_table(breakdown_rows)

    # [6] Tropical convergence
    print("\n[6/7] Tropical matrix convergence...")
    plot_tropical_convergence(G, first["source"], n_nodes=20)

    # [7] Visualisations
    print("\n[7/7] Generating visualisations...")
    for r in all_semiring_results[:2]:
        safe = r["origin_name"].split()[0].lower()
        make_route_map(G, r["results"], r["origin_name"], r["dest_name"],
                       r["source"], r["target"], filename=f"routes_{safe}")

    plot_fare_savings(all_query_results)
    plot_time_vs_fare(all_query_results)
    plot_mode_split(all_query_results)
    plot_semiring_vs_pareto(sem_analysis.get("pareto_size", {}))
    plot_cost_breakdown(breakdown_rows)
    plot_semiring_runtime(runtime_data)

    print("\n" + "="*68)
    for f in ["routes_*.html","fare_savings.png","time_vs_fare.png",
              "mode_split.png","semiring_vs_pareto.png","cost_breakdown.png",
              "tropical_convergence.png","semiring_runtime.png"]:
        print(f"    output/{f}")
    print("="*68 + "\n")


if __name__ == "__main__":
    main()
