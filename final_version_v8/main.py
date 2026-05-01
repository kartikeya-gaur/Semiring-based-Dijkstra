"""
main.py — Run the full Road+Bus intermodal routing project
==========================================================
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from graph_setup  import load_road_graph, build_or_load_multimodal, resolve_od_pairs
from dijkstra     import run_all_semirings, print_results_table
from bellman_ford import compare_runtimes, run_negative_weight_demo
from visualise    import (make_route_map, plot_cost_comparison,
                           plot_runtime_scaling, plot_mode_breakdown)


def main():
    print("\n" + "="*65)
    print("  Chapter 4: Multimodal Routing — Road + Bus, Bengaluru")
    print("  Dijkstra over 4 semirings on the intermodal graph")
    print("="*65)

    # 1. Load graphs
    print("\n[1/5] Loading road graph...")
    G_road = load_road_graph()

    print("\n[2/5] Building Road+Bus intermodal graph...")
    G = build_or_load_multimodal(G_road)

    # 2. Resolve OD pairs
    print("\n[3/5] Resolving OD pairs...")
    pairs = resolve_od_pairs(G, G_road)
    if not pairs:
        print("ERROR: No OD pairs resolved. Check internet / geocoding.")
        return

    # 3. Run all 4 semirings on all pairs
    print("\n[4/5] Running Dijkstra (4 semirings) on all OD pairs...")
    all_results = []
    for pair in pairs:
        results = run_all_semirings(G, pair["source"], pair["target"])
        print_results_table(results, pair["origin_name"], pair["dest_name"])
        all_results.append({**pair, "results": results})

    # 4. Bellman-Ford
    # print("\n[5/5] Bellman-Ford comparison + negative-weight demo...")
    # first = pairs[0]
    # print(f"\n  Runtime scaling ({first['origin_name']} → {first['dest_name']}):")
    # scaling = compare_runtimes(G, first["source"], first["target"],
    #                             sizes=[50, 100, 200, 300, 500])

    # print("\n  Negative-weight demo (toll subsidy corridor):")
    # run_negative_weight_demo(G, first["source"], first["target"])

    # 5. Visualisations
    print("\n  Generating visualisations...")
    for r in all_results[:2]:
        safe = r["origin_name"].split()[0].lower()
        make_route_map(G, r["results"],
                       r["origin_name"], r["dest_name"],
                       r["source"], r["target"],
                       filename=f"routes_{safe}")

    plot_cost_comparison(all_results)
    plot_mode_breakdown(all_results)
    if scaling:
        plot_runtime_scaling(scaling)

    print("\n" + "="*65)
    print("  Output files:")
    print("    output/routes_*.html      (open in browser)")
    print("    output/cost_comparison.png")
    print("    output/mode_breakdown.png  ← NEW: road vs bus km per semiring")
    print("    output/runtime_scaling.png")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
