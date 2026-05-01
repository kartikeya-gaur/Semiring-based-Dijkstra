"""
graph_setup.py — Build the Road+Bus intermodal graph for Bengaluru
===================================================================
Run once to download and save. Subsequent runs load from file.

    python graph_setup.py
"""

import os
import osmnx as ox
import networkx as nx
from bus_graph import (fetch_bus_data, build_bus_graph,
                        build_transfer_edges, build_multimodal_graph)

ROAD_FILE    = "data/bengaluru_road.graphml"
MULTIGRAPH_FILE = "data/bengaluru_multimodal.gpickle"

# Inner Bengaluru bounding box (south, west, north, east)
BBOX = (12.93, 77.55, 13.01, 77.65)

# 5 OD pairs (origin name, destination name)
OD_PAIRS_NAMES = [
    ("MG Road, Bengaluru",           "Koramangala 5th Block, Bengaluru"),
    ("Indiranagar 100ft Road, Bengaluru", "Whitefield, Bengaluru"),
    ("Jayanagar 4th Block, Bengaluru",    "Hebbal, Bengaluru"),
    ("Rajajinagar, Bengaluru",        "Banashankari, Bengaluru"),
    ("Ulsoor, Bengaluru",             "Silk Board Junction, Bengaluru"),
]


def load_road_graph(radius_m=5000):
    """Download road graph or load from file."""
    if os.path.exists(ROAD_FILE):
        print(f"Loading road graph from {ROAD_FILE}...")
        G = ox.load_graphml(ROAD_FILE)
    else:
        os.makedirs("data", exist_ok=True)
        print(f"Downloading road graph ({radius_m}m from MG Road)...")
        G = ox.graph_from_point(
            (12.9716, 77.5946),
            dist=radius_m,
            network_type="drive",
            simplify=True,
        )
        ox.save_graphml(G, ROAD_FILE)
        print(f"  Saved to {ROAD_FILE}")
    print(f"  Road: {G.number_of_nodes():,} nodes, "
          f"{G.number_of_edges():,} edges")
    return G


def build_or_load_multimodal(G_road):
    """Build road+bus intermodal graph, or load if already built."""
    import pickle

    if os.path.exists(MULTIGRAPH_FILE):
        print(f"Loading multimodal graph from {MULTIGRAPH_FILE}...")
        with open(MULTIGRAPH_FILE, "rb") as f:
            G = pickle.load(f)
        print(f"  Multimodal: {G.number_of_nodes():,} nodes, "
              f"{G.number_of_edges():,} edges")
        return G

    print("\nBuilding Road + Bus intermodal graph...")

    # Fetch bus data from OSM (with synthetic fallback)
    print("  Fetching bus stops and routes...")
    stops, routes = fetch_bus_data(BBOX)

    # Build bus sub-category
    print("  Building bus sub-graph G_b...")
    G_bus = build_bus_graph(stops, routes)

    # Build transfer edges G_t
    print("  Building transfer edges G_t...")
    transfers = build_transfer_edges(G_road, G_bus)

    # Compose: G = G_r ∪ G_b ∪ G_t
    print("  Composing intermodal graph G = G_r ∪ G_b ∪ G_t...")
    G = build_multimodal_graph(G_road, G_bus, transfers)

    # Save
    with open(MULTIGRAPH_FILE, "wb") as f:
        pickle.dump(G, f)
    print(f"  Saved to {MULTIGRAPH_FILE}")

    return G


def resolve_od_pairs(G, G_road):
    """Geocode OD pairs and find nearest nodes in the multimodal graph."""
    print("\nResolving OD pairs...")
    resolved = []

    for origin_name, dest_name in OD_PAIRS_NAMES:
        try:
            olat, olon = ox.geocode(origin_name)
            dlat, dlon = ox.geocode(dest_name)

            # Snap to nearest road node; wrap as (space, state) tuple
            src = (ox.nearest_nodes(G_road, olon, olat), 'road')
            tgt = (ox.nearest_nodes(G_road, dlon, dlat), 'road')

            if src not in G.nodes or tgt not in G.nodes:
                print(f"  Skip (nodes not in multimodal graph): "
                      f"{origin_name} → {dest_name}")
                continue
            if src == tgt:
                print(f"  Skip (same node): {origin_name}")
                continue

            resolved.append({
                "origin_name": origin_name.split(",")[0],
                "dest_name":   dest_name.split(",")[0],
                "source":      src,
                "target":      tgt,
            })
            print(f"  OK: {origin_name.split(',')[0]} → "
                  f"{dest_name.split(',')[0]}")

        except Exception as e:
            print(f"  Failed ({e}): {origin_name} → {dest_name}")

    print(f"\n  {len(resolved)}/{len(OD_PAIRS_NAMES)} pairs ready")
    return resolved


if __name__ == "__main__":
    G_road = load_road_graph()
    G      = build_or_load_multimodal(G_road)
    pairs  = resolve_od_pairs(G, G_road)
    print("\nSetup complete. Run main.py to start experiments.")
