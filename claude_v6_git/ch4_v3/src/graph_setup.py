"""
graph_setup.py — Build G = G_r ∪ G_b ∪ G_m ∪ G_t
===================================================
Road + Bus + Metro + Transfer edges composed into one intermodal DiGraph.
"""

import os, pickle
import osmnx as ox
import networkx as nx

from bus_graph   import fetch_bus_data, build_bus_graph, build_transfer_edges
from metro_graph import build_metro_graph, build_metro_transfer_edges

ROAD_FILE   = "data/bengaluru_road.graphml"
MULTI_FILE  = "data/bengaluru_multimodal.pkl"
BBOX        = (12.88, 77.49, 13.05, 77.70)   # broader bbox for metro coverage

OD_PAIRS = [
    ("MG Road, Bengaluru",                "Koramangala 5th Block, Bengaluru"),
    ("Indiranagar 100ft Road, Bengaluru", "Whitefield, Bengaluru"),
    ("Jayanagar 4th Block, Bengaluru",    "Hebbal, Bengaluru"),
    ("Rajajinagar, Bengaluru",            "Banashankari, Bengaluru"),
    ("Ulsoor, Bengaluru",                 "Silk Board Junction, Bengaluru"),
]


def load_road_graph(radius_m=6000):
    if os.path.exists(ROAD_FILE):
        print(f"  Loading road graph from {ROAD_FILE}...")
        G = ox.load_graphml(ROAD_FILE)
    else:
        os.makedirs("data", exist_ok=True)
        print(f"  Downloading road graph ({radius_m}m from MG Road)...")
        G = ox.graph_from_point((12.9716, 77.5946), dist=radius_m,
                                 network_type="drive", simplify=True)
        ox.save_graphml(G, ROAD_FILE)
    print(f"  Road: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


def build_multimodal(G_road):
    """Compose G = G_r ∪ G_b ∪ G_m ∪ G_t"""
    print("\n  Building intermodal graph G = G_r ∪ G_b ∪ G_m ∪ G_t...")

    # Bus sub-graph
    print("  [Bus] Fetching BMTC stops and routes...")
    stops, routes = fetch_bus_data(BBOX)
    G_bus = build_bus_graph(stops, routes)
    bus_transfers = build_transfer_edges(G_road, G_bus)

    # Metro sub-graph
    print("  [Metro] Building Namma Metro graph (42 stations hardcoded)...")
    G_metro = build_metro_graph()
    metro_transfers = build_metro_transfer_edges(G_road, G_metro)

    # Compose
    G = nx.DiGraph()

    # Road edges (mode='road')
    for n, d in G_road.nodes(data=True):
        G.add_node(n, **d, node_type="road")
    for u, v, d in G_road.edges(data=True):
        d2 = dict(d); d2.setdefault("mode","road")
        if not G.has_edge(u, v):
            G.add_edge(u, v, **d2)

    # Bus edges (mode='bus')
    for n, d in G_bus.nodes(data=True):  G.add_node(n, **d)
    for u, v, d in G_bus.edges(data=True): G.add_edge(u, v, **d)

    # Metro edges (mode='metro')
    for n, d in G_metro.nodes(data=True): G.add_node(n, **d)
    for u, v, d in G_metro.edges(data=True): G.add_edge(u, v, **d)

    # Transfer edges (mode='transfer')
    for u, v, d in bus_transfers + metro_transfers:
        if u in G.nodes and v in G.nodes:
            G.add_edge(u, v, **d)

    r = sum(1 for _,_,d in G.edges(data=True) if d.get("mode")=="road")
    b = sum(1 for _,_,d in G.edges(data=True) if d.get("mode")=="bus")
    m = sum(1 for _,_,d in G.edges(data=True) if d.get("mode")=="metro")
    t = sum(1 for _,_,d in G.edges(data=True) if d.get("mode")=="transfer")
    print(f"\n  Intermodal graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    print(f"    road={r:,}  bus={b:,}  metro={m:,}  transfer={t:,}")
    return G


def load_or_build_multimodal(G_road):
    if os.path.exists(MULTI_FILE):
        print(f"  Loading multimodal graph from {MULTI_FILE}...")
        with open(MULTI_FILE,"rb") as f: G = pickle.load(f)
        print(f"  Loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
        return G
    G = build_multimodal(G_road)
    os.makedirs("data", exist_ok=True)
    with open(MULTI_FILE,"wb") as f: pickle.dump(G, f)
    print(f"  Saved to {MULTI_FILE}")
    return G


def resolve_od_pairs(G, G_road):
    print("\n  Resolving OD pairs...")
    resolved = []
    for orig, dest in OD_PAIRS:
        try:
            olat, olon = ox.geocode(orig)
            dlat, dlon = ox.geocode(dest)
            src = ox.nearest_nodes(G_road, olon, olat)
            tgt = ox.nearest_nodes(G_road, dlon, dlat)
            if src in G.nodes and tgt in G.nodes and src != tgt:
                resolved.append({
                    "origin_name": orig.split(",")[0],
                    "dest_name":   dest.split(",")[0],
                    "source": src, "target": tgt,
                })
                print(f"  OK: {orig.split(',')[0]} → {dest.split(',')[0]}")
        except Exception as e:
            print(f"  Skip ({e}): {orig}")
    print(f"  {len(resolved)}/{len(OD_PAIRS)} pairs resolved")
    return resolved


if __name__ == "__main__":
    Gr = load_road_graph()
    G  = load_or_build_multimodal(Gr)
    resolve_od_pairs(G, Gr)
    print("Setup complete. Run main.py.")
