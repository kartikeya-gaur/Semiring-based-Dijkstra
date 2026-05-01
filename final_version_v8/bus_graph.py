"""
bus_graph.py — Build BMTC bus route graph from OpenStreetMap
=============================================================
Fetches bus stops and route relations from OSM via the Overpass API,
then builds a NetworkX graph representing the bus network.

CT Framing (ACT4E §14.1):
    This is the Bus Category G_b.
    Objects  = bus stop nodes (OSM node IDs, prefixed with 'b_')
    Morphisms = consecutive stop pairs on the same BMTC route
    Composition = chaining segments along a route

    G_b is a sub-category of the full intermodal graph, exactly
    mirroring ACT4E's "subway graph G_s = ⟨V_s, E_s, src_s, tgt_s⟩".

Transfer edges connect G_b to G_r (the road category):
    Each bus stop gets a 'transfer' edge to the nearest road
    intersection. This is the dashed intermodal morphism in
    ACT4E Fig. 4 — the edge that lets you switch modes.
"""

import math
import time
import networkx as nx

# ── Bus mode constants ───────────────────────────────────────────────
BUS_SPEED_KMH   = 18.0   # avg BMTC bus speed in Bengaluru (traffic-adjusted)
BUS_WAIT_SEC    = 600.0  # avg bus waiting time: 10 minutes
WALK_SPEED_KMH  = 5.0    # walking speed for transfer edges
MAX_WALK_M      = 400.0  # max walk distance to a bus stop (metres)
BUS_FARE_PER_M  = 1.5 / 1000  # ₹1.5 per km → per metre

# Bus stop safety: buses are inherently safer than being in a car
# (passenger, not driver; larger vehicle; fixed route)
BUS_SAFETY_SCORE    = 0.90
WALK_SAFETY_SCORE   = 0.88

# ── Node ID convention ───────────────────────────────────────────────
# Road graph nodes are integers (OSM node IDs).
# Bus stop nodes are prefixed with 'b_' to avoid collisions.
# e.g. OSM node 123456 as a road intersection → 123456
#      OSM node 123456 as a bus stop           → 'b_123456'


def _bus_node_id(osm_id: int) -> tuple:
    """
    Convert OSM node ID to a (space, state) bus node.
    e.g. stop 9001  →  (9001, 'bus')
    The mode tag in the tuple disambiguates from road node (9001, 'road').
    """
    return (osm_id, 'bus')


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Haversine distance between two lat/lon points in metres."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.asin(math.sqrt(a))


# ────────────────────────────────────────────────────────────────────
# Fetch from OpenStreetMap via Overpass API
# ────────────────────────────────────────────────────────────────────

def fetch_bus_data(bbox: tuple) -> tuple[list, list]:
    """
    Fetch BMTC bus stops and route relations from OSM.

    bbox: (south, west, north, east) in decimal degrees
    Returns:
        stops:  list of dicts {osm_id, lat, lon, name}
        routes: list of dicts {route_id, name, stop_ids: [osm_id,...]}

    Falls back to synthetic data if Overpass API is unavailable.
    """
    try:
        import overpy
        return _fetch_from_overpass(bbox)
    except ImportError:
        print("    overpy not installed. Run: pip install overpy")
        print("    Falling back to synthetic bus network...")
        return _synthetic_bus_data(bbox)
    except Exception as e:
        print(f"    Overpass API error: {e}")
        print("    Falling back to synthetic bus network...")
        return _synthetic_bus_data(bbox)


def _fetch_from_overpass(bbox: tuple) -> tuple[list, list]:
    """Fetch real BMTC data from OpenStreetMap Overpass API."""
    import overpy
    api = overpy.Overpass()
    s, w, n, e = bbox

    print("    Querying Overpass API for BMTC bus stops...")
    # Fetch bus stops in bbox
    stop_query = f"""
    [out:json][timeout:30];
    (
      node["highway"="bus_stop"]({s},{w},{n},{e});
      node["public_transport"="stop_position"]["bus"="yes"]({s},{w},{n},{e});
    );
    out body;
    """
    stop_result = api.query(stop_query)
    time.sleep(1)  # be polite to Overpass

    stops = []
    for node in stop_result.nodes:
        stops.append({
            "osm_id": node.id,
            "lat":    float(node.lat),
            "lon":    float(node.lon),
            "name":   node.tags.get("name", f"Stop_{node.id}"),
        })
    print(f"    Found {len(stops)} bus stops")

    # Fetch BMTC route relations
    print("    Querying Overpass API for BMTC routes...")
    route_query = f"""
    [out:json][timeout:45];
    (
      relation["route"="bus"]["operator"~"BMTC|Bangalore",i]({s},{w},{n},{e});
      relation["route"="bus"]["network"~"BMTC",i]({s},{w},{n},{e});
    );
    out tags members;
    """
    try:
        route_result = api.query(route_query)
        time.sleep(1)

        routes = []
        stop_ids_set = {s["osm_id"] for s in stops}

        for relation in route_result.relations:
            route_stops = []
            for member in relation.members:
                if (hasattr(member, 'ref') and
                        member.role in ('stop', 'stop_exit_only',
                                        'stop_entry_only', '') and
                        member.ref in stop_ids_set):
                    route_stops.append(member.ref)

            if len(route_stops) >= 2:
                routes.append({
                    "route_id": relation.id,
                    "name":     relation.tags.get("name", f"Route_{relation.id}"),
                    "stop_ids": route_stops,
                })

        print(f"    Found {len(routes)} BMTC routes "
              f"with stops in area")

    except Exception as e:
        print(f"    Route fetch failed ({e}). Using stops only.")
        routes = []

    # If no routes found, synthesise connectivity from proximity
    if not routes:
        print("    No route relations found. "
              "Building proximity-based bus network...")
        routes = _routes_from_proximity(stops, max_dist_m=600)

    return stops, routes


def _routes_from_proximity(stops: list, max_dist_m: float = 600) -> list:
    """
    Fallback: connect bus stops that are within max_dist_m of each other.
    This approximates bus corridors when route relations are missing.
    Each stop becomes a mini-route connecting to its 2 nearest neighbours.
    """
    if not stops:
        return []

    routes = []
    for i, s1 in enumerate(stops):
        neighbours = []
        for j, s2 in enumerate(stops):
            if i == j:
                continue
            d = _haversine_m(s1["lat"], s1["lon"], s2["lat"], s2["lon"])
            if d <= max_dist_m:
                neighbours.append((d, s2["osm_id"]))

        neighbours.sort()
        # Connect to 2 nearest neighbours as a mini-route
        for _, nbr_id in neighbours[:2]:
            routes.append({
                "route_id": f"prox_{s1['osm_id']}_{nbr_id}",
                "name":     f"Bus_{s1['osm_id']}",
                "stop_ids": [s1["osm_id"], nbr_id],
            })

    return routes


def _synthetic_bus_data(bbox: tuple) -> tuple[list, list]:
    """
    Synthetic BMTC bus network for when OSM is unavailable.
    Uses real Bengaluru landmark coordinates.
    """
    print("    Building synthetic bus network from Bengaluru landmarks...")
    s, w, n, e = bbox
    centre_lat = (s + n) / 2
    centre_lon = (w + e) / 2

    # Major BMTC bus stops (approximate coordinates)
    raw_stops = [
        (9001, 12.9784, 77.5738, "Majestic Bus Stand"),
        (9002, 12.9756, 77.6011, "MG Road"),
        (9003, 12.9352, 77.6245, "Koramangala BDA Complex"),
        (9004, 12.9783, 77.6408, "Indiranagar 100ft Road"),
        (9005, 12.9165, 77.6101, "Jayanagar 4th Block"),
        (9006, 12.9698, 77.7499, "Whitefield"),
        (9007, 13.0354, 77.5970, "Hebbal Flyover"),
        (9008, 12.9279, 77.5539, "Banashankari"),
        (9009, 12.9538, 77.5955, "Lalbagh"),
        (9010, 12.9499, 77.6190, "Silk Board"),
        (9011, 12.9850, 77.5533, "Rajajinagar"),
        (9012, 12.9902, 77.5596, "Vijayanagar"),
        (9013, 13.0050, 77.5665, "Yeshwanthpur"),
        (9014, 12.9592, 77.6974, "Marathahalli"),
        (9015, 12.9122, 77.5938, "BTM Layout"),
    ]

    stops = []
    for osm_id, lat, lon, name in raw_stops:
        # Only include stops within the bbox
        if s <= lat <= n and w <= lon <= e:
            stops.append({"osm_id": osm_id, "lat": lat,
                          "lon": lon, "name": name})

    if not stops:
        # If bbox excludes all landmarks, place stops near centre
        stops = [
            {"osm_id": 9001, "lat": centre_lat + 0.01,
             "lon": centre_lon - 0.02, "name": "Stop A"},
            {"osm_id": 9002, "lat": centre_lat,
             "lon": centre_lon, "name": "Stop B (Central)"},
            {"osm_id": 9003, "lat": centre_lat - 0.01,
             "lon": centre_lon + 0.02, "name": "Stop C"},
        ]

    # Synthetic routes connecting nearby stops
    routes = _routes_from_proximity(stops, max_dist_m=3000)

    print(f"    Synthetic network: {len(stops)} stops, {len(routes)} segments")
    return stops, routes


# ────────────────────────────────────────────────────────────────────
# Build the bus sub-category graph
# ────────────────────────────────────────────────────────────────────

def build_bus_graph(stops: list, routes: list) -> nx.DiGraph:
    """
    Build the Bus Category G_b as a NetworkX DiGraph.

    Nodes: bus stop node IDs ('b_{osm_id}'), with lat/lon/name attributes
    Edges: directed segments between consecutive stops on a route,
           annotated with mode='bus', length, and route info

    Edges are added in both directions (most bus routes are bidirectional).
    """
    G_b = nx.DiGraph()

    # Build stop lookup by OSM ID
    stop_lookup = {s["osm_id"]: s for s in stops}

    # Add stop nodes
    for s in stops:
        node_id = _bus_node_id(s["osm_id"])
        G_b.add_node(
            node_id,
            y=s["lat"],
            x=s["lon"],
            name=s["name"],
            node_type="bus_stop",
            osm_id=s["osm_id"],
        )

    # Add route segment edges
    edges_added = 0
    for route in routes:
        stop_ids = route["stop_ids"]
        for i in range(len(stop_ids) - 1):
            s1_id = stop_ids[i]
            s2_id = stop_ids[i + 1]

            if s1_id not in stop_lookup or s2_id not in stop_lookup:
                continue

            s1 = stop_lookup[s1_id]
            s2 = stop_lookup[s2_id]
            dist_m = _haversine_m(s1["lat"], s1["lon"],
                                   s2["lat"], s2["lon"])

            if dist_m < 10:   # skip duplicate/coincident stops
                continue

            n1 = _bus_node_id(s1_id)
            n2 = _bus_node_id(s2_id)

            edge_attrs = {
                "mode":     "bus",
                "length":   dist_m,
                "route_id": route["route_id"],
                "route_name": route.get("name", ""),
                "highway":  "bus_route",
            }

            G_b.add_edge(n1, n2, **edge_attrs)
            G_b.add_edge(n2, n1, **edge_attrs)   # bidirectional
            edges_added += 2

    print(f"    Bus graph: {G_b.number_of_nodes()} stops, "
          f"{edges_added} directed segments")
    return G_b


# ────────────────────────────────────────────────────────────────────
# Transfer edges: G_r ↔ G_b  (the intermodal morphisms)
# ────────────────────────────────────────────────────────────────────

def build_transfer_edges(G_road: nx.MultiDiGraph,
                         G_bus: nx.DiGraph,
                         max_walk_m: float = MAX_WALK_M) -> list[tuple]:
    """
    For each bus stop in G_bus, find the nearest road intersection
    in G_road and create a bidirectional 'transfer' edge between them.

    Transfer edge semantics:
        weight = walking distance (metres)
        mode   = 'transfer'
        Plus a fixed bus wait penalty applied in the time weight function.

    Returns:
        list of (road_node, bus_node, attrs) tuples.
        Add these to the combined graph after merging G_r and G_b.

    CT Note (ACT4E §14.1):
        These are the "dashed arrows" representing intermodal morphisms —
        the edges that let you switch from one sub-category to another.
    """
    transfers = []
    road_nodes = list(G_road.nodes)

    # Build list of (lat, lon, node_id) for road nodes once
    road_coords = [
        (G_road.nodes[n]['y'], G_road.nodes[n]['x'], n)
        for n in road_nodes
        if 'y' in G_road.nodes[n] and 'x' in G_road.nodes[n]
    ]

    for bus_node in G_bus.nodes:
        bdata = G_bus.nodes[bus_node]
        blat, blon = bdata.get('y'), bdata.get('x')
        if blat is None or blon is None:
            continue

        # Find nearest road node (brute force — fine for <10k nodes)
        best_dist = float('inf')
        best_road_node = None

        for rlat, rlon, rnode in road_coords:
            d = _haversine_m(blat, blon, rlat, rlon)
            if d < best_dist:
                best_dist = d
                best_road_node = rnode

        if best_road_node is None or best_dist > max_walk_m:
            continue

        attrs = {
            "mode":    "transfer",
            "length":  best_dist,
            "highway": "footway",
        }

        # (space, state) node IDs: road side is (int_id, 'road'),
        # bus side is already (osm_id, 'bus') from _bus_node_id.
        transfers.append(((best_road_node, 'road'), bus_node, attrs))
        transfers.append((bus_node, (best_road_node, 'road'), attrs))

    print(f"    Transfer edges: {len(transfers)//2} stop-intersection pairs "
          f"within {max_walk_m}m walk")
    return transfers


# ────────────────────────────────────────────────────────────────────
# Compose into the full intermodal multigraph
# ────────────────────────────────────────────────────────────────────

def build_multimodal_graph(G_road: nx.MultiDiGraph,
                            G_bus: nx.DiGraph,
                            transfers: list[tuple]) -> nx.DiGraph:
    """
    Compose G_r, G_b, and transfer edges into one DiGraph.

    G = G_r  ∪  G_b  ∪  G_t

    This is the full intermodal category from ACT4E §14.1,
    instantiated for Bengaluru with Road and Bus sub-categories.

    Notes:
        - Road edges get mode='road' if not already set
        - Bus edges keep mode='bus'
        - Transfer edges get mode='transfer'
        - The combined graph is a plain DiGraph (not MultiDiGraph)
          for simpler Dijkstra implementation
    """
    G = nx.DiGraph()

    # Add road nodes and edges — wrap as (osm_id, 'road') (space, state) tuples
    for node, data in G_road.nodes(data=True):
        G.add_node((node, 'road'), **data, node_type="road")

    for u, v, data in G_road.edges(data=True):
        # osmnx MultiDiGraph may have parallel edges; take first
        edge_data = dict(data)
        edge_data.setdefault("mode", "road")
        if not G.has_edge((u, 'road'), (v, 'road')):
            G.add_edge((u, 'road'), (v, 'road'), **edge_data)

    road_nodes = G_road.number_of_nodes()
    road_edges = G.number_of_edges()

    # Add bus nodes and edges (mode='bus')
    for node, data in G_bus.nodes(data=True):
        G.add_node(node, **data)

    for u, v, data in G_bus.edges(data=True):
        G.add_edge(u, v, **data)

    bus_nodes  = G_bus.number_of_nodes()
    bus_edges  = G_bus.number_of_edges()

    # Add transfer edges (mode='transfer')
    for u, v, attrs in transfers:
        if u in G.nodes and v in G.nodes:
            G.add_edge(u, v, **attrs)

    print(f"\n    Intermodal graph G = G_r ∪ G_b ∪ G_t:")
    print(f"      Road nodes:     {road_nodes:>6,}")
    print(f"      Bus stop nodes: {bus_nodes:>6,}")
    print(f"      Total nodes:    {G.number_of_nodes():>6,}")
    print(f"      Road edges:     {road_edges:>6,}")
    print(f"      Bus edges:      {bus_edges:>6,}")
    print(f"      Transfer edges: {len(transfers):>6,}")
    print(f"      Total edges:    {G.number_of_edges():>6,}")

    return G
