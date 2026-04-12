"""
metro_graph.py — Namma Metro network (Bengaluru)
=================================================
42 stations, Purple Line + Green Line.
Hardcoded coordinates are stable (metro doesn't change like bus OSM data).

CT Framing (ACT4E §14.1):
    This is the Metro Sub-Category G_m.
    Objects  = metro stations (node IDs prefixed 'm_')
    Morphisms = consecutive station pairs on the same line
    Mode     = 'metro'

Transfer edges connect G_m to G_r (nearest road intersection).
Metro stations are generally closer to major roads than bus stops,
so max_walk for metro transfers is set to 600m.
"""

import math
import networkx as nx

# ── Metro constants ──────────────────────────────────────────────────
METRO_SPEED_KMH   = 32.0    # Namma Metro average speed (incl. dwell)
METRO_WAIT_SEC    = 300.0   # 5-minute average wait (trains every 10 min)
METRO_WALK_M      = 600.0   # max walk to metro station

# Fare: Namma Metro flat + distance-based
# Rs 10 for <2km, Rs 15 2-4km, Rs 20 4-6km, Rs 25 6-8km, Rs 30 8-10km, Rs 40 10-15km, Rs 50 15-20km, Rs 60 >20km
# Approximate as linear: Rs 45 base + Rs 4/km
METRO_BASE_FARE   = 45.0
METRO_FARE_PER_M  = 4.0 / 1000   # Rs 4/km

METRO_SAFETY      = 0.95    # highest safety — enclosed, controlled

def _metro_node_id(station_id: int) -> str:
    return f"m_{station_id}"

def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))


# ── Station data ─────────────────────────────────────────────────────
# (station_id, lat, lon, name, line)
# Purple Line: East-West  |  Green Line: North-South
# Source: BMRCL public data + OpenStreetMap

METRO_STATIONS = [
    # Purple Line (West → East)
    (101, 12.9784, 77.5726, "Baiyappanahalli",      "Purple"),
    (102, 12.9801, 77.5643, "Swami Vivekananda Road","Purple"),
    (103, 12.9812, 77.5580, "Indiranagar",           "Purple"),
    (104, 12.9816, 77.5508, "Halasuru",              "Purple"),
    (105, 12.9817, 77.5427, "Trinity",               "Purple"),
    (106, 12.9761, 77.5715, "Mahatma Gandhi Road",   "Purple"),
    (107, 12.9766, 77.5648, "Cubbon Park",           "Purple"),
    (108, 12.9782, 77.5571, "Vidhana Soudha",        "Purple"),
    (109, 12.9764, 77.5493, "Sir M Visvesvaraya",    "Purple"),
    (110, 12.9749, 77.5411, "Nadaprabhu Kempegowda", "Purple"),  # Majestic interchange
    (111, 12.9722, 77.5330, "City Railway Station",  "Purple"),
    (112, 12.9663, 77.5249, "Magadi Road",           "Purple"),
    (113, 12.9594, 77.5199, "Hosahalli",             "Purple"),
    (114, 12.9533, 77.5148, "Vijayanagar",           "Purple"),
    (115, 12.9471, 77.5097, "Attiguppe",             "Purple"),
    (116, 12.9414, 77.5047, "Deepanjali Nagar",      "Purple"),
    (117, 12.9350, 77.4993, "Mysuru Road",           "Purple"),

    # Green Line (North → South)
    (201, 13.0481, 77.5576, "Nagasandra",            "Green"),
    (202, 13.0385, 77.5593, "Dasarahalli",           "Green"),
    (203, 13.0299, 77.5605, "Jalahalli",             "Green"),
    (204, 13.0221, 77.5607, "Peenya Industry",       "Green"),
    (205, 13.0170, 77.5559, "Peenya",                "Green"),
    (206, 13.0095, 77.5516, "Goraguntepalya",        "Green"),
    (207, 13.0011, 77.5501, "Yeshwanthpur",          "Green"),
    (208, 12.9929, 77.5490, "Sandal Soap Factory",   "Green"),
    (209, 12.9867, 77.5499, "Mahalakshmi",           "Green"),
    (210, 12.9800, 77.5496, "Rajajinagar",           "Green"),
    (211, 12.9727, 77.5496, "Kuvempu Road",          "Green"),
    (110, 12.9749, 77.5411, "Nadaprabhu Kempegowda", "Green"),  # interchange
    (212, 12.9671, 77.5418, "Chickpete",             "Green"),
    (213, 12.9609, 77.5399, "Krishna Rajendra Market","Green"),
    (214, 12.9538, 77.5381, "National College",      "Green"),
    (215, 12.9460, 77.5367, "Lalbagh",               "Green"),
    (216, 12.9381, 77.5371, "South End Circle",      "Green"),
    (217, 12.9307, 77.5374, "Jayanagar",             "Green"),
    (218, 12.9226, 77.5358, "Rashtreeya Vidyalaya Road","Green"),
    (219, 12.9155, 77.5358, "Banashankari",          "Green"),
    (220, 12.9071, 77.5369, "Jaya Prakash Nagar",    "Green"),
    (221, 12.8992, 77.5393, "Yelachenahalli",        "Green"),
]

# ── Line sequences ────────────────────────────────────────────────────
PURPLE_LINE = [101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117]
GREEN_LINE  = [201,202,203,204,205,206,207,208,209,210,211,110,212,213,214,215,216,217,218,219,220,221]


def build_metro_graph() -> nx.DiGraph:
    """
    Build the Metro Sub-Category G_m.
    Nodes: station IDs prefixed 'm_', with lat/lon/name attributes.
    Edges: consecutive station pairs on each line, bidirectional.
    """
    G_m = nx.DiGraph()
    station_lookup = {}

    # Add all unique stations (station_id → attrs)
    for sid, lat, lon, name, line in METRO_STATIONS:
        if sid not in station_lookup:
            station_lookup[sid] = {"lat": lat, "lon": lon, "name": name, "lines": []}
        if line not in station_lookup[sid]["lines"]:
            station_lookup[sid]["lines"].append(line)

    for sid, attrs in station_lookup.items():
        G_m.add_node(_metro_node_id(sid),
                     y=attrs["lat"], x=attrs["lon"],
                     name=attrs["name"],
                     lines=attrs["lines"],
                     node_type="metro_station",
                     osm_id=sid)

    # Add edges along each line
    edges_added = 0
    for line_name, sequence in [("Purple", PURPLE_LINE), ("Green", GREEN_LINE)]:
        for i in range(len(sequence) - 1):
            s1_id, s2_id = sequence[i], sequence[i+1]
            if s1_id not in station_lookup or s2_id not in station_lookup:
                continue
            s1, s2 = station_lookup[s1_id], station_lookup[s2_id]
            dist_m = _haversine_m(s1["lat"], s1["lon"], s2["lat"], s2["lon"])

            n1, n2 = _metro_node_id(s1_id), _metro_node_id(s2_id)
            attrs = {
                "mode":      "metro",
                "length":    dist_m,
                "line":      line_name,
                "highway":   "metro_rail",
            }
            G_m.add_edge(n1, n2, **attrs)
            G_m.add_edge(n2, n1, **attrs)
            edges_added += 2

    print(f"    Metro graph: {G_m.number_of_nodes()} stations, "
          f"{edges_added} directed segments "
          f"(Purple={len(PURPLE_LINE)-1} stops, Green={len(GREEN_LINE)-1} stops)")
    return G_m


def build_metro_transfer_edges(G_road: nx.DiGraph,
                                G_metro: nx.DiGraph,
                                max_walk_m: float = METRO_WALK_M) -> list:
    """
    Build transfer edges between metro stations and nearest road intersections.
    Same logic as bus transfer edges — walking morphisms between sub-categories.
    """
    transfers = []
    road_coords = [
        (G_road.nodes[n]['y'], G_road.nodes[n]['x'], n)
        for n in G_road.nodes
        if 'y' in G_road.nodes[n]
    ]

    for m_node in G_metro.nodes:
        mdata = G_metro.nodes[m_node]
        mlat, mlon = mdata.get('y'), mdata.get('x')
        if mlat is None:
            continue

        best_dist, best_road = float('inf'), None
        for rlat, rlon, rnode in road_coords:
            d = _haversine_m(mlat, mlon, rlat, rlon)
            if d < best_dist:
                best_dist, best_road = d, rnode

        if best_road and best_dist <= max_walk_m:
            attrs = {"mode": "transfer", "length": best_dist, "highway": "footway"}
            transfers.append((best_road, m_node, attrs))
            transfers.append((m_node, best_road, attrs))

    print(f"    Metro transfer edges: {len(transfers)//2} station-intersection pairs")
    return transfers
