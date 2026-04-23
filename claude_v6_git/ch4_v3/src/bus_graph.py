"""
bus_graph.py — BMTC bus network from OSM (unchanged from v2)
See previous version for full implementation.
"""
import math, time, networkx as nx

BUS_SPEED_KMH=18.0; BUS_WAIT_SEC=600.0; WALK_SPEED_KMH=5.0
MAX_WALK_M=400.0; BUS_FARE_PER_M=1.5/1000; BUS_SAFETY=0.90; WALK_SAFETY=0.88

def _bus_node_id(osm_id): return f"b_{osm_id}"
def _haversine_m(la1,lo1,la2,lo2):
    R=6371000; p1,p2=math.radians(la1),math.radians(la2)
    dp=math.radians(la2-la1); dl=math.radians(lo2-lo1)
    a=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

def fetch_bus_data(bbox):
    try:
        import overpy
        return _fetch_overpass(bbox)
    except Exception as e:
        print(f"    Overpass unavailable ({e}). Using synthetic bus network.")
        return _synthetic(bbox)

def _fetch_overpass(bbox):
    import overpy, time as t
    api=overpy.Overpass(); s,w,n,e=bbox
    try:
        r=api.query(f'[out:json][timeout:30];(node["highway"="bus_stop"]({s},{w},{n},{e});node["public_transport"="stop_position"]["bus"="yes"]({s},{w},{n},{e}););out body;')
        t.sleep(1)
        stops=[{"osm_id":nd.id,"lat":float(nd.lat),"lon":float(nd.lon),"name":nd.tags.get("name",f"Stop_{nd.id}")} for nd in r.nodes]
        print(f"    Found {len(stops)} bus stops from OSM")
    except Exception: stops=[]
    if not stops: return _synthetic(bbox)
    routes=_routes_from_proximity(stops,600)
    return stops, routes

def _routes_from_proximity(stops,max_d=600):
    routes=[]
    for i,s1 in enumerate(stops):
        nbrs=sorted([((_haversine_m(s1["lat"],s1["lon"],s2["lat"],s2["lon"])),s2["osm_id"]) for j,s2 in enumerate(stops) if i!=j])
        for d,nid in nbrs[:2]:
            if d<=max_d: routes.append({"route_id":f"p_{s1['osm_id']}_{nid}","name":"Bus","stop_ids":[s1["osm_id"],nid]})
    return routes

def _synthetic(bbox):
    s,w,n,e=bbox; clat=(s+n)/2; clon=(w+e)/2
    raw=[(9001,12.9784,77.5738,"Majestic"),(9002,12.9756,77.6011,"MG Road"),
         (9003,12.9352,77.6245,"Koramangala"),(9004,12.9783,77.6408,"Indiranagar"),
         (9005,12.9165,77.6101,"Jayanagar"),(9006,12.9698,77.7499,"Whitefield"),
         (9007,13.0354,77.5970,"Hebbal"),(9008,12.9279,77.5539,"Banashankari"),
         (9009,12.9538,77.5955,"Lalbagh"),(9010,12.9499,77.6190,"Silk Board"),
         (9011,12.9850,77.5533,"Rajajinagar"),(9012,12.9663,77.5249,"Magadi Road"),]
    stops=[{"osm_id":oid,"lat":lat,"lon":lon,"name":nm} for oid,lat,lon,nm in raw if s<=lat<=n and w<=lon<=e]
    if not stops: stops=[{"osm_id":9001,"lat":clat+0.01,"lon":clon-0.02,"name":"Stop A"},{"osm_id":9002,"lat":clat,"lon":clon,"name":"Stop B"},{"osm_id":9003,"lat":clat-0.01,"lon":clon+0.02,"name":"Stop C"}]
    routes=_routes_from_proximity(stops,3000)
    print(f"    Synthetic bus: {len(stops)} stops, {len(routes)} segments")
    return stops, routes

def build_bus_graph(stops,routes):
    G=nx.DiGraph(); sl={s["osm_id"]:s for s in stops}
    for s in stops: G.add_node(_bus_node_id(s["osm_id"]),y=s["lat"],x=s["lon"],name=s["name"],node_type="bus_stop",osm_id=s["osm_id"])
    ea=0
    for route in routes:
        for i in range(len(route["stop_ids"])-1):
            s1i,s2i=route["stop_ids"][i],route["stop_ids"][i+1]
            if s1i not in sl or s2i not in sl: continue
            s1,s2=sl[s1i],sl[s2i]; d=_haversine_m(s1["lat"],s1["lon"],s2["lat"],s2["lon"])
            if d<10: continue
            a={"mode":"bus","length":d,"route_id":route["route_id"],"highway":"bus_route"}
            G.add_edge(_bus_node_id(s1i),_bus_node_id(s2i),**a)
            G.add_edge(_bus_node_id(s2i),_bus_node_id(s1i),**a); ea+=2
    print(f"    Bus graph: {G.number_of_nodes()} stops, {ea} segments")
    return G

def build_transfer_edges(G_road,G_bus,max_walk_m=MAX_WALK_M):
    transfers=[]; rc=[(G_road.nodes[n]['y'],G_road.nodes[n]['x'],n) for n in G_road.nodes if 'y' in G_road.nodes[n]]
    for bn in G_bus.nodes:
        bd=G_bus.nodes[bn]; blat,blon=bd.get('y'),bd.get('x')
        if blat is None: continue
        best_d,best_r=float('inf'),None
        for rlat,rlon,rn in rc:
            d=_haversine_m(blat,blon,rlat,rlon)
            if d<best_d: best_d,best_r=d,rn
        if best_r and best_d<=max_walk_m:
            a={"mode":"transfer","length":best_d,"highway":"footway"}
            transfers.append((best_r,bn,a)); transfers.append((bn,best_r,a))
    print(f"    Bus transfer edges: {len(transfers)//2} pairs within {max_walk_m}m")
    return transfers
