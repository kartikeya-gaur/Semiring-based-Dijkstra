"""
category_demo.py — The Road Graph as a Free Category
=====================================================
This file makes the CT claim EXECUTABLE, not just cited.

It demonstrates, on a real 10-node subgraph of Bengaluru:

    1. A Category class (objects, morphisms, compose)
    2. The road graph IS a free category (closing under composition)
    3. Dijkstra finds the "optimal morphism" in the tropical enrichment
    4. The hom-set hom(A,B) = {cost of best path from A to B}

ACT4E §12.3: Definition of a category
ACT4E §12.5: Free category on a graph
ACT4E §14.3: Enriched category over the tropical semiring
"""

import math
import networkx as nx
from dijkstra import dijkstra, reconstruct_path, w_distance, w_fare


# ════════════════════════════════════════════════════════════════════
# A minimal Category implementation
# ════════════════════════════════════════════════════════════════════

class Category:
    """
    A category C consists of:
        Objects:    a set Ob(C)
        Morphisms:  for each pair (A,B), a set Hom(A,B)
        Composition:f: A→B, g: B→C  →  g∘f: A→C
        Identities: id_A: A→A for every A

    Laws:
        Associativity: h∘(g∘f) = (h∘g)∘f
        Unit:          id_B ∘ f = f = f ∘ id_A
    """
    def __init__(self, name="C"):
        self.name    = name
        self.objects = set()
        self._hom    = {}   # (A, B) → list of morphisms

    def add_object(self, A):
        self.objects.add(A)
        # Every object has an identity morphism (cost 0)
        key = (A, A)
        if key not in self._hom:
            self._hom[key] = [{"label": f"id_{A}", "cost": 0.0, "path": [A]}]

    def add_morphism(self, A, B, label, cost, path=None):
        key = (A, B)
        self._hom.setdefault(key, [])
        self._hom[key].append({
            "label": label,
            "cost":  cost,
            "path":  path or [A, B],
        })

    def hom(self, A, B):
        """All morphisms from A to B."""
        return self._hom.get((A, B), [])

    def optimal_morphism(self, A, B):
        """The morphism in hom(A,B) with minimum cost (tropical best)."""
        candidates = self.hom(A, B)
        if not candidates:
            return None
        return min(candidates, key=lambda m: m["cost"])

    def compose(self, f, g):
        """
        f: A→B,  g: B→C  →  g∘f: A→C
        Tropical composition: cost(g∘f) = cost(f) + cost(g)  [⊗ = +]
        """
        assert f["path"][-1] == g["path"][0], \
            f"Cannot compose: {f['path'][-1]} ≠ {g['path'][0]}"
        return {
            "label": f"{g['label']}∘{f['label']}",
            "cost":  f["cost"] + g["cost"],   # ← tropical ⊗
            "path":  f["path"] + g["path"][1:],
        }

    def close_under_composition(self, max_morphisms_per_pair=5):
        """
        Compute the free category transitive closure.
        Uses Floyd-Warshall style: for each intermediate node k,
        update all (i,j) pairs via i->k->j.
        Single O(n^3) pass — guaranteed to terminate.

        max_morphisms_per_pair: keep only cheapest N paths per pair
        to prevent exponential blowup on dense graphs.
        """
        objects = list(self.objects)
        for k in objects:
            for A in objects:
                if A == k: continue
                f_list = self._hom.get((A, k), [])
                if not f_list: continue
                for C in objects:
                    if C == A: continue
                    g_list = self._hom.get((k, C), [])
                    if not g_list: continue
                    # Best composition only (cheapest f + cheapest g)
                    best_f = min(f_list, key=lambda m: m["cost"])
                    best_g = min(g_list, key=lambda m: m["cost"])
                    comp = self.compose(best_f, best_g)
                    key  = (A, C)
                    existing = self._hom.get(key, [])
                    # Only add if cheaper than existing best, or new pair
                    if not existing or comp["cost"] < min(m["cost"] for m in existing):
                        self._hom[key] = [comp]  # keep only best

    def print_hom_sets(self, top_k=3):
        """Print hom-sets (up to top_k morphisms per pair, by cost)."""
        print(f"\n  Category: {self.name}")
        print(f"  Objects: {len(self.objects)}")
        total_morphisms = sum(len(v) for v in self._hom.values())
        print(f"  Morphisms (after closure): {total_morphisms}")
        print()
        # Show interesting pairs (non-identity, non-empty)
        shown = 0
        for (A, B), morphs in sorted(self._hom.items()):
            if A == B: continue
            if not morphs: continue
            best = sorted(morphs, key=lambda m: m["cost"])[:top_k]
            print(f"  hom({A[:6] if isinstance(A,str) else A}, "
                  f"{B[:6] if isinstance(B,str) else B})  "
                  f"[{len(morphs)} paths]")
            for m in best:
                print(f"    cost={m['cost']:.1f}  hops={len(m['path'])-1}")
            shown += 1
            if shown >= 6: break


# ════════════════════════════════════════════════════════════════════
# Build the road graph as a free category
# ════════════════════════════════════════════════════════════════════

def road_graph_as_category(G, source, n_nodes=12):
    """
    Take a BFS subgraph of n_nodes from source.
    Represent it as a Category where:
        Objects   = road intersections
        Morphisms = direct road edges (generating morphisms)
    Then close under composition to get all paths.

    Returns (cat, nodes) where cat is the Category after closure.
    """
    bfs    = list(nx.bfs_tree(G, source).nodes)[:n_nodes]
    H      = G.subgraph(bfs).copy()
    # Road edges only for clean demo
    H_road = nx.DiGraph()
    for n in bfs:
        if n in H: H_road.add_node(n)
    for u, v, d in H.edges(data=True):
        if d.get("mode","road") == "road":
            H_road.add_edge(u, v, **d)

    nodes = [n for n in bfs if n in H_road.nodes][:n_nodes]

    cat = Category(name="BengaluruRoad (10 nodes)")
    for n in nodes:
        cat.add_object(n)

    # Add generating morphisms (direct edges)
    for u, v, d in H_road.edges(data=True):
        if u in nodes and v in nodes:
            dist = float(d.get("length", 50.0))
            cat.add_morphism(u, v,
                             label=f"road_{u}_{v}",
                             cost=dist,
                             path=[u, v])

    print(f"  Before closure: {sum(len(v) for v in cat._hom.values())} morphisms")
    # Note: full closure is exponential for large graphs.
    # For demo we show the concept on n<=12 nodes.
    cat.close_under_composition()
    print(f"  After closure:  {sum(len(v) for v in cat._hom.values())} morphisms")

    return cat, nodes


def dijkstra_as_optimal_morphism(G, source, target, nodes):
    """
    Run Dijkstra and show: the result IS the optimal morphism
    in the tropical-enriched road category.

    optimal_morphism(source, target).cost
        == Dijkstra dist[target]

    This is the categorical interpretation of Dijkstra:
    it finds argmin over all morphisms in hom(source, target).
    """
    dist, prev = dijkstra(G, source, w_distance)
    path = reconstruct_path(prev, source, target)
    dijkstra_cost = dist[target]

    print(f"\n  Dijkstra optimal morphism:")
    print(f"    source → target cost: {dijkstra_cost:.1f}m")
    print(f"    path hops: {len(path)-1}")
    print(f"    path: {' → '.join(str(n)[:8] for n in path[:4])}{'...' if len(path)>4 else ''}")

    return dijkstra_cost, path


def run_category_demo(G, source, target):
    """Run the full category demo. Call from main.py."""
    print("\n" + "="*65)
    print("  Category Demo: Road Graph as a Free Category")
    print("  ACT4E §12.3 (categories) + §12.5 (free category) + §14.3 (enriched)")
    print("="*65)

    print("""
  WHAT THIS DEMONSTRATES:
  ──────────────────────────────────────────────────────────────
  1. The road graph is a DIRECTED MULTIGRAPH (ACT4E §11.2)
  2. Closing it under path composition + adding identity loops
     produces a FREE CATEGORY (ACT4E §12.5)
  3. Equipping the hom-sets with costs (tropical semiring) gives
     a TROPICAL-ENRICHED CATEGORY (ACT4E §14.3)
  4. Dijkstra computes the OPTIMAL MORPHISM: argmin of hom(A,B)
  ──────────────────────────────────────────────────────────────""")

    print("\n  Building free category on 10-node road subgraph...")
    cat, nodes = road_graph_as_category(G, source, n_nodes=10)
    cat.print_hom_sets(top_k=2)

    if target in nodes:
        tgt = target
    else:
        tgt = nodes[-1]

    print(f"\n  Comparing Category optimal morphism vs Dijkstra:")
    dijk_cost, _ = dijkstra_as_optimal_morphism(G, source, tgt, nodes)

    opt = cat.optimal_morphism(source, tgt)
    if opt:
        cat_cost = opt["cost"]
        match = abs(cat_cost - dijk_cost) < 0.01
        print(f"\n  Category optimal morphism cost: {cat_cost:.1f}m")
        print(f"  Dijkstra dist[target]:          {dijk_cost:.1f}m")
        print(f"  Match: {'YES ✓ — Dijkstra IS computing the optimal morphism' if match else 'DIFF (subgraph mismatch)'}")
    else:
        print(f"  (target not reachable within 10-node subgraph)")

    print(f"""
  CONCLUSION:
  ──────────────────────────────────────────────────────────────
  Free category has {sum(len(v) for v in cat._hom.values())} morphisms from 10 objects.
  Dijkstra selects the optimal one from hom(source, target)
  without enumerating all of them — O((V+E) log V) vs O(|hom|).
  This is the meaning of "Dijkstra computes the tropical closure"
  (ACT4E §14.3, Mohri 2002 §3).
  ──────────────────────────────────────────────────────────────""")

    return cat
