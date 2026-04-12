# v3: Road + Bus + Metro Multimodal Routing

**References:** CLRS §24 · ACT4E §14.1/14.3/14.6

## Install
```bash
pip install osmnx networkx numpy matplotlib folium overpy
```

## Run
```bash
cd src
python main.py
```
- Namma Metro (42 stations, Purple + Green line) added as G_m
- 5 query types: road / bus+walk / metro+walk / bus+metro+walk / full multimodal
- Semiring vs Pareto front analysis with citations (Mohri 2002, Martins 1984, Hansen 1980)
- Redesigned visualisations: fixed mode colors, core insight per chart

## Mode color palette
| Mode      | Color   | Meaning |
|-----------|---------|---------|
| Road      | #334155 | Auto/car — slate |
| Bus       | #16A34A | BMTC — green |
| Metro     | #7C3AED | Namma Metro — purple (brand) |
| Walk      | #EA580C | Transfer/walking — orange |

## 5 Query types
| Query | Graph | Weight | Question answered |
|-------|-------|--------|-------------------|
| (a) Road only | G_r | time | Fastest auto route |
| (b) Bus+walk | G_b∪G_t | fare | Cheapest bus journey |
| (c) Metro+walk | G_m∪G_t | fare | Metro journey |
| (d) Bus+metro+walk | G_b∪G_m∪G_t | fare | Best transit-only |
| (e) Full multimodal | G_r∪G_b∪G_m∪G_t | fare | Globally optimal |

## Semiring vs n-dimensional weight vectors
Traditional multi-criteria routing computes the Pareto front
all non-dominated paths across k criteria. This is NP-hard for k≥2
(Martins 1984) and the front can have exponential size O(2^k) (Hansen 1980).

Semiring routing runs k separate Dijkstra queries O(k×(V+E)logV) total
(Mohri 2002) and each result is guaranteed to lie on the Pareto front.
For a fixed decision context (e.g. "minimise fare"), semiring gives the
exact optimal answer in polynomial time with no additional overhead.

## Key citations
- Mohri 2002: "Semiring Frameworks and Algorithms for Shortest-Distance Problems"
- Martins 1984: "On a multicriteria shortest path problem" (NP-hardness)
- Hansen 1980: "Bicriterion path problems" (exponential Pareto size)
- Gondran & Minoux 1984: "Graphs and Algorithms" (dioid/semiring generalisation)
- CLRS §24.3: Dijkstra correctness over tropical semiring
- ACT4E §14.1: Intermodal city category
- ACT4E §14.3: Currency categories = enriched categories
- ACT4E §14.6: CT advantage: one proof covers all enrichments
