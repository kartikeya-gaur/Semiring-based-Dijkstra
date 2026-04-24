# Semiring-based-Dijkstra
The goal is to implement a Multimodal Route Planner that uses categorical abstractions specifically Enriched Categories and Semirings to unify different routing objectives under a single algorithmic framework.

# Motivation
The motivation for transitioning toward enriched categories in multimodal routing stems from the inherent limitations of traditional graph-based optimization. While Pareto-optimal sets are mathematically sound, they are effectively "useless at scale" because they grow exponentially as more criteria are added. Even advanced algorithmic frameworks like McRAPTOR and BM-RAPTOR struggle to overcome this; they face a significant technical bottleneck where parallelization provides a meager 1.7 times speedup. Some state-of-art algorithams delivers the near-instantaneous (roughly 30ms) response times required for real-world applications but only for 2-3 criterias, what if we are to expand.

The explosion of Pareto frontiers under multiple criteria isn't just a computational problem — it's a sign that the algebraic structure being used (real-valued scalar costs) is the wrong enriching category. A richer algebraic structure (a tropical semiring, a lattice-ordered monoid, or a preordered commutative monoid) might represent multi-criteria costs in a way that admits efficient computation of "optimal" paths without full Pareto enumeration.

# Semi-ring based approach
We have decoupled the Topology (the graph) from the Algebra (the semiring), the algorithm itself never changes. If you want the top 3 fastest, the top 3 best-value, or a diverse mix of trade-offs, you do not touch the Dijkstra loop. You simply change the sorting rule inside the + operator of your semiring. This is what makes our proposed framework a "Structure Maintaining Paradigm" rather than a rigid set of hardcoded instructions.
