"""
semirings.py — Abstract Semiring interface + concrete implementations
====================================================================
CT Framing (ACT4E §14.3):
    A semiring (S, ⊕, ⊗, 0, 1) generalises the tropical semiring.
    Dijkstra/Bellman-Ford work over *any* semiring; only the weight
    algebra changes, never the algorithm.

    SafetySemiring     — (R, max, min)  bottleneck / reliability
    LexicographicSemiring — (R², lex, +)  primary time, tie-break fare
"""

from abc import ABC, abstractmethod


class Semiring(ABC):
    """
    Abstract semiring interface.

    Methods
    -------
    combine(a, b) : path-extension operator  (analogous to + in tropical)
    better(a, b)  : True when `a` is strictly preferred over `b`
    zero()        : identity for "no path" / worst possible value
    one()         : identity element for combine (source node cost)
    """

    @abstractmethod
    def combine(self, a, b):
        """Extend path cost `a` by edge cost `b`."""

    @abstractmethod
    def better(self, a, b):
        """Return True iff `a` is strictly better than `b`."""

    @abstractmethod
    def zero(self):
        """Absorbing element — represents an unreachable / worst state."""

    @abstractmethod
    def one(self):
        """Identity for combine — cost of the empty path (source)."""


# ── Concrete semirings ───────────────────────────────────────────────

class SafetySemiring(Semiring):
    """
    Bottleneck-safety semiring  (R, max, min).

    The cost of a path is the *minimum* safety score along it
    (worst link dominates). We maximise this minimum — i.e., we
    prefer paths whose weakest segment is as safe as possible.

    zero()  = 0      — worst possible safety (unreachable sentinel)
    one()   = +∞     — identity for min; drops out of combine(one, x) = x
    """

    def combine(self, a, b):
        return min(a, b)   # worst safety along path

    def better(self, a, b):
        return a > b       # maximize safety

    def zero(self):
        return 0           # worst safety

    def one(self):
        return float('inf')  # start with best safety


class LexicographicSemiring(Semiring):
    """
    Lexicographic (time, fare) semiring  (R², lex, +).

    Costs are (time_seconds, fare_rupees) pairs.
    Paths are added component-wise; comparison is lexicographic:
    minimise time first, break ties by minimising fare.

    zero() = (∞, ∞)  — unreachable sentinel
    one()  = (0, 0)  — empty path has zero time and zero fare
    """

    def combine(self, a, b):
        return (a[0] + b[0], a[1] + b[1])

    def better(self, a, b):
        if a[0] != b[0]:
            return a[0] < b[0]   # prioritize time
        return a[1] < b[1]       # tie-break with fare

    def zero(self):
        return (float('inf'), float('inf'))

    def one(self):
        return (0, 0)


class TransferSemiring(Semiring):
    """
    Transfer-penalised semiring  (R³, weighted-sum-lex, +).

    Costs are (time_seconds, fare_rupees, transfer_count) triples.
    Paths are added component-wise; comparison uses a weighted score
    that heavily penalises each mode switch:

        score = time + fare + 5 * transfers

    This biases the algorithm toward through-routes with fewer
    road↔bus transitions, even at a moderate time/fare premium.

    zero() = (∞, ∞, ∞)  — unreachable sentinel
    one()  = (0, 0, 0)  — empty path costs nothing
    """

    def combine(self, a, b):
        return (
            a[0] + b[0],   # time
            a[1] + b[1],   # fare
            a[2] + b[2],   # transfers
        )

    def better(self, a, b):
        # penalize transfers heavily
        score_a = a[0] + a[1] + 5 * a[2]
        score_b = b[0] + b[1] + 5 * b[2]
        return score_a < score_b

    def zero(self):
        return (float('inf'), float('inf'), float('inf'))

    def one(self):
        return (0, 0, 0)
