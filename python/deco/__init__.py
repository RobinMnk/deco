"""Deco: expander decomposition, congestion approximator and normalized cut, on the CPU or with CUDA.

    import deco

    g = deco.Graph.from_mtx("graph.mtx")
    d = deco.expander_decomposition(g, sparsity=0.3)
    d.cluster_of, d.num_clusters, d.cut_edges
"""

from __future__ import annotations

import math
import operator
import os
from dataclasses import dataclass
from importlib.metadata import version

import numpy as np

from . import _core

__version__ = version("deco-graph")

__all__ = [
    "Graph",
    "Clustering",
    "Decomposition",
    "Level",
    "NormalizedCut",
    "Sparsifier",
    "cuda_available",
    "expander_decomposition",
    "sparsifier",
    "normalized_cut",
]

_MAX_INDEX = np.iinfo(np.uint32).max


class Graph:
    """An undirected graph, converted once and reused by every computation.

    Create it with Graph.from_mtx or Graph.from_scipy. `weights` is "unweighted", "integer" or "real".
    """

    def __init__(self, graph: _core.Graph):
        self._graph = graph

    @classmethod
    def from_mtx(cls, path: str | os.PathLike) -> Graph:
        """Reads a Matrix Market file (.mtx) with Deco's reader, like the command-line tool."""
        path = os.fspath(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"no such file: {path!r}")
        return cls(_core.Graph.from_mtx(path))

    @classmethod
    def from_scipy(cls, matrix, weighted: bool | None = None) -> Graph:
        """The graph of a square SciPy sparse matrix: an edge for every nonzero entry off the diagonal.

        A symmetric matrix gives one edge per node pair, any other matrix the graph A + A^T, in which every entry
        is an edge (the same rule as for "general" .mtx files). The diagonal is dropped.
        Weights: if all values are 1 the graph is unweighted, if all are whole numbers it has integer weights,
        otherwise real weights. weighted=False ignores the values, weighted=True keeps them even if all are 1.
        """
        import scipy.sparse   # optional dependency: pip install deco-graph[scipy]

        if not scipy.sparse.issparse(matrix):
            raise TypeError(f"expected a SciPy sparse matrix, got {type(matrix).__name__}")
        n, cols = matrix.shape
        if n != cols:
            raise ValueError(f"the matrix must be square, got shape {matrix.shape}")
        if n >= _MAX_INDEX:
            raise ValueError("too many nodes")

        a = matrix.tocsr()   # no copy if it is CSR already
        if not a.has_canonical_format:   # duplicate entries (they add up) or unsorted columns
            a = a.copy()
            a.sum_duplicates()
        data = a.data
        if data.dtype.kind not in "biuf":
            raise TypeError(f"the matrix entries must be real numbers, got {data.dtype}")
        if a.nnz > _MAX_INDEX:
            raise ValueError("too many edges")

        rows = np.repeat(np.arange(n, dtype=np.uint32), np.diff(a.indptr))
        off_diagonal = rows != a.indices
        if data.dtype.kind == "f" and not (np.isfinite(data) | ~off_diagonal).all():
            raise ValueError("edge weights must be finite")
        if ((data < 0) & off_diagonal).any():
            raise ValueError("edge weights must be nonnegative")
        keep = off_diagonal & (data != 0)
        if keep.all():
            indptr, indices = a.indptr, a.indices
        else:
            rows, indices, data = rows[keep], a.indices[keep], data[keep]
            indptr = np.concatenate(([0], np.cumsum(np.bincount(rows, minlength=n))))
        b = scipy.sparse.csr_array((data, indices, indptr), shape=(n, n))
        symmetric = (b != b.T).nnz == 0

        if weighted is None:
            weighted = not (data == 1).all()
        if not weighted:
            kind, weights = "unweighted", ()
        elif _whole_numbers(data):
            kind, weights = "integer", (data.astype(np.int64, copy=False),)
        else:
            kind, weights = "real", (data.astype(np.float64, copy=False),)

        indices = indices.astype(np.uint32)
        if symmetric:   # the matrix is the adjacency structure: every edge is in the rows of both of its nodes
            graph = getattr(_core.Graph, f"from_{kind}_csr")(n, indptr.astype(np.uint32), indices, *weights)
        else:           # every entry is an edge
            graph = getattr(_core.Graph, f"from_{kind}_edges")(n, rows, indices, *weights)
        return cls(graph)

    @property
    def num_nodes(self) -> int:
        return self._graph.num_nodes

    @property
    def num_edges(self) -> int:
        return self._graph.num_edges

    @property
    def weights(self) -> str:
        return self._graph.weights

    def __repr__(self) -> str:
        return f"Graph(num_nodes={self.num_nodes}, num_edges={self.num_edges}, weights={self.weights!r})"


def _whole_numbers(data: np.ndarray) -> bool:
    """Whether all values are whole numbers that fit 64-bit integers."""
    if data.size == 0:
        return True
    if data.dtype.kind == "f":
        return bool((data == np.floor(data)).all()) and data.max() < 2.0**63
    return data.max() <= np.iinfo(np.int64).max


@dataclass(eq=False)
class Clustering:
    """A partition of the nodes: cluster_of[v] is the cluster of node v, numbered 0 .. num_clusters - 1.

    cut_edges counts the edges between different clusters, cut_weight is their total weight.
    """

    cluster_of: np.ndarray
    num_clusters: int
    cut_edges: int
    cut_weight: int | float


@dataclass(eq=False)
class Decomposition(Clustering):
    """An expander decomposition. backend is where it ran ("cpu" or "cuda"), seconds its running time."""

    backend: str
    iterations: int
    seconds: float


@dataclass(eq=False)
class NormalizedCut(Clustering):
    """A partition into k clusters. normalized_cut is its value: the sum over all clusters of the weight of the edges
    leaving the cluster divided by its volume."""

    normalized_cut: float


@dataclass(eq=False)
class Level:
    """One level of a congestion approximator.

    parents[v] is the cluster on this level of node v of the level below (on level 0: of the graph).
    volumes and cut_weights have one value per cluster: its volume, and the weight of the edges leaving it.
    seconds is the time of the expander decomposition that produced the level.
    """

    parents: np.ndarray
    volumes: np.ndarray
    cut_weights: np.ndarray
    seconds: float


class Sparsifier:
    """A congestion approximator (tree cut sparsifier): the clusters of repeated expander decompositions.

    It does not change: use it for any number of predictions and normalized cuts.
    """

    def __init__(self, core: _core.Sparsifier, graph: Graph):
        self._sparsifier = core
        self._graph = graph   # the graph it was built from, which normalized_cut partitions
        self.levels: list[Level] = [Level(**level) for level in core.levels()]

    def predict(self, demand) -> float:
        """The predicted congestion of routing `demand`, one value per node, summing to 0."""
        demand = np.ascontiguousarray(demand, dtype=np.float64)
        if demand.shape != (self._sparsifier.num_nodes,):
            raise ValueError(f"the demand needs one value per node ({self._sparsifier.num_nodes}), got shape {demand.shape}")
        if not np.isfinite(demand).all():
            raise ValueError("the demand must be finite")
        return self._sparsifier.predict(demand)

    def __repr__(self) -> str:
        return f"Sparsifier(levels={len(self.levels)})"


def cuda_available() -> bool:
    """Whether the CUDA backend can run: Deco was built with CUDA and a CUDA device is present."""
    return _core.cuda_available()


def expander_decomposition(
    graph: Graph,
    *,
    sparsity: float = _core.DEFAULT_SPARSITY,
    walk_threshold: float = _core.DEFAULT_WALK_THRESHOLD,
    seed: int = _core.DEFAULT_SEED,
    backend: str = "auto",
) -> Decomposition:
    """Expander decomposition: clusters without a sweep cut sparser than `sparsity`, and few edges between them.

    walk_threshold: a cluster is done once its random-walk values differ by less than this.
    seed: seed for the random walks.
    backend: "auto" (CUDA if available and the graph is supported, otherwise the CPU), "cpu" or "cuda".
    """
    result = _core.expander_decomposition(_graph(graph), *_options(sparsity, walk_threshold, seed, backend))
    return Decomposition(**result)


def sparsifier(
    graph: Graph,
    *,
    sparsity: float = _core.DEFAULT_SPARSITY,
    walk_threshold: float = _core.DEFAULT_WALK_THRESHOLD,
    seed: int = _core.DEFAULT_SEED,
    backend: str = "auto",
) -> Sparsifier:
    """Congestion approximator (expander hierarchy): repeated expander decompositions of the contracted graph.

    The options are those of expander_decomposition.
    """
    return Sparsifier(_core.sparsifier(_graph(graph), *_options(sparsity, walk_threshold, seed, backend)), graph)


def normalized_cut(sparsifier: Sparsifier, k: int) -> NormalizedCut:
    """Partition of the graph of `sparsifier` into k clusters with a small normalized cut.

    The result has the clusters and the normalized cut value. The sparsifier does not change, so one sparsifier
    serves any number of calls, for example with different k.
    """
    if not isinstance(sparsifier, Sparsifier):
        raise TypeError(f"expected a deco.Sparsifier (deco.normalized_cut(deco.sparsifier(graph), k)), "
                        f"got {type(sparsifier).__name__}")
    graph = sparsifier._graph
    k = operator.index(k)
    if not 1 <= k <= graph.num_nodes:
        raise ValueError(f"k must be from 1 to the number of nodes ({graph.num_nodes}), got {k}")
    return NormalizedCut(**_core.normalized_cut(graph._graph, sparsifier._sparsifier, k))


def _graph(graph: Graph) -> _core.Graph:
    if not isinstance(graph, Graph):
        raise TypeError(f"expected a deco.Graph, got {type(graph).__name__}")
    return graph._graph


def _options(sparsity: float, walk_threshold: float, seed: int, backend: str) -> tuple[float, float, int, str]:
    sparsity, walk_threshold, seed = float(sparsity), float(walk_threshold), operator.index(seed)
    if not (sparsity > 0 and math.isfinite(sparsity)):
        raise ValueError(f"sparsity must be a positive number, got {sparsity}")
    if not (walk_threshold > 0 and math.isfinite(walk_threshold)):
        raise ValueError(f"walk_threshold must be a positive number, got {walk_threshold}")
    if not 0 <= seed <= _MAX_INDEX:
        raise ValueError(f"seed must be from 0 to 2^32 - 1, got {seed}")
    if backend not in ("auto", "cpu", "cuda"):
        raise ValueError(f"backend must be 'auto', 'cpu' or 'cuda', got {backend!r}")
    return sparsity, walk_threshold, seed, backend
