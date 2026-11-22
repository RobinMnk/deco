#
# Tests of the Python package: reading and converting graphs, the three computations, and errors. The test graphs are
# those of the C++ tests (tests/data): two 5-cliques joined by one edge, unweighted, with integer and with real weights.
# The CUDA cases need a CUDA device and are skipped without one.
#

import json
import shutil
import subprocess
import sysconfig
import threading
from pathlib import Path

import numpy as np
import pytest
import scipy.io
import scipy.sparse as sp

import deco

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
TWO_CLIQUES = {   # file -> weights, cut weight of the edge between the cliques, normalized cut of the two cliques
    "two_cliques.mtx": ("unweighted", 1, 1 / 21 + 1 / 21),
    "two_cliques_integer.mtx": ("integer", 1, 1 / 59 + 1 / 61),
    "two_cliques_real.mtx": ("real", 0.5, 0.5 / 48.5 + 0.5 / 50.5),
}


def are_cliques(cluster_of):
    """The clusters are the two cliques, nodes 0..4 and 5..9 (in any numbering)."""
    return len(set(cluster_of[:5])) == 1 and len(set(cluster_of[5:])) == 1 and cluster_of[0] != cluster_of[5]


def planted_clusters(clusters=4, size=40, seed=1):
    """A symmetric matrix: dense random clusters of `size` nodes, joined by a few edges."""
    rng = np.random.default_rng(seed)
    n = clusters * size
    same = np.arange(n)[:, None] // size == np.arange(n)[None, :] // size
    dense = np.triu(rng.random((n, n)) < np.where(same, 0.5, 0.002), k=1)
    return sp.csr_array(dense + dense.T, dtype=np.float64)


def expect_valid(matrix, result):
    """`result` is a clustering of the graph of the symmetric `matrix` with the right cut."""
    assert result.cluster_of.shape == (matrix.shape[0],)
    assert set(np.unique(result.cluster_of)) == set(range(result.num_clusters))
    upper = sp.triu(matrix, k=1).tocoo()
    cut = result.cluster_of[upper.row] != result.cluster_of[upper.col]
    assert result.cut_edges == cut.sum()
    assert result.cut_weight == pytest.approx(upper.data[cut].sum())


def expected_backend(graph):
    return "cuda" if deco.cuda_available() and graph.weights != "real" else "cpu"


# ---------------------------------------------------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", TWO_CLIQUES)
def test_from_mtx(name):
    g = deco.Graph.from_mtx(DATA / name)
    assert (g.num_nodes, g.num_edges, g.weights) == (10, 21, TWO_CLIQUES[name][0])
    assert repr(g) == f"Graph(num_nodes=10, num_edges=21, weights='{g.weights}')"


@pytest.mark.parametrize("name", TWO_CLIQUES)
def test_from_scipy_is_the_mtx_graph(name):
    a, b = deco.Graph.from_mtx(DATA / name), deco.Graph.from_scipy(scipy.io.mmread(DATA / name, spmatrix=False))
    assert (b.num_nodes, b.num_edges, b.weights) == (a.num_nodes, a.num_edges, a.weights)
    # the same graph: the same total weight, and the same result
    assert deco.sparsifier(b).levels[-1].volumes.sum() == deco.sparsifier(a).levels[-1].volumes.sum()
    np.testing.assert_array_equal(deco.expander_decomposition(b, backend="cpu").cluster_of,
                                  deco.expander_decomposition(a, backend="cpu").cluster_of)


def edges(matrix, **kwargs):
    g = deco.Graph.from_scipy(matrix, **kwargs)
    return g.num_nodes, g.num_edges, g.weights


def test_from_scipy_symmetry():
    path = sp.coo_array(([1, 1, 1, 1], ([0, 1, 1, 2], [1, 0, 2, 1])), shape=(3, 3))
    assert edges(path) == (3, 2, "unweighted")                  # symmetric: one edge per node pair
    assert edges(sp.triu(path)) == (3, 2, "unweighted")         # one direction per edge (not symmetric)
    directed = sp.coo_array(([1, 1, 1], ([0, 1, 1], [1, 0, 2])), shape=(3, 3))
    assert edges(directed) == (3, 3, "unweighted")              # A + A^T: every entry is an edge
    different = sp.coo_array(([2, 3], ([0, 1], [1, 0])), shape=(2, 2))
    assert edges(different) == (2, 2, "integer")                # not symmetric as the values differ


def test_from_scipy_entries():
    # duplicates add up, the diagonal and zeros are dropped
    m = sp.coo_array(([1, 1, 5, 0, 0], ([0, 0, 1, 2, 1], [1, 1, 1, 0, 2])), shape=(4, 4))
    g = deco.Graph.from_scipy(m)
    assert (g.num_nodes, g.num_edges, g.weights) == (4, 1, "integer")
    assert deco.sparsifier(g).levels[-1].volumes.sum() == 4     # one edge of weight 2
    assert edges(sp.csr_array((5, 5))) == (5, 0, "unweighted")  # no edges
    assert edges(sp.csr_array((0, 0))) == (0, 0, "unweighted")


@pytest.mark.parametrize("values, weights", [
    ([1, 1], "unweighted"),
    ([True, True], "unweighted"),
    ([2, 3], "integer"),
    ([2.0, 1.0], "integer"),
    ([0.5, 1.0], "real"),
    ([1e30, 1.0], "real"),   # whole, but too large for 64-bit integers
])
def test_from_scipy_weights(values, weights):
    m = sp.coo_array((values, ([0, 1], [1, 2])), shape=(3, 3))
    assert edges(m) == (3, 2, weights)
    assert edges(m, weighted=False) == (3, 2, "unweighted")
    assert edges(m, weighted=True)[2] == ("integer" if weights == "unweighted" else weights)


def test_from_scipy_errors():
    with pytest.raises(TypeError):
        deco.Graph.from_scipy(np.eye(3))
    with pytest.raises(TypeError):
        deco.Graph.from_scipy(sp.csr_array(np.array([[0, 1j], [1j, 0]])))
    with pytest.raises(ValueError, match="square"):
        deco.Graph.from_scipy(sp.csr_array((2, 3)))
    with pytest.raises(ValueError, match="nonnegative"):
        deco.Graph.from_scipy(sp.csr_array(np.array([[0, -1], [-1, 0]])))
    with pytest.raises(ValueError, match="finite"):
        deco.Graph.from_scipy(sp.csr_array(np.array([[0, np.nan], [np.nan, 0]])))


def test_from_mtx_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        deco.Graph.from_mtx(tmp_path / "missing.mtx")
    broken = tmp_path / "broken.mtx"
    broken.write_text("not a matrix\n")
    with pytest.raises(ValueError, match="header"):
        deco.Graph.from_mtx(broken)


# ---------------------------------------------------------------------------------------------------------------------
# Computations
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", TWO_CLIQUES)
def test_expander_decomposition(name):
    g = deco.Graph.from_mtx(DATA / name)
    d = deco.expander_decomposition(g)
    assert are_cliques(d.cluster_of)
    assert (d.num_clusters, d.cut_edges, d.cut_weight) == (2, 1, TWO_CLIQUES[name][1])
    assert d.backend == expected_backend(g)
    assert d.iterations > 0 and d.seconds >= 0


@pytest.mark.parametrize("name", TWO_CLIQUES)
def test_sparsifier(name):
    g = deco.Graph.from_mtx(DATA / name)
    t = deco.sparsifier(g)
    assert len(t.levels) == 2
    first, top = t.levels
    assert are_cliques(first.parents)
    assert first.cut_weights.tolist() == [TWO_CLIQUES[name][1]] * 2
    assert top.parents.tolist() == [0, 0] and top.volumes.sum() == first.volumes.sum()

    demand = np.zeros(10)
    demand[[0, 9]] = 1, -1                                     # across the cut: congestion 1 / cut weight
    assert t.predict(demand) == pytest.approx(1 / TWO_CLIQUES[name][1])
    assert t.predict([1, -1] + [0] * 8) == 0                   # inside a clique
    with pytest.raises(ValueError, match="sum to zero"):
        t.predict(np.ones(10))
    with pytest.raises(ValueError, match="one value per node"):
        t.predict(np.zeros(9))


@pytest.mark.parametrize("name", TWO_CLIQUES)
def test_normalized_cut(name):
    g = deco.Graph.from_mtx(DATA / name)
    t = deco.sparsifier(g)
    c = deco.normalized_cut(t, k=2)
    assert are_cliques(c.cluster_of)
    assert (c.num_clusters, c.cut_edges, c.cut_weight) == (2, 1, TWO_CLIQUES[name][1])
    assert c.normalized_cut == pytest.approx(TWO_CLIQUES[name][2])
    one = deco.normalized_cut(t, 1)
    assert (one.num_clusters, one.normalized_cut) == (1, 0)


def test_larger_graph():
    m = planted_clusters()
    g = deco.Graph.from_scipy(m)
    expect_valid(m, deco.expander_decomposition(g, sparsity=0.1))
    t = deco.sparsifier(g)
    c = deco.normalized_cut(t, 4)
    expect_valid(m, c)
    assert c.num_clusters == 4
    for below, level in zip([g.num_nodes] + [len(l.volumes) for l in t.levels], t.levels):
        assert level.parents.shape == (below,) and level.cut_weights.shape == level.volumes.shape


def test_graph_is_reusable():
    g = deco.Graph.from_scipy(planted_clusters())
    first = deco.expander_decomposition(g, backend="cpu", seed=5)
    deco.sparsifier(g, backend="cpu")
    deco.normalized_cut(deco.sparsifier(g, backend="cpu"), 4)
    np.testing.assert_array_equal(deco.expander_decomposition(g, backend="cpu", seed=5).cluster_of, first.cluster_of)


def test_sparsifier_is_reusable():
    m = planted_clusters()
    t = deco.sparsifier(deco.Graph.from_scipy(m), backend="cpu")
    demand = np.zeros(m.shape[0])
    demand[[0, -1]] = 1, -1
    congestion = t.predict(demand)

    first = deco.normalized_cut(t, 4)
    deco.normalized_cut(t, 2)
    again = deco.normalized_cut(t, 4)
    np.testing.assert_array_equal(again.cluster_of, first.cluster_of)
    assert again.normalized_cut == first.normalized_cut
    assert t.predict(demand) == congestion

    # the same as with a new graph and sparsifier
    fresh = deco.normalized_cut(deco.sparsifier(deco.Graph.from_scipy(m), backend="cpu"), 4)
    np.testing.assert_array_equal(fresh.cluster_of, first.cluster_of)
    assert fresh.normalized_cut == first.normalized_cut


def test_threads():
    g = deco.Graph.from_scipy(planted_clusters())
    expected = deco.expander_decomposition(g, backend="cpu").cluster_of
    results = [None] * 4

    def work(i):
        results[i] = deco.expander_decomposition(g, backend="cpu").cluster_of

    threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for result in results:
        np.testing.assert_array_equal(result, expected)


def test_option_errors():
    g = deco.Graph.from_mtx(DATA / "two_cliques.mtx")
    for options in [dict(sparsity=0), dict(walk_threshold=-1), dict(seed=-1), dict(seed=2**32), dict(backend="gpu")]:
        with pytest.raises(ValueError):
            deco.expander_decomposition(g, **options)
    t = deco.sparsifier(g)
    for k in [0, 11]:
        with pytest.raises(ValueError, match="k must be"):
            deco.normalized_cut(t, k)
    with pytest.raises(TypeError, match="deco.Sparsifier"):
        deco.normalized_cut(g, 2)
    with pytest.raises(TypeError, match="deco.Graph"):
        deco.expander_decomposition(sp.eye(3))


def test_real_weights_on_cuda():
    g = deco.Graph.from_mtx(DATA / "two_cliques_real.mtx")
    with pytest.raises(ValueError, match="CUDA"):
        deco.expander_decomposition(g, backend="cuda")


@pytest.mark.skipif(deco.cuda_available(), reason="a CUDA device is available")
def test_cuda_unavailable():
    g = deco.Graph.from_mtx(DATA / "two_cliques.mtx")
    with pytest.raises(ValueError, match="CUDA"):
        deco.expander_decomposition(g, backend="cuda")


@pytest.mark.skipif(not deco.cuda_available(), reason="no CUDA device")
@pytest.mark.parametrize("name", ["two_cliques.mtx", "two_cliques_integer.mtx"])
def test_cuda(name):
    g = deco.Graph.from_mtx(DATA / name)
    d = deco.expander_decomposition(g, backend="cuda")
    assert d.backend == "cuda" and are_cliques(d.cluster_of)
    assert are_cliques(deco.sparsifier(g, backend="cuda").levels[0].parents)
    assert are_cliques(deco.normalized_cut(deco.sparsifier(g, backend="cuda"), 2).cluster_of)
    m = planted_clusters()
    expect_valid(m, deco.expander_decomposition(deco.Graph.from_scipy(m), backend="cuda"))


# ---------------------------------------------------------------------------------------------------------------------
# The command-line tool, which pip installs as well: the same results as the module
# ---------------------------------------------------------------------------------------------------------------------

def test_command_line_tool(tmp_path):
    cli = shutil.which("Deco") or shutil.which("Deco", path=sysconfig.get_path("scripts"))
    assert cli, "the command-line tool Deco is not installed"
    for name in TWO_CLIQUES:
        output = tmp_path / "clusters.txt"
        run = subprocess.run([cli, "decompose", str(DATA / name), "--backend", "cpu", "--json", "-o", str(output)],
                             capture_output=True, text=True, check=True)
        summary = json.loads(run.stdout)
        d = deco.expander_decomposition(deco.Graph.from_mtx(DATA / name), backend="cpu")
        assert (summary["numClusters"], summary["cutEdges"], summary["cutWeight"]) == (d.num_clusters, d.cut_edges, d.cut_weight)
        assert summary["iterations"] == d.iterations
        np.testing.assert_array_equal(np.loadtxt(output, dtype=np.uint32), d.cluster_of)

        run = subprocess.run([cli, "ncut", str(DATA / name), "-k", "2", "--backend", "cpu", "--json"],
                             capture_output=True, text=True, check=True)
        c = deco.normalized_cut(deco.sparsifier(deco.Graph.from_mtx(DATA / name), backend="cpu"), 2)
        assert json.loads(run.stdout)["normalizedCut"] == pytest.approx(c.normalized_cut)
