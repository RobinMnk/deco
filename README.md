# Deco: Graph Decomposition Library

[![Tests](https://github.com/RobinMnk/deco/actions/workflows/tests.yml/badge.svg)](https://github.com/RobinMnk/deco/actions/workflows/tests.yml)
[![SPAA2026](https://img.shields.io/badge/DOI-10.1145%2F3816782.3819180-blue)](https://doi.org/10.1145/3816782.3819180)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus&logoColor=white)](#building-and-testing)
[![CUDA 12](https://img.shields.io/badge/CUDA-12-76B900?logo=nvidia&logoColor=white)](#building-and-testing)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](#python)
[![CMake 3.18+](https://img.shields.io/badge/CMake-3.18%2B-064F8C?logo=cmake&logoColor=white)](#building-and-testing)

**Deco** is a C++ library for computing graph decompositions based on algorithms with provable guarantees.
It provides scalable implementations for computing
- **Expander Decompositions** and
- **Hierarchical Congestion Approximators** (also known as Tree Cut Sparsifiers)

on undirected edge-weighted graphs.
All algorithms can be run sequentially on the CPU (C++20) and (shared-memory) parallel on NVIDIA GPUs (CUDA), behind one interface. Either implementation handles graphs
with millions of edges in seconds.

**Random walks instead of flows.** Expander decomposition algorithms usually find sparse cuts with repeated maximum-flow
computations, which dominate their running time on large graphs. Deco finds them with random walks: it needs no flow
computations at all. Random walks are simple, local and highly parallel: they make Deco fast even on a single CPU core,
map naturally onto GPUs, and let Deco scale to much larger graphs than earlier,
[flow-based implementations](https://doi.org/10.4230/LIPIcs.ESA.2024.61).

Deco implements the algorithm of

> _**A Practical Parallel Algorithm for Expander Decompositions**_ \
>  Robin M&uuml;nk \
>  in *Proceedings of the 38th ACM Symposium on Parallelism in Algorithms and Architectures* (SPAA '26), 2026 \
>  [10.1145/3816782.3819180](https://doi.org/10.1145/3816782.3819180)


## Quick start

```bash
git clone https://github.com/RobinMnk/deco.git && cd deco
# builds the CUDA backend if a CUDA toolkit is found
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDECO_BUILD_TESTS=OFF
cmake --build build --config Release
./build/Deco decompose tests/data/two_cliques.mtx
```

The example graph consists of two cliques joined by a single edge, and Deco separates them:

```text
command: decompose
graph: tests/data/two_cliques.mtx
nodes: 10
edges: 21
weights: unweighted
backend: cpu
iterations: 6
time: 0.021
numClusters: 2
cutEdges: 1
cutWeight: 1
```


## What Deco computes

```mermaid
flowchart LR
    G[Graph] --> E[Expander decomposition] --> C[Congestion approximator] --> N[Normalized cut]
```

Each step builds on the previous one, and each result is useful on its own.

### Expander decomposition

An expander decomposition partitions the nodes of a graph into clusters that are well connected inside (expanders),
while only a few edges run between clusters. Here, a cluster is well connected if it has no cut sparser than a threshold
&phi;, where the sparsity of a cut is its weight divided by the smaller volume of its two sides.

```bash
Deco decompose graph.mtx --sparsity 0.3
```

### Congestion approximator

A hierarchical congestion approximator is also known as a *tree cut sparsifier*. Deco computes it with an efficient
implementation of an *expander hierarchy*: it contracts every cluster of an expander decomposition into a single node and
decomposes the contracted graph again, until one node is left. The clusters of all levels form a tree whose cuts
approximate the cuts of the graph: for any set of demands, the tree predicts the congestion needed to route them in the
graph.

```bash
Deco sparsifier graph.mtx
```

### Normalized cut

One application of the congestion approximator is partitioning a graph into k clusters with a small normalized cut, the sum
over all clusters of the weight leaving the cluster divided by its volume. See
[Expander Hierarchies for Normalized Cuts on Graphs](https://arxiv.org/abs/2406.14111) (KDD '24).

```bash
Deco ncut graph.mtx -k 16
```


## How it works

The decomposition runs in rounds, on all active clusters at once. In each round, every cluster takes a step of a lazy
random walk, which spreads random initial values over the cluster. Sorting the nodes of a cluster by their value gives a
sequence of cuts (a sweep), and if the sparsest of them is below the threshold, Deco cuts the cluster there. A cluster
is done once its values have evened out, which happens quickly in an expander. Each round is a few passes over the
edges, which the CUDA implementation runs as parallel kernels over all clusters.

Compared with flow-based algorithms, the decomposition may cut more edges; in return it is far faster. The paper has the
details and the analysis.


## Performance

<!-- TODO: numbers from the paper, and the hardware they were measured on -->

| Graph | Nodes | Edges | CPU | GPU |
| --- | ---: | ---: | ---: | ---: |
| | | | | |


## Command-line tool

```text
Deco decompose  <graph.mtx> [options]
Deco sparsifier <graph.mtx> [options]
Deco ncut       <graph.mtx> -k N [options]

--backend auto|cpu|cuda   --sparsity X   --walk-threshold X   --seed N   -o FILE   --json   -h
```

The three commands compute the expander decomposition, the congestion approximator and the normalized cut described
[above](#what-deco-computes).

| Option | Description |
| --- | --- |
| `--backend B` | `auto`, `cpu` or `cuda` (default: `auto`, which uses CUDA if it can) |
| `--sparsity X` | cut a cluster if it has a sweep cut sparser than X (default: 0.3) |
| `--walk-threshold X` | a cluster is done once its walk values differ by less than X (default: 0.0001) |
| `--seed N` | seed for the random walks (default: 11) |
| `-k N` | number of clusters for `ncut` |
| `-o FILE` | write the cluster of every node to FILE, one line per node (`sparsifier`: its clusters on all levels) |
| `--json` | print the summary as one JSON object |
| `-h`, `--help` | show the help |

The summary goes to stdout, times are in milliseconds, and errors go to stderr.


## Python

The Python package `deco-graph` (imported as `deco`) offers the same three computations. pip builds it from source, with
the CUDA backend if a CUDA toolkit is found, and also installs the command-line tool `Deco`:

```bash
pip install .            # in a clone, or: pip install git+https://github.com/RobinMnk/deco
pip install ".[scipy]"   # with SciPy, for Graph.from_scipy
```

```python
import deco

g = deco.Graph.from_mtx("tests/data/two_cliques.mtx")   # or deco.Graph.from_scipy(A) for a SciPy sparse matrix
d = deco.expander_decomposition(g, sparsity=0.3)         # backend="auto" (default), "cpu" or "cuda"
print(d.num_clusters, d.cut_edges, d.cluster_of)         # 2 1 [0 0 0 0 0 1 1 1 1 1]

t = deco.sparsifier(g)                                   # congestion approximator
print(t.predict([1, 0, 0, 0, 0, 0, 0, 0, 0, -1]))        # 1.0
print(len(t.levels), t.levels[0].parents)                # 2 [0 0 0 0 0 1 1 1 1 1]

c = deco.normalized_cut(t, k=2)                          # from the congestion approximator t
print(c.num_clusters, c.normalized_cut)                  # 2 0.09523809523809523
```

A graph is converted once and can be used for any number of computations, and a congestion approximator for any number
of predictions and normalized cuts; the computations release the GIL. A symmetric SciPy
matrix gives one edge per node pair, any other matrix the graph A + A^T. The weights follow from the values (all 1:
unweighted, whole numbers: integer weights, otherwise real weights), unless `weighted=True` or `False` is given.
Invalid input raises `ValueError`.

Requirements: Python 3.10 or newer and a C++20 compiler (pip fetches CMake if needed). Run the tests with
`pip install ".[test]"` and `pytest`.


## Using Deco as a library

Add Deco to your CMake project and link `DecoLib`, plus `DecoExpanderDecompositionCuda` for the CUDA backend:

```cmake
add_subdirectory(deco)   # builds only the library; the CUDA backend if a CUDA toolkit is found

target_link_libraries(my_app PRIVATE DecoLib)
if(DECO_ENABLE_CUDA)
    target_link_libraries(my_app PRIVATE DecoExpanderDecompositionCuda)
endif()
```

Everything is available through `deco.h`:

```cpp
#include "deco.h"
#include "utils/graph_io.h"

#include <iostream>
#include <vector>

int main() {
    CsrGraph<> graph = load_mtx_unweighted("graph.mtx");

    deco::Options options;                      // backend Auto: CUDA if available, otherwise the CPU
    options.config.targetSparsity = 0.2;

    // Expander decomposition: clusterOf[v] is the cluster of node v
    const auto decomposition = deco::expanderDecomposition(graph, options);
    std::cout << decomposition.numClusters << " clusters, " << decomposition.cutEdges << " cut edges\n";

    // Congestion approximator: predicts the congestion of routing a demand (one value per node, summing to 0)
    auto tree = deco::expanderHierarchy(graph, options);
    std::vector<double> demand(graph.numNodes, 0.0);
    demand[0] = 1;
    demand[1] = -1;
    std::cout << "predicted congestion: " << tree.predict(demand) << "\n";

    // Normalized cut into 8 clusters, from the congestion approximator (or deco::normalizedCut(graph, 8, options))
    const auto clustering = deco::normalizedCut(graph, tree, 8);
    std::cout << "normalized cut: " << clustering.normalizedCut << " (" << clustering.cutEdges << " cut edges)\n";
}
```

Weighted graphs work the same way: `load_mtx_dynamic` returns unweighted, integer-weighted or real-weighted graphs.

| `Config` field | Description | Default |
| --- | --- | --- |
| `targetSparsity` | cut a cluster if it has a sweep cut sparser than this | 0.3 |
| `randomWalkThreshold` | a cluster is done once its walk values differ by less than this | 1e-4 |
| `seed` | seed for the initial random-walk values | 11 |

The CUDA backend handles unweighted and integer-weighted graphs. With `Backend::Auto`, Deco uses the CPU for other graphs
or when no GPU is available; asking for `Backend::Cuda` explicitly reports why it cannot be used.


## Input

Deco reads graphs in the [Matrix Market](https://math.nist.gov/MatrixMarket/formats.html) format (`.mtx`), unweighted or
with integer or real edge weights, as used by the [SuiteSparse Matrix Collection](https://sparse.tamu.edu/). Directed
graphs are read as undirected: every directed edge becomes an undirected edge. Support for edge weights comes with no
performance overhead for unweighted graphs on the CPU.


## Building and testing

Requirements:
- CMake 3.18 or newer and a C++20 compiler (tested with GCC 11 and 13, and MSVC 2022)
- for the CUDA backend: the CUDA Toolkit (tested with 12.9) and an NVIDIA GPU

| CMake option | Description | Default |
| --- | --- | --- |
| `DECO_ENABLE_CUDA` | build the CUDA backend | `ON` if a CUDA compiler is found |
| `DECO_BUILD_CLI` | build the command-line tool `Deco` | `ON` if Deco is the top-level project |
| `DECO_BUILD_TESTS` | build the tests (fetches GoogleTest) | `ON` if Deco is the top-level project |
| `DECO_BUILD_PYTHON` | build the Python module (`pip install` turns it on) | `OFF` |

The CUDA code is compiled for compute capabilities 8.0 and 8.6 (`CUDA_ARCHITECTURES` in
`lib/algorithms/expander_decomposition/cuda/internal/CMakeLists.txt`); adjust this for other GPUs.

The tests are built unless `DECO_BUILD_TESTS` is `OFF`, as in the quick start. Run them with:

```bash
cd build && ctest -C Release
```

Tests that need a GPU are skipped when there is none.


## Citation

If you use Deco in academic work, please cite:

```bibtex
@inproceedings{muenk2026parex,
  author    = {Robin M{\"u}nk},
  title     = {A Practical Parallel Algorithm for Expander Decompositions},
  booktitle = {Proceedings of the 38th ACM Symposium on Parallelism in Algorithms and Architectures},
  series    = {SPAA '26},
  year      = {2026},
  doi       = {10.1145/3816782.3819180}
}
```


## License

Deco is released under the [MIT License](LICENSE).
