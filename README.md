# Deco: Graph Decomposition Library

[![Tests](https://github.com/RobinMnk/deco/actions/workflows/tests.yml/badge.svg)](https://github.com/RobinMnk/deco/actions/workflows/tests.yml)
[![Paper](https://img.shields.io/badge/DOI-10.1145%2F3816782.3819180-blue)](https://doi.org/10.1145/3816782.3819180)

Deco is a high-performance C++ library for computing graph decompositions based on **algorithms with provable guarantees**.
It provides scalable implementations for computing
- **Expander Decompositions** and
- **Tree Cut Sparsifiers** (i.e., Hierarchical Congestion Approximators)

Both run on the CPU (C++20) and on NVIDIA GPUs (CUDA), behind one interface, and either implementation handles graphs
with millions of edges in seconds.

**Random walks instead of flows.** Expander decomposition algorithms usually find sparse cuts with repeated maximum-flow
computations, which dominate their running time on large graphs. Deco finds them with random walks: it needs no flow
computations at all. Random walks are simple, local and highly parallel. This makes Deco fast even on a single CPU core,
maps naturally onto GPUs, and lets Deco scale to much larger graphs than earlier,
[flow-based implementations](https://doi.org/10.4230/LIPIcs.ESA.2024.61).

Deco implements the algorithm of

> _**A Practical Parallel Algorithm for Expander Decompositions**_ \
>  Robin M&uuml;nk \
>  in *Proceedings of the 38th ACM Symposium on Parallelism in Algorithms and Architectures* (SPAA '26), 2026 \
>  [10.1145/3816782.3819180](https://doi.org/10.1145/3816782.3819180)


## Quick start

```bash
git clone https://github.com/RobinMnk/deco.git && cd deco
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release        # add -DDECO_ENABLE_CUDA=OFF without a CUDA toolkit
cmake --build build --config Release
./build/Deco decompose tests/data/two_cliques.mtx     # Visual Studio: build\Release\Deco.exe
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
time: 0
numClusters: 2
cutEdges: 1
cutWeight: 1
```


## What Deco computes

```mermaid
flowchart LR
    G[Graph] --> E[Expander decomposition] --> T[Tree cut sparsifier] --> N[Normalized cut]
```

Each step builds on the previous one, and each result is useful on its own.

### Expander decomposition

An expander decomposition partitions the nodes of a graph into clusters that are well connected inside (expanders),
while only few edges run between clusters. Here, a cluster is well connected if it has no cut sparser than a threshold
&phi;, where the sparsity of a cut is its weight divided by the smaller volume of its two sides.

```bash
Deco decompose graph.mtx --sparsity 0.3
```

### Tree cut sparsifier

Also known as an *expander hierarchy* or a *hierarchical congestion approximator*. Deco contracts every cluster of an
expander decomposition into a single node and decomposes the contracted graph again, until one node is left. The
clusters of all levels form a tree whose cuts approximate the cuts of the graph: for any set of demands, the tree
predicts the congestion needed to route them in the graph.

```bash
Deco hierarchy graph.mtx
```

### Normalized cut

One application of the tree cut sparsifier is partitioning a graph into k clusters with a small normalized cut, the sum
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
Deco <command> <graph.mtx> [options]

Commands:
  decompose            expander decomposition
  hierarchy            expander hierarchy (tree cut sparsifier)
  ncut                 partition into k clusters with a small normalized cut (needs -k)
```

| Option | Description |
| --- | --- |
| `--backend B` | `auto`, `cpu` or `cuda` (default: `auto`, which uses CUDA if it can) |
| `--sparsity X` | cut a cluster if it has a sweep cut sparser than X (default: 0.3) |
| `--walk-threshold X` | a cluster is done once its walk values differ by less than X (default: 0.0001) |
| `--seed N` | seed for the random walks (default: 11) |
| `-k N` | number of clusters for `ncut` |
| `-o FILE` | write the cluster of every node to FILE, one line per node (`hierarchy`: its clusters on all levels) |
| `--json` | print the summary as one JSON object |

The summary goes to stdout, times are in milliseconds, and errors go to stderr.


## Using Deco as a library

Add Deco to your CMake project and link `DecoLib`, plus `DecoExpanderDecompositionCuda` for the CUDA backend:

```cmake
set(DECO_BUILD_CLI OFF)
set(DECO_BUILD_TESTS OFF)
add_subdirectory(deco)

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

    // Tree cut sparsifier: predicts the congestion of routing a demand (one value per node, summing to 0)
    auto tree = deco::expanderHierarchy(graph, options);
    std::vector<double> demand(graph.numNodes, 0.0);
    demand[0] = 1;
    demand[1] = -1;
    std::cout << "predicted congestion: " << tree.predict(demand) << "\n";

    // Normalized cut into 8 clusters
    const auto clustering = deco::normalizedCut(graph, 8, options);
    std::cout << "normalized cut: " << clustering.cutEdges << " cut edges\n";
}
```

Weighted graphs work the same way: `load_mtx_dynamic` returns unweighted, integer-weighted or real-weighted graphs.

| `Config` field | Default | Description |
| --- | --- | --- |
| `targetSparsity` | 0.3 | cut a cluster if it has a sweep cut sparser than this |
| `randomWalkThreshold` | 1e-4 | a cluster is done once its walk values differ by less than this |
| `seed` | 11 | seed for the initial random-walk values |
| `initialNumSteps`, `mainNumSteps` | 1 | CPU only: walk steps before the first sweep cut, and between later ones |

The CUDA backend takes unweighted graphs and integer weights up to 2<sup>32</sup> &minus; 1, as long as twice the total
weight fits into 32 bits. With `Backend::Auto`, Deco falls back to the CPU otherwise; asking for `Backend::Cuda`
explicitly reports why it cannot be used.


## Input

Deco reads graphs in the [Matrix Market](https://math.nist.gov/MatrixMarket/formats.html) format (`.mtx`), unweighted or
with integer or real edge weights, as used by the [SuiteSparse Matrix Collection](https://sparse.tamu.edu/). Directed
graphs are read as undirected: every directed edge becomes an undirected edge.


## Building and testing

Requirements:
- CMake 3.18 or newer and a C++20 compiler (tested with GCC 11 and 13, and MSVC 2022)
- for the CUDA backend: the CUDA Toolkit (tested with 12.9) and an NVIDIA GPU

| CMake option | Default | Description |
| --- | --- | --- |
| `DECO_ENABLE_CUDA` | `ON` | build the CUDA backend |
| `DECO_BUILD_CLI` | `ON` | build the command-line tool `Deco` |
| `DECO_BUILD_TESTS` | `ON` | build the tests (fetches GoogleTest) |

The CUDA code is compiled for compute capabilities 8.0 and 8.6 (`CUDA_ARCHITECTURES` in
`lib/algorithms/expander_decomposition/cuda/internal/CMakeLists.txt`); adjust this for other GPUs.

Run the tests with:

```bash
cd build && ctest -C Release
```

Without a GPU, the interface tests skip their CUDA cases, but `CudaTest` fails; exclude it with `ctest -C Release -E Cuda`.


## Repository layout

```text
lib/deco.h                               public interface
lib/core/                                graphs and partitions
lib/algorithms/expander_decomposition/   expander decomposition: CPU, and CUDA in cuda/
lib/algorithms/tree_cut_sparsifier/      expander hierarchy (tree cut sparsifier)
lib/algorithms/applications/             normalized cut
lib/utils/                               Matrix Market reader, timer
cli/                                     command-line tool
tests/                                   unit tests and command-line smoke tests
```


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

<!-- TODO: choose a license and add a LICENSE file -->
