# Parex

Parex is a C++/CUDA implementation of fast, random-walk based expander decompositions and congestion approximators.

Expander decompositions are a central tool in modern graph algorithms. They partition a graph into well-connected pieces while keeping the number of edges between pieces small. This structure is useful for routing, cut and flow problems, sparsification, dynamic connectivity, congestion approximation, and graph partitioning objectives such as normalized cut.

Despite their theoretical power, expander decompositions have traditionally been difficult to use in practice. Many near-optimal algorithms rely heavily on repeated max-flow computations. Those flow computations come with large hidden constants and often dominate the total running time on real-world graphs.

Parex explores a different direction: replacing the main flow-based cut-finding step with random walks. Random walks are simple, local, and highly parallelizable. The resulting decomposition may cut more inter-cluster edges than stronger flow-based approaches, but it is much better suited to scalable parallel execution. In the experiments from the paper, the CUDA implementation decomposes graphs with millions to tens of millions of edges in seconds, while serving as a useful backbone for downstream tasks.

The implementation is based on:

> Robin M&uuml;nk, "A Practical Parallel Algorithm for Expander Decompositions," in *Proceedings of the 38th ACM Symposium on Parallelism in Algorithms and Architectures* (SPAA '26), 2026, doi: [10.1145/3816782.3819180](https://doi.org/10.1145/3816782.3819180)

## Why Parex?

Parex is motivated by a practical bottleneck: expander decompositions are theoretically powerful, but many implementations remain too expensive for routine use on large real-world graphs.

The project focuses on three ideas:

- **Avoid expensive flow computations in the main cut-finding loop.** Parex uses random walks to expose sparse cuts instead of repeatedly solving max-flow instances.
- **Exploit parallelism.** The random-walk based procedure maps naturally to CUDA, making it possible to process graphs with millions of edges in seconds.
- **Evaluate usefulness through downstream tasks.** Rather than treating decomposition quality only as an abstract guarantee, Parex uses the decomposition hierarchy for congestion approximation and normalized cut computation.

This makes Parex especially interesting as a practical experiment in weak expander decompositions: it investigates how far weaker, more scalable decompositions can replace stronger but more expensive ones in downstream graph algorithms.

## Algorithmic Structure

Parex can be seen as a pipeline:

```text
expander decomposition -> congestion approximator -> normalized cut
```

Each intermediate result is useful on its own:

- **Expander decomposition:** partitions the graph into clusters that are internally well-connected while exposing sparse cuts between clusters.
- **Congestion approximator:** builds a hierarchical structure that approximates the congestion of routing demands through the graph.
- **Normalized cut:** uses the congestion approximator to compute normalized cut values and derive graph partitions.

At the base of the pipeline, Parex provides two implementations of expander decomposition:

| Implementation | Description |
| --- | --- |
| CPU expander decomposition | A single-threaded baseline and reference implementation. |
| CUDA expander decomposition | A GPU-accelerated implementation for scalable decomposition on NVIDIA GPUs. |

Both implementations follow the same high-level principle: diffuse values inside active clusters using random walks, use those values to search for sparse cuts, and split clusters when a sufficiently sparse cut is found. For the algorithmic details and analysis, see the paper.

## Project Status

Parex is research code accompanying the paper above. It contains:

- a single-threaded CPU expander decomposition implementation
- a CUDA-accelerated expander decomposition implementation
- hierarchy construction for congestion approximation
- normalized cut computation built on the hierarchy

The current command-line executable runs the CUDA expander decomposition path and reports partition statistics. Some higher-level components are available as library code rather than as polished command-line tools.

## Requirements

- CMake 3.18 or newer
- A C++20 compiler
- NVIDIA CUDA Toolkit with C++20-capable CUDA compiler support
- An NVIDIA GPU compatible with the configured CUDA architectures
- Git, if CMake needs to fetch test dependencies

GoogleTest is fetched automatically through CMake for the test targets.

## CUDA Architecture

The CUDA library currently targets these GPU architectures:

```cmake
CUDA_ARCHITECTURES "80;86"
```

This is configured in `cuda/CMakeLists.txt`. If your GPU uses a different compute capability, update that setting before building.

## Building

From the repository root:

```powershell
cmake -S . -B cmake-build-release -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release --config Release
```

## Running

The current executable runs the CUDA expander decomposition path.

```text
Parex <filename> [seed] [sweep_cut_threshold] [random_walk_threshold]
```

Arguments:

| Argument | Description |
| --- | --- |
| `filename` | Path to the input graph file. |
| `seed` | Optional random seed. |
| `sweep_cut_threshold` | Optional sparsity threshold for accepting a sweep cut. |
| `random_walk_threshold` | Optional threshold for random-walk convergence. |

Example:

```powershell
.\cmake-build-release\Parex graphs\example.mtx 6582 0.3 1e-4
```

The program prints:

- running time
- number of clusters
- number of cut edges
- number of nodes
- number of edges

## Input Format

The current graph reader expects a Matrix Market-style coordinate input.

Comment lines beginning with `%` are skipped. The first non-comment line is interpreted as graph dimensions and edge count. Edge endpoints are read as 1-indexed vertex IDs and converted to 0-indexed IDs internally.

## Configuration

Several algorithm parameters are exposed either through the CLI or through configuration structs and global defaults:

| Parameter | Description |
| --- | --- |
| `targetSparsity` / `sc_threshold` | Threshold below which a sweep cut is considered sparse enough to split a cluster. |
| `randomWalkThreshold` / `rw_threshold` | Threshold used for random-walk convergence and potential checks. |
| `initialNumSteps` | Number of random-walk steps before the first sweep cut in the CPU implementation. |
| `mainNumSteps` | Number of random-walk steps between later rounds in the CPU implementation. |
| `randSeed` | Seed for the random source used by the CPU implementation. |

The CUDA implementation also uses constants from `lib/core/definitions.h`, including the thread and warp configuration.

## Citation

If you use Parex in academic work, please cite the paper this implementation is based on:

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
