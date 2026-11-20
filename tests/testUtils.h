//
// Shared helpers for the test suite: small graph builders and a full consistency check for Partition.
//

#ifndef DECO_TEST_UTILS_H
#define DECO_TEST_UTILS_H

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "core/graph.h"
#include "core/partition.h"
#include "algorithms/tree_cut_sparsifier/expander_hierarchy.h"

namespace testutil {

// Graph types the tests run on
using Unweighted32 = CsrGraph<>;            // unweighted (weight type EdgeIx)
using Weighted32   = CsrGraph<EdgeIx>;      // 32-bit weights, e.g. contracted levels and the CUDA weighted type
using Weighted64   = CsrGraph<IntWeight>;   // integer weights as read from input files

struct WeightedEdge {
    NodeIx u;
    NodeIx v;
    uint32_t w;
};

/**
 * Builds a CsrGraph from an undirected edge list. For unweighted graphs, the weights are ignored.
 */
template<class GraphT>
GraphT makeGraph(NodeIx numNodes, const std::vector<WeightedEdge>& edges) {
    using W = typename GraphT::EdgeWeight;
    if constexpr (std::same_as<GraphT, CsrGraph<>>) {
        DynamicGraph<> dg(numNodes, static_cast<EdgeIx>(edges.size()));
        for (const auto& e : edges) dg.addEdge(e.u, e.v);
        return dg.finalize();
    } else {
        DynamicGraph<W> dg(numNodes, static_cast<EdgeIx>(edges.size()));
        for (const auto& e : edges) dg.addEdge(e.u, e.v, static_cast<W>(e.w));
        return dg.finalize();
    }
}

// Clique on the nodes [offset, offset + k), all weights w
inline void addClique(std::vector<WeightedEdge>& edges, NodeIx offset, NodeIx k, uint32_t w = 1) {
    for (NodeIx i = 0; i < k; ++i)
        for (NodeIx j = i + 1; j < k; ++j)
            edges.push_back({offset + i, offset + j, w});
}

// Two cliques of size k (nodes [0, k) and [k, 2k)), connected by `bridges` edges i -- k + i
inline std::vector<WeightedEdge> twoCliques(NodeIx k, NodeIx bridges, uint32_t cliqueWeight = 1, uint32_t bridgeWeight = 1) {
    std::vector<WeightedEdge> edges;
    addClique(edges, 0, k, cliqueWeight);
    addClique(edges, k, k, cliqueWeight);
    for (NodeIx i = 0; i < bridges; ++i) edges.push_back({i, k + i, bridgeWeight});
    return edges;
}

/**
 * Random simple graph with n nodes and (up to) m edges, weights uniform in [1, maxWeight].
 * If connected is set, the path 0 - 1 - ... - (n-1) is included (and counts towards m).
 * Deterministic for a given seed on a given standard library.
 */
inline std::vector<WeightedEdge> randomEdges(NodeIx n, EdgeIx m, uint32_t maxWeight, unsigned seed, bool connected = false) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<NodeIx> node(0, n - 1);
    std::uniform_int_distribution<uint32_t> weight(1, maxWeight);
    std::set<std::pair<NodeIx, NodeIx>> seen;
    std::vector<WeightedEdge> edges;

    if (connected) {
        for (NodeIx v = 0; v + 1 < n; ++v) {
            seen.insert({v, v + 1});
            edges.push_back({v, v + 1, weight(rng)});
        }
    }

    for (EdgeIx tries = 0; edges.size() < m && tries < 20 * m; ++tries) {
        NodeIx u = node(rng), v = node(rng);
        if (u == v) continue;
        if (u > v) std::swap(u, v);
        if (!seen.insert({u, v}).second) continue;
        edges.push_back({u, v, weight(rng)});
    }
    return edges;
}

/**
 * `numClusters` cliques of `size` nodes (weights in [2, 6]), chained by one edge of weight 1 between consecutive
 * cliques, plus up to `extraBridges` random weight-1 edges between different cliques. Connected.
 */
inline std::vector<WeightedEdge> plantedClusters(NodeIx numClusters, NodeIx size, NodeIx extraBridges, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> weight(2, 6);
    std::uniform_int_distribution<NodeIx> node(0, numClusters * size - 1);
    std::set<std::pair<NodeIx, NodeIx>> seen;
    std::vector<WeightedEdge> edges;

    auto add = [&](NodeIx u, NodeIx v, uint32_t w) {
        if (u > v) std::swap(u, v);
        if (u != v && seen.insert({u, v}).second) edges.push_back({u, v, w});
    };

    for (NodeIx c = 0; c < numClusters; ++c)
        for (NodeIx i = 0; i < size; ++i)
            for (NodeIx j = i + 1; j < size; ++j)
                add(c * size + i, c * size + j, weight(rng));

    for (NodeIx c = 0; c + 1 < numClusters; ++c) add(c * size, (c + 1) * size + 1, 1);

    for (NodeIx b = 0, tries = 0; b < extraBridges && tries < 100 * extraBridges; ++tries) {
        const NodeIx u = node(rng), v = node(rng);
        if (u / size == v / size) continue;
        const size_t before = edges.size();
        add(u, v, 1);
        b += edges.size() > before;
    }
    return edges;
}

/**
 * The same graph as a multigraph: every edge of weight w becomes w parallel edges of weight 1.
 * Algorithmically, both should be equivalent.
 */
inline std::vector<WeightedEdge> expandToParallelEdges(const std::vector<WeightedEdge>& edges) {
    std::vector<WeightedEdge> parallel;
    for (const auto& e : edges) {
        for (uint32_t i = 0; i < e.w; ++i) parallel.push_back({e.u, e.v, 1});
    }
    return parallel;
}

/**
 * Instances for the quality tests: `clusters` cliques of `size` nodes, joined by a chain of single edges and
 * `bridges` extra edges (see plantedClusters). The cuts between cliques are far below the sparsity threshold.
 */
struct PlantedInstance {
    NodeIx clusters;
    NodeIx size;
    NodeIx bridges;
};

inline constexpr PlantedInstance plantedInstances[] = {{6, 10, 10}, {6, 20, 10}, {8, 30, 8}, {4, 40, 4}, {10, 12, 0}};

inline std::string describe(const PlantedInstance& instance, unsigned seed) {
    return std::to_string(instance.clusters) + " cliques of " + std::to_string(instance.size) + ", "
         + std::to_string(instance.bridges) + " extra bridges, seed " + std::to_string(seed);
}

/**
 * Clique on `size` nodes with weights 2 and 3. Every cut has sparsity above 1/3 (cut weight / smaller volume:
 * at least 2 k (size - k) / (3 (size - 1) min(k, size - k)) > 1/3 for a side of k nodes), so a decomposition that
 * only cuts below a threshold of at most 1/3 has to keep it as one cluster.
 */
inline std::vector<WeightedEdge> expanderClique(NodeIx size) {
    std::vector<WeightedEdge> edges;
    for (NodeIx i = 0; i < size; ++i)
        for (NodeIx j = i + 1; j < size; ++j)
            edges.push_back({i, j, 2 + (i + j) % 2});
    return edges;
}

/**
 * Sorted (neighbor, weight) pairs of every node. Taken before running a Partition on a graph and compared
 * afterwards: Partition reorders edges within each adjacency list, which must not change the graph itself.
 */
template<class GraphT>
std::vector<std::vector<std::pair<NodeIx, uint64_t>>> adjacencySnapshot(const GraphT& graph) {
    std::vector<std::vector<std::pair<NodeIx, uint64_t>>> adj(graph.numNodes);
    for (NodeIx v = 0; v < graph.numNodes; ++v) {
        for (EdgeIx eix = graph.startIndex(v); eix < graph.endIndex(v); ++eix) {
            adj[v].emplace_back(graph.edges[eix], static_cast<uint64_t>(graph.edgeWeight(eix)));
        }
        std::sort(adj[v].begin(), adj[v].end());
    }
    return adj;
}

/**
 * Independent copy of a CsrGraph (CsrGraph itself is not copyable). Partition modifies the edge order of its graph,
 * so tests that build several Partitions, or compare against the GPU, work on copies.
 */
template<class GraphT>
GraphT copyGraph(const GraphT& graph) {
    std::vector<NodeIx> edges = graph.edges;
    std::vector<EdgeIx> ranges = graph.ranges;
    if constexpr (std::same_as<GraphT, CsrGraph<>>) {
        return GraphT(graph.numNodes, graph.numEdges, std::move(edges), std::move(ranges));
    } else {
        const auto w = graph.edgeWeights();
        std::vector<typename GraphT::EdgeWeight> weights(w.begin(), w.end());
        return GraphT(graph.numNodes, graph.numEdges, std::move(edges), std::move(ranges), std::move(weights));
    }
}

/**
 * Sparsity of every sweep-cut prefix of a cluster, computed directly: order the cluster's nodes by value (ties by node
 * index), and for prefix i (the first i + 1 nodes) compute cut weight / min(prefix volume, rest volume), using internal
 * degrees and internal edges only. Stops once the prefix holds the whole internal volume (like Partition::sweepCut),
 * so the returned vector has an entry for every prefix that is a proper cut.
 */
template<class GraphT, class Values>
std::vector<frac_t> prefixSparsities(const Partition<GraphT>& partition, NodeIx clusterId, const Values& values) {
    const GraphT& graph = partition.getGraph();
    const auto& cluster = partition.getCluster(clusterId);

    std::vector<NodeIx> order;
    for (const auto& cv : cluster) order.push_back(cv.nix);
    std::sort(order.begin(), order.end(), [&](NodeIx a, NodeIx b) {
        return values[a] < values[b] || (values[a] == values[b] && a < b);
    });

    std::vector<uint8_t> inPrefix(graph.numNodes, 0);
    const auto total = static_cast<int64_t>(cluster.internalVolume);
    int64_t cut = 0, volume = 0;
    std::vector<frac_t> sparsities;

    for (const NodeIx v : order) {
        const auto& cv = partition.vertexFor(v);
        volume += static_cast<int64_t>(cv.internalDegree);
        for (EdgeIx eix = graph.startIndex(v); eix < cv.end; ++eix) {
            const auto w = static_cast<int64_t>(graph.edgeWeight(eix));
            cut += inPrefix[graph.edges[eix]] ? -w : w;   // edge to the prefix stops crossing, edge to the rest starts
        }
        inPrefix[v] = 1;

        if (static_cast<frac_t>(volume) >= static_cast<frac_t>(total) - 1e-8) break;
        sparsities.push_back(static_cast<frac_t>(cut) / static_cast<frac_t>(std::min(volume, total - volume)));
    }
    return sparsities;
}

/**
 * Cluster index of every node, derived from iterating the clusters (not from any cached field).
 */
template<class GraphT>
std::vector<NodeIx> clusterLabels(const Partition<GraphT>& partition) {
    std::vector<NodeIx> label(partition.getGraph().numNodes, static_cast<NodeIx>(-1));
    NodeIx c = 0;
    for (const auto& cluster : partition) {
        for (const auto& cv : cluster) label[cv.nix] = c;
        ++c;
    }
    return label;
}

/**
 * Checks every piece of state a Partition maintains against a recomputation from scratch:
 *  - every node is in exactly one cluster, the clusters' node ranges are disjoint and cover all nodes,
 *    and vertexFor(nix) refers to the node's entry in its cluster,
 *  - every adjacency list starts with the internal edges (neighbor in the same cluster), followed by the cut edges,
 *  - internalDegree, cluster volume and internalVolume equal the recomputed (weighted) sums,
 *  - getNumCutEdges() and getCutSize() equal the number / weight of edges between different clusters,
 *  - the graph is unchanged apart from the order within adjacency lists (if a snapshot is given).
 */
template<class GraphT>
void expectPartitionConsistent(
    const Partition<GraphT>& partition,
    const std::vector<std::vector<std::pair<NodeIx, uint64_t>>>* snapshot = nullptr
) {
    // clusters point back to their Partition: this breaks if a Partition is copied or moved (e.g. returned by value
    // without copy elision) without updateSelfReferences(). Checked first, iterating the clusters relies on it.
    for (const auto& cluster : partition) {
        ASSERT_EQ(cluster.partition, &partition) << "a cluster points to a different Partition object (Partition was copied or moved)";
    }

    const GraphT& graph = partition.getGraph();
    const NodeIx n = graph.numNodes;
    constexpr NodeIx UNASSIGNED = static_cast<NodeIx>(-1);

    // --- nodes are assigned exactly once, node ranges are contiguous and cover [0, n)
    std::vector<NodeIx> label(n, UNASSIGNED);
    std::vector<std::pair<NodeIx, NodeIx>> ranges;
    NodeIx c = 0;
    for (const auto& cluster : partition) {
        ASSERT_LE(cluster.nodeBegin, cluster.nodeEnd) << "cluster " << c;
        ranges.emplace_back(cluster.nodeBegin, cluster.nodeEnd);
        for (const auto& cv : cluster) {
            ASSERT_LT(cv.nix, n);
            ASSERT_EQ(label[cv.nix], UNASSIGNED) << "node " << cv.nix << " is in two clusters";
            label[cv.nix] = c;
            EXPECT_EQ(&partition.vertexFor(cv.nix), &cv) << "vertexFor(" << cv.nix << ") points to the wrong entry";
        }
        ++c;
    }
    for (NodeIx v = 0; v < n; ++v) ASSERT_NE(label[v], UNASSIGNED) << "node " << v << " is in no cluster";

    std::sort(ranges.begin(), ranges.end());
    NodeIx expectedBegin = 0;
    for (const auto& [b, e] : ranges) {
        EXPECT_EQ(b, expectedBegin) << "node ranges of clusters are not contiguous";
        expectedBegin = e;
    }
    EXPECT_EQ(expectedBegin, n);

    // --- per node: internal edges first, internalDegree; per cluster: volumes
    uint64_t totalVolume = 0, summedClusterVolumes = 0;
    c = 0;
    for (const auto& cluster : partition) {
        uint64_t volume = 0, internalVolume = 0;
        for (const auto& cv : cluster) {
            ASSERT_GE(cv.end, graph.startIndex(cv.nix));
            ASSERT_LE(cv.end, graph.endIndex(cv.nix));

            uint64_t internalDegree = 0;
            for (EdgeIx eix = graph.startIndex(cv.nix); eix < cv.end; ++eix) {
                EXPECT_EQ(label[graph.edges[eix]], c) << "edge " << cv.nix << " -> " << graph.edges[eix] << " is marked internal but crosses clusters";
                internalDegree += static_cast<uint64_t>(graph.edgeWeight(eix));
            }
            for (EdgeIx eix = cv.end; eix < graph.endIndex(cv.nix); ++eix) {
                EXPECT_NE(label[graph.edges[eix]], c) << "edge " << cv.nix << " -> " << graph.edges[eix] << " is marked as cut but is internal";
            }

            EXPECT_EQ(static_cast<uint64_t>(cv.internalDegree), internalDegree) << "internalDegree of node " << cv.nix;
            volume += static_cast<uint64_t>(graph.weightedDegree(cv.nix));
            internalVolume += internalDegree;
        }
        EXPECT_EQ(static_cast<uint64_t>(cluster.volume), volume) << "volume of cluster " << c;
        EXPECT_EQ(static_cast<uint64_t>(cluster.internalVolume), internalVolume) << "internalVolume of cluster " << c;
        summedClusterVolumes += volume;
        ++c;
    }
    for (NodeIx v = 0; v < n; ++v) totalVolume += static_cast<uint64_t>(graph.weightedDegree(v));
    EXPECT_EQ(summedClusterVolumes, totalVolume);

    // --- cut edges and cut weight
    uint64_t cutDirectedEdges = 0, cutDirectedWeight = 0;
    for (NodeIx v = 0; v < n; ++v) {
        for (EdgeIx eix = graph.startIndex(v); eix < graph.endIndex(v); ++eix) {
            if (label[v] != label[graph.edges[eix]]) {
                ++cutDirectedEdges;
                cutDirectedWeight += static_cast<uint64_t>(graph.edgeWeight(eix));
            }
        }
    }
    EXPECT_EQ(static_cast<uint64_t>(partition.getNumCutEdges()), cutDirectedEdges / 2);
    EXPECT_EQ(static_cast<uint64_t>(partition.getCutSize()), cutDirectedWeight / 2);

    // --- the graph itself is unchanged (edges were only reordered within adjacency lists)
    if (snapshot) {
        EXPECT_EQ(adjacencySnapshot(graph), *snapshot) << "Partition changed the graph (not just the edge order)";
    }
}

/**
 * Every cluster is connected (using only its internal edges). Holds after consolidation.
 */
template<class GraphT>
void expectClustersConnected(const Partition<GraphT>& partition) {
    const GraphT& graph = partition.getGraph();
    const std::vector<NodeIx> label = clusterLabels(partition);
    std::vector<uint8_t> visited(graph.numNodes, 0);

    NodeIx c = 0;
    for (const auto& cluster : partition) {
        if (cluster.size() == 0) { ++c; continue; }
        std::vector<NodeIx> stack{cluster.begin()->nix};
        visited[stack.back()] = 1;
        NodeIx reached = 0;
        while (!stack.empty()) {
            const NodeIx u = stack.back();
            stack.pop_back();
            ++reached;
            for (EdgeIx eix = graph.startIndex(u); eix < graph.endIndex(u); ++eix) {
                const NodeIx nb = graph.edges[eix];
                if (label[nb] == c && !visited[nb]) {
                    visited[nb] = 1;
                    stack.push_back(nb);
                }
            }
        }
        EXPECT_EQ(reached, cluster.size()) << "cluster " << c << " is not connected";
        ++c;
    }
}

/**
 * No cluster still contains a sparse cut between planted cliques (nodes v of plantedClusters(..., size, ...) belong
 * to clique v / size). In every cluster with nodes of several cliques, the nodes of each clique are separated from
 * the rest of the cluster by a cut of sparsity at least `threshold` (cut weight / smaller volume, counting only the
 * cluster's internal edges). The decomposition keeps cutting while it finds a cut below its threshold, so it must not
 * stop with such a cut left inside a cluster.
 */
template<class GraphT>
void expectNoSparsePlantedCutLeft(const Partition<GraphT>& partition, NodeIx size, double threshold) {
    const GraphT& graph = partition.getGraph();
    const std::vector<NodeIx> label = clusterLabels(partition);

    NodeIx c = 0;
    for (const auto& cluster : partition) {
        std::map<NodeIx, std::pair<uint64_t, uint64_t>> cutAndVolume;   // per clique, over its nodes in this cluster
        uint64_t clusterVolume = 0;
        for (const auto& cv : cluster) {
            auto& [cut, volume] = cutAndVolume[cv.nix / size];
            for (EdgeIx eix = graph.startIndex(cv.nix); eix < graph.endIndex(cv.nix); ++eix) {
                const NodeIx nb = graph.edges[eix];
                if (label[nb] != c) continue;
                const auto w = static_cast<uint64_t>(graph.edgeWeight(eix));
                volume += w;
                clusterVolume += w;
                if (nb / size != cv.nix / size) cut += w;
            }
        }
        if (cutAndVolume.size() > 1) {
            for (const auto& [clique, cutVolume] : cutAndVolume) {
                const auto [cut, volume] = cutVolume;
                const uint64_t smaller = std::min(volume, clusterVolume - volume);
                if (smaller == 0) continue;
                const double sparsity = static_cast<double>(cut) / static_cast<double>(smaller);
                EXPECT_GE(sparsity, threshold) << "cluster " << c << " (" << cluster.size() << " nodes) contains a cut of sparsity "
                                               << sparsity << ": its nodes of clique " << clique << " against the rest";
            }
        }
        ++c;
    }
}

/**
 * Expected result of contracting `graph` along `parents` (cluster id of every node): for every pair of different
 * clusters p < q, the summed weight of all edges between them. Edges within a cluster disappear.
 */
template<class GraphT>
std::map<std::pair<NodeIx, NodeIx>, uint64_t> expectedContraction(const GraphT& graph, const std::vector<NodeIx>& parents) {
    std::map<std::pair<NodeIx, NodeIx>, uint64_t> weight;
    for (NodeIx u = 0; u < graph.numNodes; ++u) {
        for (EdgeIx eix = graph.startIndex(u); eix < graph.endIndex(u); ++eix) {
            const NodeIx p = parents[u], q = parents[graph.edges[eix]];
            if (p < q) weight[{p, q}] += static_cast<uint64_t>(graph.edgeWeight(eix));   // every undirected edge once
        }
    }
    return weight;
}

/**
 * `contracted` is exactly the contraction described by `expected`: no self-loops, no parallel edges, and the right
 * weight on every edge (in both directions).
 */
template<class ContractedGraph>
void expectGraphMatchesContraction(const ContractedGraph& contracted, const std::map<std::pair<NodeIx, NodeIx>, uint64_t>& expected) {
    EXPECT_EQ(contracted.numEdges, expected.size()) << "number of edges after contraction";
    for (NodeIx p = 0; p < contracted.numNodes; ++p) {
        std::set<NodeIx> neighbors;
        for (EdgeIx eix = contracted.startIndex(p); eix < contracted.endIndex(p); ++eix) {
            const NodeIx q = contracted.edges[eix];
            EXPECT_NE(p, q) << "self-loop at " << p;
            EXPECT_TRUE(neighbors.insert(q).second) << "parallel edge " << p << " -- " << q;
            const auto it = expected.find({std::min(p, q), std::max(p, q)});
            if (it == expected.end()) {
                ADD_FAILURE() << "edge " << p << " -- " << q << " should not exist";
            } else {
                EXPECT_EQ(static_cast<uint64_t>(contracted.edgeWeight(eix)), it->second) << "weight of " << p << " -- " << q;
            }
        }
    }
}

/**
 * One level of an expander hierarchy against the graph one level below (`below`, with the volume of each of its nodes
 * in the original graph in `belowVolumes`):
 *  - sizes match, `parents` and `children` are inverse to each other,
 *  - volumes[c] is the summed volume of c's children,
 *  - cutSizes[c] is the weight of the edges leaving c's children in `below`, which is also c's weighted degree,
 *  - the level's graph is exactly the contraction of `below` along `parents`.
 */
template<class Level, class BelowGraph>
void expectLevelConsistent(const Level& level, const BelowGraph& below, const std::vector<uint64_t>& belowVolumes) {
    const auto& g = level.graph;
    ASSERT_EQ(level.volumes.size(), g.numNodes);
    ASSERT_EQ(level.cutSizes.size(), g.numNodes);
    ASSERT_EQ(level.childrenStart.size(), g.numNodes + 1);
    ASSERT_EQ(level.parents.size(), below.numNodes);
    ASSERT_EQ(level.children.size(), below.numNodes);

    std::vector<int> seen(below.numNodes, 0);
    for (NodeIx p = 0; p < g.numNodes; ++p) {
        uint64_t volume = 0;
        for (const NodeIx child : level.childrenOf(p)) {
            ASSERT_LT(child, below.numNodes);
            EXPECT_EQ(level.parents[child], p) << "children and parents disagree for node " << child;
            ++seen[child];
            volume += belowVolumes[child];
        }
        EXPECT_EQ(static_cast<uint64_t>(level.volumes[p]), volume) << "volume of cluster " << p;
    }
    for (NodeIx v = 0; v < below.numNodes; ++v) EXPECT_EQ(seen[v], 1) << "node " << v << " is not a child of exactly one cluster";

    std::vector<uint64_t> cut(g.numNodes, 0);
    for (NodeIx u = 0; u < below.numNodes; ++u) {
        for (EdgeIx eix = below.startIndex(u); eix < below.endIndex(u); ++eix) {
            if (level.parents[u] != level.parents[below.edges[eix]]) cut[level.parents[u]] += static_cast<uint64_t>(below.edgeWeight(eix));
        }
    }
    for (NodeIx p = 0; p < g.numNodes; ++p) {
        EXPECT_EQ(static_cast<uint64_t>(level.cutSizes[p]), cut[p]) << "cut size of cluster " << p;
        EXPECT_EQ(static_cast<uint64_t>(g.weightedDegree(p)), cut[p]) << "weighted degree of contracted node " << p;
    }

    expectGraphMatchesContraction(g, expectedContraction(below, level.parents));
}

/**
 * Every level of the hierarchy is consistent (see expectLevelConsistent), the total volume is preserved,
 * and the hierarchy ends in a single node or a graph without edges.
 */
template<class EdgeWeight, class InputGraph>
void expectHierarchyConsistent(ExpanderHierarchy<EdgeWeight>& hierarchy, const InputGraph& original) {
    ASSERT_GT(hierarchy.height(), 0u);

    std::vector<uint64_t> volumes(original.numNodes);
    for (NodeIx v = 0; v < original.numNodes; ++v) volumes[v] = static_cast<uint64_t>(original.weightedDegree(v));
    const uint64_t totalVolume = std::accumulate(volumes.begin(), volumes.end(), uint64_t{0});

    for (NodeIx l = 0; l < hierarchy.height(); ++l) {
        SCOPED_TRACE("level " + std::to_string(l));
        const auto& level = hierarchy.lvl(l);
        if (l == 0) expectLevelConsistent(level, original, volumes);
        else        expectLevelConsistent(level, hierarchy.lvl(l - 1).graph, volumes);
        if (::testing::Test::HasFatalFailure()) return;

        volumes.assign(level.volumes.begin(), level.volumes.end());
        EXPECT_EQ(std::accumulate(volumes.begin(), volumes.end(), uint64_t{0}), totalVolume) << "total volume changed";
    }

    const auto& top = hierarchy.lvl(hierarchy.height() - 1).graph;
    EXPECT_TRUE(top.numNodes == 1 || top.numEdges == 0) << "hierarchy ends with " << top.numNodes << " nodes and " << top.numEdges << " edges";
}

} // namespace testutil

#endif //DECO_TEST_UTILS_H
