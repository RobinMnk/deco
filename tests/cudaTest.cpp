//
// Chunk 2: CUDA consistency.
//
// Runs the GPU decomposition and checks it against host-side computations of the same quantities:
// the uploaded graph, every random-walk step, every sweep cut, the state after cutting (labels, valid edges,
// internal degrees), and the final partition. Tests run on unweighted graphs and on 32-bit weighted graphs
// (the type of contracted hierarchy levels). All test graphs are connected and generated in code.
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <vector>

#include "testUtils.h"
#include "algorithms/expander_decomposition/cuda.h"
#include "algorithms/tree_cut_sparsifier/expander_hierarchy.h"

using namespace testutil;

template<class GraphT>
class CudaTest : public ::testing::Test {};

using CudaGraphTypes = ::testing::Types<Unweighted32, Weighted32>;
TYPED_TEST_SUITE(CudaTest, CudaGraphTypes);


namespace {

/**
 * `numClusters` cliques of `size` nodes (weights in [2, 6]), chained by one edge of weight 1 between consecutive
 * cliques, plus up to `extraBridges` random weight-1 edges between different cliques. Connected.
 */
std::vector<WeightedEdge> plantedClusters(NodeIx numClusters, NodeIx size, NodeIx extraBridges, unsigned seed) {
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

bool isActive(int label) { return label >= 0; }

bool anyActive(const std::vector<int>& labels) {
    return std::any_of(labels.begin(), labels.end(), isActive);
}

/**
 * Splits a freshly constructed host Partition (single cluster, connected graph) into one cluster per distinct GPU
 * label. Inactive clusters have negative labels. Returns label -> host cluster id.
 */
template<class GraphT>
std::map<int, NodeIx> splitByLabels(Partition<GraphT>& partition, const std::vector<int>& labels) {
    std::map<int, NodeIx> index;
    EXPECT_EQ(partition.numClusters(), 1u) << "test graphs must be connected";
    if (partition.numClusters() != 1) return index;

    for (const int l : labels) index.emplace(l, 0);
    NodeIx next = 0;
    for (auto& [label, id] : index) id = next++;

    std::vector<NodeIx> parts(labels.size());
    for (size_t v = 0; v < labels.size(); ++v) parts[v] = index.at(labels[v]);

    // cluster 0 keeps part 0, part i becomes cluster i (no consolidation)
    std::vector<NodeIx> modified;
    partition.template splitByIndices<true, false>(0, parts, next, modified);
    return index;
}

/**
 * One GPU random-walk step against the host: for every active node v,
 *   x'[v] = rw_stay * x[v] + (1 - rw_stay) * sum over valid edges (v, u) of w(v, u) * x[u] / d[u],
 * with the valid edges and internal degrees d taken from the GPU's current state.
 */
template<class GraphT>
void expectWalkStepMatchesHost(CudaDeviceManager& cuda, const GraphT& graph) {
    const auto labels = cuda.downloadLabels();
    const auto valid = cuda.downloadActiveEdgeMap();
    const auto degrees = cuda.downloadDegrees();
    const auto x = cuda.readRandomWalkValues();

    cuda.iterateRandomWalk();
    const auto gpu = cuda.readRandomWalkValues();

    int mismatches = 0;
    for (NodeIx v = 0; v < graph.numNodes && mismatches < 10; ++v) {
        if (!isActive(labels[v])) continue;   // the GPU does not update inactive nodes

        double sum = 0.0;
        for (EdgeIx eix = graph.startIndex(v); eix < graph.endIndex(v); ++eix) {
            if (!valid[eix]) continue;
            const NodeIx u = graph.edges[eix];
            if (degrees[u] > 0) sum += static_cast<double>(graph.edgeWeight(eix)) * x[u] / static_cast<double>(degrees[u]);
        }
        const double expected = rw_stay * x[v] + (1.0 - rw_stay) * sum;

        // the GPU sums in a different order (warp reduction), so allow for rounding
        if (std::abs(gpu[v] - expected) > 1e-9 * (1.0 + std::abs(expected))) {
            ADD_FAILURE() << "walk value of node " << v << ": GPU " << gpu[v] << ", host " << expected;
            ++mismatches;
        }
    }
}

/**
 * GPU sweep cuts against brute force on the host. The GPU orders the nodes of each cluster by the float value of
 * their walk value; clusters with ties in that order are skipped, since the order is then ambiguous.
 * The GPU reports per cluster the best sparsity and the prefix length (offset) where it cuts. Ties between equally
 * sparse prefixes are allowed: the GPU's prefix only has to be one of the best.
 * Returns the number of clusters compared.
 */
template<class GraphT>
NodeIx expectSweepCutsMatchHost(CudaDeviceManager& cuda, const GraphT& graph) {
    cuda.computeSweepCuts();
    const AllSweepCuts gpu = cuda.readSweepCuts();
    const auto labels = cuda.downloadLabels();
    const auto values = cuda.readRandomWalkValues();
    const std::vector<float> keys(values.begin(), values.end());

    auto copy = copyGraph(graph);
    Partition host{copy};
    const auto index = splitByLabels(host, labels);
    if (::testing::Test::HasFailure()) return 0;

    std::set<int> activeLabels;
    for (const int l : labels) if (isActive(l)) activeLabels.insert(l);
    EXPECT_EQ(gpu.clusterIds.size(), activeLabels.size()) << "one sweep cut per active cluster";

    NodeIx compared = 0;
    for (size_t i = 0; i < gpu.clusterIds.size(); ++i) {
        const int label = gpu.clusterIds[i];
        const SweepCutData& cut = gpu.cuts[i];
        EXPECT_EQ(cut.clusterId, label);
        if (!activeLabels.contains(label)) {
            ADD_FAILURE() << "sweep cut for label " << label << ", which is not an active cluster";
            continue;
        }
        const NodeIx c = index.at(label);

        std::vector<float> clusterKeys;
        for (const auto& cv : host.getCluster(c)) clusterKeys.push_back(keys[cv.nix]);
        std::sort(clusterKeys.begin(), clusterKeys.end());
        if (std::adjacent_find(clusterKeys.begin(), clusterKeys.end()) != clusterKeys.end()) continue;

        const auto sparsities = prefixSparsities(host, c, keys);
        if (sparsities.empty()) {
            ADD_FAILURE() << "active cluster " << label << " has no proper cut";
            continue;
        }
        const double best = *std::min_element(sparsities.begin(), sparsities.end());
        const double tolerance = 1e-5 * (1.0 + best);   // the GPU computes the sparsity in float

        EXPECT_NEAR(cut.sparsity, best, tolerance) << "best sparsity of cluster " << label;
        if (cut.offset >= 1 && cut.offset <= sparsities.size()) {
            EXPECT_NEAR(sparsities[cut.offset - 1], best, tolerance) << "GPU cuts cluster " << label << " after a prefix that is not optimal";
        } else {
            ADD_FAILURE() << "sweep-cut offset " << cut.offset << " of cluster " << label << " is not a proper prefix length";
        }
        ++compared;
    }
    return compared;
}

/**
 * GPU state after cutClusters against the host:
 *  - every cluster (active or not) is connected: the GPU splits cut clusters into their components,
 *  - for every node (active or not), an edge is valid exactly if both endpoints have the same label,
 *  - for every node, the internal degree is the weight of its valid edges.
 * Inactive nodes are no longer processed after their deactivation, so this also checks that their edges were
 * finalized correctly when they were deactivated.
 */
template<class GraphT>
void expectStateAfterCutMatchesHost(CudaDeviceManager& cuda, const GraphT& graph) {
    const auto labels = cuda.downloadLabels();
    const auto valid = cuda.downloadActiveEdgeMap();
    const auto degrees = cuda.downloadDegrees();
    ASSERT_EQ(labels.size(), graph.numNodes);
    ASSERT_EQ(valid.size(), graph.edges.size());

    auto copy = copyGraph(graph);
    Partition host{copy};
    splitByLabels(host, labels);
    if (::testing::Test::HasFailure()) return;
    expectClustersConnected(host);

    int mismatches = 0;
    for (NodeIx v = 0; v < graph.numNodes && mismatches < 10; ++v) {
        uint64_t internalDegree = 0;
        for (EdgeIx eix = graph.startIndex(v); eix < graph.endIndex(v); ++eix) {
            const bool sameCluster = labels[graph.edges[eix]] == labels[v];
            if (static_cast<bool>(valid[eix]) != sameCluster) {
                ADD_FAILURE() << "edge " << v << " -> " << graph.edges[eix] << " is " << (valid[eix] ? "valid" : "disabled")
                              << ", but the labels are " << labels[v] << " and " << labels[graph.edges[eix]];
                ++mismatches;
            }
            if (sameCluster) internalDegree += static_cast<uint64_t>(graph.edgeWeight(eix));
        }
        if (static_cast<uint64_t>(degrees[v]) != internalDegree) {
            ADD_FAILURE() << "internal degree of node " << v << ": GPU " << degrees[v] << ", host " << internalDegree;
            ++mismatches;
        }
    }
}

// Runs the full GPU decomposition and returns the final labels
template<class GraphT>
std::vector<int> decomposeOnGpu(const GraphT& graph) {
    CudaDeviceManager cuda;
    cuda.initialize(graph);
    cuda.expanderDecomposition();
    const FinalPartition result = cuda.getFinalPartition();
    EXPECT_EQ(result.numClusters, static_cast<int>(std::set<int>(result.clusterIds.begin(), result.clusterIds.end()).size()));
    return result.clusterIds;
}

} // namespace


// ---------------------------------------------------------------------------------------------------------------------
// Upload
// ---------------------------------------------------------------------------------------------------------------------

TYPED_TEST(CudaTest, UploadMatchesHostGraph) {
    constexpr NodeIx n = 100;
    const auto graph = makeGraph<TypeParam>(n, randomEdges(n, 400, 9, 1, true));

    CudaDeviceManager cuda;
    cuda.initialize(graph);

    const CsrGraph<> structure = cuda.downloadGraph();
    EXPECT_EQ(structure.numNodes, graph.numNodes);
    EXPECT_EQ(structure.numEdges, graph.numEdges);
    EXPECT_EQ(structure.ranges, graph.ranges);
    EXPECT_EQ(structure.edges, graph.edges);

    const auto weights = cuda.downloadWeights();
    const auto hostWeights = graph.edgeWeights();
    EXPECT_EQ(weights, std::vector<DevEdgeWeight>(hostWeights.begin(), hostWeights.end()));   // empty for unweighted

    const auto degrees = cuda.downloadDegrees();
    ASSERT_EQ(degrees.size(), n);
    for (NodeIx v = 0; v < n; ++v) EXPECT_EQ(degrees[v], graph.weightedDegree(v)) << "node " << v;

    const auto labels = cuda.downloadLabels();
    EXPECT_TRUE(std::all_of(labels.begin(), labels.end(), [](int l) { return l == 0; })) << "all nodes start in cluster 0";

    const auto valid = cuda.downloadActiveEdgeMap();
    EXPECT_TRUE(std::all_of(valid.begin(), valid.end(), [](int x) { return x == 1; })) << "all edges start valid";
}


// ---------------------------------------------------------------------------------------------------------------------
// Every phase of the decomposition loop against the host
// ---------------------------------------------------------------------------------------------------------------------

TYPED_TEST(CudaTest, EveryPhaseMatchesHost) {
    const auto graph = makeGraph<TypeParam>(48, plantedClusters(4, 12, 6, 1));

    CudaDeviceManager cuda;
    cuda.initialize(graph);

    // same loop as CudaDeviceManager::expanderDecomposition, with checks between the phases
    std::vector<int> previous = cuda.downloadLabels();
    NodeIx comparedSweepCuts = 0;
    int iteration = 0;
    constexpr int maxIterations = 2000;

    for (; iteration < maxIterations && anyActive(previous); ++iteration) {
        SCOPED_TRACE("iteration " + std::to_string(iteration));

        expectWalkStepMatchesHost(cuda, graph);
        comparedSweepCuts += expectSweepCutsMatchHost(cuda, graph);
        cuda.cutClusters();
        expectStateAfterCutMatchesHost(cuda, graph);

        // deactivated clusters never change again, and newly deactivated clusters get labels that are not in use
        const auto labels = cuda.downloadLabels();
        std::set<int> inactiveBefore;
        for (const int l : previous) {
            if (!isActive(l)) inactiveBefore.insert(l);
        }
        for (NodeIx v = 0; v < graph.numNodes; ++v) {
            if (!isActive(previous[v])) {
                EXPECT_EQ(labels[v], previous[v]) << "label of inactive node " << v << " changed";
            } else if (!isActive(labels[v])) {
                EXPECT_FALSE(inactiveBefore.contains(labels[v]))
                    << "node " << v << " was deactivated with label " << labels[v] << ", which an earlier inactive cluster already has";
            }
        }
        previous = labels;

        if (::testing::Test::HasFailure()) return;
    }

    EXPECT_FALSE(anyActive(previous)) << "no termination within " << maxIterations << " iterations";
    EXPECT_GT(comparedSweepCuts, 0u) << "no sweep cut could be compared (float ties in every cluster)";
}


// ---------------------------------------------------------------------------------------------------------------------
// Final partition
// ---------------------------------------------------------------------------------------------------------------------

TYPED_TEST(CudaTest, DecompositionGivesConnectedClusters) {
    // several graphs: whether a bug shows up can depend on the order in which clusters converge
    struct Config { NodeIx clusters, size, bridges; };
    for (const Config cfg : {Config{6, 10, 10}, Config{8, 6, 12}, Config{5, 15, 4}, Config{10, 5, 20}}) {
        for (unsigned seed = 0; seed < 3; ++seed) {
            SCOPED_TRACE(std::to_string(cfg.clusters) + " clusters of " + std::to_string(cfg.size) + ", seed " + std::to_string(seed));
            const auto graph = makeGraph<TypeParam>(cfg.clusters * cfg.size, plantedClusters(cfg.clusters, cfg.size, cfg.bridges, seed));
            const auto labels = decomposeOnGpu(graph);

            auto copy = copyGraph(graph);
            const auto snapshot = adjacencySnapshot(copy);
            Partition host{copy};
            splitByLabels(host, labels);
            expectPartitionConsistent(host, &snapshot);
            expectClustersConnected(host);
            if (::testing::Test::HasFailure()) return;
        }
    }
}

TYPED_TEST(CudaTest, CudaExpanderDecompositionReturnsConsistentPartition) {
    // the entry point in cuda.h, used by the CLI and as decomposer of the hierarchy
    auto graph = makeGraph<TypeParam>(60, plantedClusters(6, 10, 10, 3));
    const auto snapshot = adjacencySnapshot(graph);

    Partition<TypeParam> result = cudaExpanderDecomposition(graph);
    result.updateSelfReferences();   // the clusters point to their Partition, which was returned by value

    expectPartitionConsistent(result, &snapshot);
    expectClustersConnected(result);
}

TYPED_TEST(CudaTest, NumCutEdgesMatchesHost) {
    const auto graph = makeGraph<TypeParam>(60, plantedClusters(6, 10, 10, 4));

    CudaDeviceManager cuda;
    cuda.initialize(graph);
    cuda.expanderDecomposition();
    const FinalPartition result = cuda.getFinalPartition();

    auto copy = copyGraph(graph);
    Partition host{copy};
    splitByLabels(host, result.clusterIds);

    EXPECT_EQ(cuda.getNumCutEdges(), host.getNumCutEdges());
}


// ---------------------------------------------------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------------------------------------------------

TEST(CudaWeightsTest, AllOnesWeightsBehaveExactlyLikeUnweighted) {
    auto edges = plantedClusters(5, 10, 8, 5);
    for (auto& e : edges) e.w = 1;
    const auto unweighted = makeGraph<Unweighted32>(50, edges);
    const auto weighted = makeGraph<Weighted32>(50, edges);

    // the same random walk, bit for bit
    {
        CudaDeviceManager a, b;
        a.initialize(unweighted);
        b.initialize(weighted);
        EXPECT_EQ(a.readRandomWalkValues(), b.readRandomWalkValues());
        for (int step = 0; step < 5; ++step) {
            a.iterateRandomWalk();
            b.iterateRandomWalk();
            EXPECT_EQ(a.readRandomWalkValues(), b.readRandomWalkValues()) << "after step " << step;
        }
    }

    // the same decomposition
    EXPECT_EQ(decomposeOnGpu(unweighted), decomposeOnGpu(weighted));
}

TEST(CudaWeightsTest, ContractedHierarchyLevelsDecomposeConsistently) {
    constexpr NodeIx n = 300;
    auto graph = makeGraph<Unweighted32>(n, randomEdges(n, 1200, 1, 7, true));

    random_source.seed(11);
    auto hierarchy = buildExpanderHierarchy(graph);   // CPU decomposition; levels are CsrGraph<EdgeIx>
    static_assert(std::same_as<std::remove_cvref_t<decltype(hierarchy.lvl(0).graph)>, Weighted32>);

    NodeIx checked = 0;
    for (NodeIx l = 0; l < hierarchy.height(); ++l) {
        const auto& level = hierarchy.lvl(l).graph;
        if (level.numEdges == 0) continue;
        SCOPED_TRACE("level " + std::to_string(l));

        const auto labels = decomposeOnGpu(level);

        auto copy = copyGraph(level);
        Partition host{copy};
        splitByLabels(host, labels);
        expectPartitionConsistent(host);
        expectClustersConnected(host);
        ++checked;
    }
    EXPECT_GT(checked, 0u);
}
