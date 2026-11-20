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

/**
 * Runs the decomposition loop of CudaDeviceManager::expanderDecomposition step by step and checks every phase
 * against the host: walk step, sweep cuts, state after cutting, and the labels of deactivated clusters.
 */
template<class GraphT>
void expectEveryPhaseMatchesHost(const GraphT& graph) {
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
    {
        SCOPED_TRACE("4 cliques of 12");
        expectEveryPhaseMatchesHost(makeGraph<TypeParam>(48, plantedClusters(4, 12, 6, 1)));
    }
    if (::testing::Test::HasFailure()) return;

    // The warp kernels go through the neighbours of a node 32 at a time, so degrees above 32 take a second round.
    // With 2000 nodes, Thrust's sort, scan and reduce also take their code paths for larger inputs.
    SCOPED_TRACE("50 cliques of 40");
    const auto graph = makeGraph<TypeParam>(2000, plantedClusters(50, 40, 100, 1));
    EdgeIx maxDegree = 0;
    for (NodeIx v = 0; v < graph.numNodes; ++v) maxDegree = std::max(maxDegree, graph.degree(v));
    ASSERT_GT(maxDegree, 32u) << "the graph must have nodes with more neighbours than a warp has threads";
    expectEveryPhaseMatchesHost(graph);
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

// Quality, as in algorithmsTest: expanders stay whole, and no sparse cut is left inside a cluster.
// The GPU uses the global threshold sc_threshold.

TYPED_TEST(CudaTest, DecompositionKeepsExpandersWhole) {
    ASSERT_LT(sc_threshold, 1.0 / 3) << "the cliques have no cut below 1/3, the test needs a lower threshold";
    for (const NodeIx size : {5u, 12u, 40u}) {
        SCOPED_TRACE("clique of " + std::to_string(size));
        const auto labels = decomposeOnGpu(makeGraph<TypeParam>(size, expanderClique(size)));
        EXPECT_EQ(std::set<int>(labels.begin(), labels.end()).size(), 1u);
    }
}

TYPED_TEST(CudaTest, DecompositionLeavesNoSparseCut) {
    for (const PlantedInstance& instance : plantedInstances) {
        for (unsigned seed = 0; seed < 5; ++seed) {
            SCOPED_TRACE(describe(instance, seed));
            const auto graph = makeGraph<TypeParam>(instance.clusters * instance.size,
                                                    plantedClusters(instance.clusters, instance.size, instance.bridges, seed));
            const auto labels = decomposeOnGpu(graph);

            auto copy = copyGraph(graph);
            Partition host{copy};
            splitByLabels(host, labels);
            expectNoSparsePlantedCutLeft(host, instance.size, sc_threshold);
            if (::testing::Test::HasFailure()) return;
        }
    }
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

TEST(CudaWeightsTest, WeightedEdgesBehaveLikeParallelEdges) {
    const auto edges = plantedClusters(4, 8, 6, 9);   // weights 1 to 6
    const auto weighted = makeGraph<Weighted32>(32, edges);
    const auto parallel = makeGraph<Unweighted32>(32, expandToParallelEdges(edges));

    CudaDeviceManager a, b;
    a.initialize(weighted);
    b.initialize(parallel);

    // same degrees -> same starting values, bit for bit
    EXPECT_EQ(a.downloadDegrees(), b.downloadDegrees());
    EXPECT_EQ(a.readRandomWalkValues(), b.readRandomWalkValues());

    // w * y vs. y + ... + y: equal up to rounding
    for (int step = 0; step < 5; ++step) {
        a.iterateRandomWalk();
        b.iterateRandomWalk();
        const auto va = a.readRandomWalkValues(), vb = b.readRandomWalkValues();
        for (NodeIx v = 0; v < weighted.numNodes; ++v) {
            EXPECT_NEAR(va[v], vb[v], 1e-9 * (1.0 + std::abs(va[v]))) << "node " << v << " after step " << step;
        }
    }

    // the same best sweep cut (sparsity in float on the GPU)
    a.computeSweepCuts();
    b.computeSweepCuts();
    const AllSweepCuts sa = a.readSweepCuts(), sb = b.readSweepCuts();
    ASSERT_EQ(sa.cuts.size(), 1u);
    ASSERT_EQ(sb.cuts.size(), 1u);
    EXPECT_NEAR(sa.cuts[0].sparsity, sb.cuts[0].sparsity, 1e-5 * (1.0 + sa.cuts[0].sparsity));
}

TYPED_TEST(CudaTest, HierarchyWithCudaDecomposerIsConsistent) {
    // the way the GPU decomposition is used: as decomposer of every level of the expander hierarchy
    constexpr NodeIx n = 300;
    auto graph = makeGraph<TypeParam>(n, randomEdges(n, 1200, 9, 8, true));

    auto hierarchy = buildExpanderHierarchy(graph, [](auto& level, const Config& config) {
        return cudaExpanderDecomposition(level, config);
    });

    expectHierarchyConsistent(hierarchy, graph);
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
