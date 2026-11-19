//
// Chunk 1: Partition consistency (CPU).
//
// Every test runs on unweighted graphs, 32-bit weighted graphs (as used by contracted levels and CUDA) and
// 64-bit weighted graphs (as read from input files), and checks the full Partition state with
// expectPartitionConsistent after every operation.
//

#include <gtest/gtest.h>

#include <numeric>
#include <random>
#include <vector>

#include "testUtils.h"

using namespace testutil;

template<class GraphT>
class PartitionTest : public ::testing::Test {};

using GraphTypes = ::testing::Types<Unweighted32, Weighted32, Weighted64>;
TYPED_TEST_SUITE(PartitionTest, GraphTypes);


namespace {

// Distinct random values for the nodes, as input for sweep cuts
std::vector<double> randomValues(NodeIx n, std::mt19937& rng) {
    std::vector<double> values(n);
    std::iota(values.begin(), values.end(), 0.0);
    std::shuffle(values.begin(), values.end(), rng);
    return values;
}

// Clusters that can still be split
template<class GraphT>
std::vector<NodeIx> splittableClusters(const Partition<GraphT>& partition) {
    std::vector<NodeIx> ids;
    for (NodeIx c = 0; c < partition.numClusters(); ++c) {
        if (partition.getCluster(c).size() >= 2) ids.push_back(c);
    }
    return ids;
}

// Best sweep cut with the conventions of Partition::sweepCut: first strict minimum, starting from {1000, 0}
template<class GraphT>
SweepCut bruteForceSweepCut(const Partition<GraphT>& partition, NodeIx clusterId, const std::vector<double>& values) {
    const auto sparsities = prefixSparsities(partition, clusterId, values);
    SweepCut best{1000, 0};
    for (NodeIx offset = 0; offset < sparsities.size(); ++offset) {
        if (sparsities[offset] < best.sparsity) best = {sparsities[offset], offset};
    }
    return best;
}

} // namespace


// ---------------------------------------------------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------------------------------------------------

TYPED_TEST(PartitionTest, ConnectedGraphIsOneCluster) {
    auto graph = makeGraph<TypeParam>(10, twoCliques(5, 2, 3, 7));
    const auto snapshot = adjacencySnapshot(graph);

    Partition partition{graph};

    EXPECT_EQ(partition.numClusters(), 1u);
    EXPECT_EQ(partition.getNumCutEdges(), 0u);
    expectPartitionConsistent(partition, &snapshot);
}

TYPED_TEST(PartitionTest, DisconnectedGraphIsSplitIntoComponents) {
    // components: clique {0..3}, path {4, 5, 6}, isolated node {7}, edge {8, 9}
    std::vector<WeightedEdge> edges;
    addClique(edges, 0, 4, 2);
    edges.push_back({4, 5, 3});
    edges.push_back({5, 6, 4});
    edges.push_back({8, 9, 5});
    auto graph = makeGraph<TypeParam>(10, edges);
    const auto snapshot = adjacencySnapshot(graph);

    Partition partition{graph};

    EXPECT_EQ(partition.numClusters(), 4u);
    EXPECT_EQ(partition.getNumCutEdges(), 0u);
    expectPartitionConsistent(partition, &snapshot);
    expectClustersConnected(partition);

    const auto label = clusterLabels(partition);
    EXPECT_EQ(label[0], label[3]);
    EXPECT_EQ(label[4], label[6]);
    EXPECT_EQ(label[8], label[9]);
    EXPECT_NE(label[0], label[4]);
    EXPECT_NE(label[0], label[7]);
    EXPECT_NE(label[4], label[7]);
    EXPECT_NE(label[7], label[8]);
}


// ---------------------------------------------------------------------------------------------------------------------
// Single operations
// ---------------------------------------------------------------------------------------------------------------------

TYPED_TEST(PartitionTest, SplitByIndicesWithoutConsolidation) {
    constexpr NodeIx n = 40;
    auto graph = makeGraph<TypeParam>(n, randomEdges(n, 120, 9, 1, true));
    const auto snapshot = adjacencySnapshot(graph);
    Partition partition{graph};
    ASSERT_EQ(partition.numClusters(), 1u);

    std::mt19937 rng(2);
    std::vector<NodeIx> labels(n);
    for (auto& l : labels) l = rng() % 3;

    std::vector<NodeIx> modified;
    const EdgeIx newCutEdges = partition.template splitByIndices<true, false>(0, labels, 3, modified);

    EXPECT_EQ(partition.numClusters(), 3u);
    EXPECT_EQ(modified.size(), 2u);
    EXPECT_EQ(newCutEdges, partition.getNumCutEdges());
    expectPartitionConsistent(partition, &snapshot);

    // each part became exactly one cluster
    const auto label = clusterLabels(partition);
    for (NodeIx u = 0; u < n; ++u)
        for (NodeIx v = 0; v < n; ++v)
            EXPECT_EQ(labels[u] == labels[v], label[u] == label[v]) << "nodes " << u << ", " << v;
}

TYPED_TEST(PartitionTest, SplitByIndicesWithConsolidationSplitsDisconnectedPieces) {
    // two cliques {0..4}, {5..9}, bridge 0 -- 5.
    // part 0 = {0, 1, 2} + {7, 8} and part 1 = {3, 4} + {5, 6, 9}: each part consists of two disconnected pieces
    auto graph = makeGraph<TypeParam>(10, twoCliques(5, 1, 2, 3));
    const auto snapshot = adjacencySnapshot(graph);
    Partition partition{graph};

    const std::vector<NodeIx> labels{0, 0, 0, 1, 1, 1, 1, 0, 0, 1};
    std::vector<NodeIx> modified;
    partition.template splitByIndices<true, true>(0, labels, 2, modified);

    EXPECT_EQ(partition.numClusters(), 4u);
    EXPECT_EQ(partition.getNumCutEdges(), 13u);  // 3 * 2 in each clique + the bridge
    expectPartitionConsistent(partition, &snapshot);
    expectClustersConnected(partition);
}

TYPED_TEST(PartitionTest, SplitAtOffsetSeparatesByClusterOrder) {
    auto graph = makeGraph<TypeParam>(10, twoCliques(5, 2, 4, 1));
    const auto snapshot = adjacencySnapshot(graph);
    Partition partition{graph};

    std::vector<NodeIx> order;
    for (const auto& cv : partition.getCluster(0)) order.push_back(cv.nix);

    constexpr NodeIx offset = 3;  // positions [0, offset] stay, the rest is split off
    std::vector<NodeIx> modified;
    partition.template split<true, false>(0, offset, modified);

    ASSERT_EQ(partition.numClusters(), 2u);
    expectPartitionConsistent(partition, &snapshot);

    const auto label = clusterLabels(partition);
    for (NodeIx i = 0; i < order.size(); ++i) {
        EXPECT_EQ(label[order[i]], label[order[i <= offset ? 0 : offset + 1]]) << "position " << i;
    }
    EXPECT_NE(label[order[0]], label[order[offset + 1]]);
}

TYPED_TEST(PartitionTest, SweepCutFindsTheBridge) {
    constexpr NodeIx k = 6;
    auto graph = makeGraph<TypeParam>(2 * k, twoCliques(k, 1, 3, 1));
    const auto snapshot = adjacencySnapshot(graph);
    Partition partition{graph};

    // first clique gets the smaller values
    std::vector<double> values(2 * k);
    std::iota(values.begin(), values.end(), 0.0);

    const SweepCut cut = partition.sweepCut(0, values);

    uint64_t volumeA = 0, totalVolume = 0;
    for (NodeIx v = 0; v < 2 * k; ++v) {
        totalVolume += graph.weightedDegree(v);
        if (v < k) volumeA += graph.weightedDegree(v);
    }
    constexpr double bridgeWeight = 1.0;
    EXPECT_EQ(cut.offset, k - 1);
    EXPECT_DOUBLE_EQ(cut.sparsity, bridgeWeight / static_cast<double>(std::min(volumeA, totalVolume - volumeA)));
    expectPartitionConsistent(partition, &snapshot);

    // split at the sweep cut, as the CPU decomposition does
    std::vector<NodeIx> modified;
    partition.template split<false, true>(0, cut.offset, modified);

    ASSERT_EQ(partition.numClusters(), 2u);
    EXPECT_EQ(partition.getNumCutEdges(), 1u);
    expectPartitionConsistent(partition, &snapshot);
    expectClustersConnected(partition);
}

TYPED_TEST(PartitionTest, SweepCutMatchesBruteForce) {
    for (unsigned seed = 0; seed < 10; ++seed) {
        SCOPED_TRACE("seed " + std::to_string(seed));
        constexpr NodeIx n = 30;
        auto graph = makeGraph<TypeParam>(n, randomEdges(n, 80, 9, seed, true));
        const auto snapshot = adjacencySnapshot(graph);
        Partition partition{graph};
        std::mt19937 rng(seed);

        // split once first, so that clusters also have cut edges (sweep cuts must only use internal edges)
        std::vector<NodeIx> labels(n);
        for (auto& l : labels) l = rng() % 2;
        std::vector<NodeIx> modified;
        partition.template splitByIndices<true, true>(0, labels, 2, modified);

        for (NodeIx c : splittableClusters(partition)) {
            const auto values = randomValues(n, rng);
            const SweepCut expected = bruteForceSweepCut(partition, c, values);

            const SweepCut actual = partition.sweepCut(c, values);
            EXPECT_DOUBLE_EQ(actual.sparsity, expected.sparsity) << "cluster " << c;
            EXPECT_EQ(actual.offset, expected.offset) << "cluster " << c;

            // repeated calls give the same result (sweep-cut scratch state is reset)
            const SweepCut again = partition.sweepCut(c, values);
            EXPECT_DOUBLE_EQ(again.sparsity, actual.sparsity);
            EXPECT_EQ(again.offset, actual.offset);
        }
        expectPartitionConsistent(partition, &snapshot);
        if (::testing::Test::HasFailure()) return;
    }
}


// ---------------------------------------------------------------------------------------------------------------------
// Hand-computed weighted example (independent of the recomputation in expectPartitionConsistent)
// ---------------------------------------------------------------------------------------------------------------------

template<class GraphT>
class WeightedPartitionTest : public ::testing::Test {};

using WeightedGraphTypes = ::testing::Types<Weighted32, Weighted64>;
TYPED_TEST_SUITE(WeightedPartitionTest, WeightedGraphTypes);

TYPED_TEST(WeightedPartitionTest, SplitMatchesHandComputedCutAndVolumes) {
    using W = typename TypeParam::EdgeWeight;

    // edges (weight): 0-1 (2), 1-2 (3), 0-2 (5), 2-3 (7), 0-3 (11); split into {0, 1} and {2, 3}
    auto graph = makeGraph<TypeParam>(4, {{0, 1, 2}, {1, 2, 3}, {0, 2, 5}, {2, 3, 7}, {0, 3, 11}});
    const auto snapshot = adjacencySnapshot(graph);
    Partition partition{graph};

    const std::vector<NodeIx> labels{0, 0, 1, 1};
    std::vector<NodeIx> modified{0};
    const EdgeIx cutEdges = partition.template splitByIndices<true, false>(0, labels, 2, modified);

    ASSERT_EQ(modified.size(), 2u);
    EXPECT_EQ(cutEdges, 3u);                           // 1-2, 0-2, 0-3
    EXPECT_EQ(partition.getNumCutEdges(), 3u);
    EXPECT_EQ(partition.getCutSize(), W{19});          // 3 + 5 + 11

    const auto& left = partition.getCluster(modified[0]);   // {0, 1}
    const auto& right = partition.getCluster(modified[1]);  // {2, 3}
    EXPECT_EQ(left.volume, W{23});                     // (2 + 5 + 11) + (2 + 3)
    EXPECT_EQ(right.volume, W{33});                    // (3 + 5 + 7) + (7 + 11)
    EXPECT_EQ(left.internalVolume, W{4});              // edge 0-1, both directions
    EXPECT_EQ(right.internalVolume, W{14});            // edge 2-3, both directions

    expectPartitionConsistent(partition, &snapshot);
}


// ---------------------------------------------------------------------------------------------------------------------
// An edge of weight w behaves like w parallel edges of weight 1
// ---------------------------------------------------------------------------------------------------------------------

TEST(ParallelEdgesTest, WeightedEdgesBehaveLikeParallelEdges) {
    constexpr NodeIx n = 30;
    const auto edges = randomEdges(n, 80, 4, 21, true);
    auto weighted = makeGraph<Weighted32>(n, edges);
    auto parallel = makeGraph<Unweighted32>(n, expandToParallelEdges(edges));

    for (NodeIx v = 0; v < n; ++v) EXPECT_EQ(weighted.weightedDegree(v), parallel.weightedDegree(v)) << "node " << v;

    Partition pw{weighted};
    Partition pp{parallel};

    std::mt19937 rng(22);
    std::vector<NodeIx> labels(n);
    for (auto& l : labels) l = rng() % 3;
    std::vector<NodeIx> modifiedW, modifiedP;
    pw.template splitByIndices<true, true>(0, labels, 3, modifiedW);
    pp.template splitByIndices<true, true>(0, labels, 3, modifiedP);
    expectPartitionConsistent(pw);
    expectPartitionConsistent(pp);

    // the same clusters, with the same volumes and internal degrees
    const auto labelW = clusterLabels(pw), labelP = clusterLabels(pp);
    ASSERT_EQ(pw.numClusters(), pp.numClusters());
    for (NodeIx u = 0; u < n; ++u) {
        EXPECT_EQ(pw.vertexFor(u).internalDegree, pp.vertexFor(u).internalDegree) << "node " << u;
        for (NodeIx v = u + 1; v < n; ++v) EXPECT_EQ(labelW[u] == labelW[v], labelP[u] == labelP[v]) << "nodes " << u << ", " << v;
    }
    for (NodeIx c = 0; c < pw.numClusters(); ++c) {
        const auto& cw = pw.getCluster(c);
        if (cw.size() == 0) continue;
        const auto& cp = pp.getCluster(labelP[cw.begin()->nix]);
        EXPECT_EQ(cw.volume, cp.volume) << "cluster " << c;
        EXPECT_EQ(cw.internalVolume, cp.internalVolume) << "cluster " << c;
    }

    // cut weight equals the number of cut parallel edges
    EXPECT_EQ(pw.getCutSize(), pp.getCutSize());
    EXPECT_EQ(pp.getNumCutEdges(), pw.getCutSize());

    // identical sweep cuts (integer arithmetic on both sides)
    for (NodeIx c = 0; c < pw.numClusters(); ++c) {
        const auto& cw = pw.getCluster(c);
        if (cw.size() < 2) continue;
        const NodeIx cpId = labelP[cw.begin()->nix];
        const auto values = randomValues(n, rng);
        const SweepCut sw = pw.sweepCut(c, values);
        const SweepCut sp = pp.sweepCut(cpId, values);
        EXPECT_DOUBLE_EQ(sw.sparsity, sp.sparsity) << "cluster " << c;
        EXPECT_EQ(sw.offset, sp.offset) << "cluster " << c;
    }
}


// ---------------------------------------------------------------------------------------------------------------------
// Random sequences of operations
// ---------------------------------------------------------------------------------------------------------------------

TYPED_TEST(PartitionTest, RandomSplitSequencesKeepPartitionConsistent) {
    for (unsigned seed = 0; seed < 20; ++seed) {
        constexpr NodeIx n = 60;
        auto graph = makeGraph<TypeParam>(n, randomEdges(n, 150, 20, 100 + seed, seed % 2 == 0));
        const auto snapshot = adjacencySnapshot(graph);
        Partition partition{graph};
        std::mt19937 rng(seed);
        bool allConsolidated = true;

        for (int step = 0; step < 15; ++step) {
            SCOPED_TRACE("seed " + std::to_string(seed) + ", step " + std::to_string(step));
            const auto candidates = splittableClusters(partition);
            if (candidates.empty()) break;
            const NodeIx c = candidates[rng() % candidates.size()];
            std::vector<NodeIx> modified;

            const unsigned op = rng() % 3;
            if (op == 2) {
                // sweep cut on random values, then split there (as the CPU decomposition does)
                const auto values = randomValues(n, rng);
                const SweepCut cut = partition.sweepCut(c, values);
                partition.template split<false, true>(c, cut.offset, modified);
            } else {
                // split into random parts; op 0 consolidates the pieces, op 1 keeps them as they are
                const NodeIx parts = 2 + rng() % 2;
                std::vector<NodeIx> labels(n);
                for (auto& l : labels) l = rng() % parts;
                if (op == 0) {
                    partition.template splitByIndices<true, true>(c, labels, parts, modified);
                } else {
                    partition.template splitByIndices<true, false>(c, labels, parts, modified);
                    allConsolidated = false;
                }
            }

            expectPartitionConsistent(partition, &snapshot);
            if (allConsolidated) expectClustersConnected(partition);
            if (::testing::Test::HasFailure()) return;
        }
    }
}
