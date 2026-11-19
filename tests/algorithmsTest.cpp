//
// Chunk 3: algorithms built on Partition: the CPU expander decomposition, contraction and the expander hierarchy,
// and the normalized cut. Runs on unweighted, 32-bit weighted and 64-bit weighted graphs.
//

#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "testUtils.h"
#include "algorithms/applications/normalized_cut.h"

using namespace testutil;

template<class GraphT>
class AlgorithmsTest : public ::testing::Test {};

using GraphTypes = ::testing::Types<Unweighted32, Weighted32, Weighted64>;
TYPED_TEST_SUITE(AlgorithmsTest, GraphTypes);


// ---------------------------------------------------------------------------------------------------------------------
// CPU expander decomposition
// ---------------------------------------------------------------------------------------------------------------------

TYPED_TEST(AlgorithmsTest, CpuDecompositionGivesConsistentConnectedClusters) {
    for (unsigned seed = 0; seed < 3; ++seed) {
        SCOPED_TRACE("seed " + std::to_string(seed));
        auto graph = makeGraph<TypeParam>(60, plantedClusters(6, 10, 10, seed));
        const auto snapshot = adjacencySnapshot(graph);

        random_source.seed(11);
        const auto result = expander_decomposition::compute(graph);

        expectPartitionConsistent(result, &snapshot);
        expectClustersConnected(result);
        if (::testing::Test::HasFailure()) return;
    }
}

TEST(CpuDecompositionTest, AllOnesWeightsGiveTheSamePartitionAsUnweighted) {
    auto edges = plantedClusters(5, 10, 8, 5);
    for (auto& e : edges) e.w = 1;
    auto unweighted = makeGraph<Unweighted32>(50, edges);
    auto weighted = makeGraph<Weighted32>(50, edges);

    random_source.seed(11);
    const auto pu = expander_decomposition::compute(unweighted);
    random_source.seed(11);
    const auto pw = expander_decomposition::compute(weighted);

    EXPECT_EQ(clusterLabels(pu), clusterLabels(pw));
}


// ---------------------------------------------------------------------------------------------------------------------
// Contraction and expander hierarchy
// ---------------------------------------------------------------------------------------------------------------------

TYPED_TEST(AlgorithmsTest, ContractGraphMergesEdgesBetweenClusters) {
    constexpr NodeIx n = 40, k = 6;
    const auto graph = makeGraph<TypeParam>(n, randomEdges(n, 150, 9, 31));

    // arbitrary cluster assignment (contractGraph does not require connected clusters)
    std::mt19937 rng(32);
    std::vector<NodeIx> parents(n);
    for (auto& p : parents) p = rng() % k;

    const auto contracted = contractGraph(graph, k, parents);

    EXPECT_EQ(contracted.numNodes, k);
    expectGraphMatchesContraction(contracted, expectedContraction(graph, parents));
}

TYPED_TEST(AlgorithmsTest, ExpanderHierarchyLevelsAreConsistent) {
    constexpr NodeIx n = 200;
    auto graph = makeGraph<TypeParam>(n, randomEdges(n, 800, 9, 41, true));

    random_source.seed(11);
    auto hierarchy = buildExpanderHierarchy(graph);

    expectHierarchyConsistent(hierarchy, graph);
}


// ---------------------------------------------------------------------------------------------------------------------
// Normalized cut
// ---------------------------------------------------------------------------------------------------------------------

TYPED_TEST(AlgorithmsTest, NormalizedCutGivesKConsistentClusters) {
    // NormalizedCut splits along tree nodes, so the first level needs at least k clusters: 10 planted clusters
    // (a random graph is usually an expander, which gives a single cluster and nothing to split)
    for (const NodeIx k : {1u, 2u, 5u, 8u}) {
        SCOPED_TRACE("k = " + std::to_string(k));
        auto graph = makeGraph<TypeParam>(100, plantedClusters(10, 10, 15, 51));

        random_source.seed(11);
        auto hierarchy = buildExpanderHierarchy(graph);
        const NodeIx top = hierarchy.height() - 1;
        ASSERT_EQ(hierarchy.lvl(top).graph.numNodes, 1u) << "the test needs a hierarchy that ends in a single node";
        ASSERT_GE(hierarchy.lvl(0).graph.numNodes, k) << "the first level has fewer than k clusters";

        const auto snapshot = adjacencySnapshot(graph);
        NormalizedCut normalizedCut{hierarchy};
        const auto partition = normalizedCut.compute(graph, k);

        ASSERT_EQ(partition.numClusters(), k);
        expectPartitionConsistent(partition, &snapshot);
        for (NodeIx c = 0; c < k; ++c) EXPECT_GT(partition.getCluster(c).size(), 0u) << "cluster " << c << " is empty";

        // The top root keeps everything that was not split off, and gets the last label. Its remaining volume in the
        // hierarchy must equal the volume of its cluster. (Regression test: split() used to subtract volumes also from
        // ancestors above the nearest root.)
        EXPECT_EQ(hierarchy.lvl(top).volumes[0], partition.getCluster(k - 1).volume);

        if (::testing::Test::HasFailure()) return;
    }
}

TEST(NormalizedCutTest, ThrowsIfThereIsNothingLeftToSplit) {
    // a random graph of average degree 8 is an expander: the first level is a single cluster, so k = 2 is impossible
    constexpr NodeIx n = 200;
    auto graph = makeGraph<Unweighted32>(n, randomEdges(n, 800, 1, 51, true));

    random_source.seed(11);
    auto hierarchy = buildExpanderHierarchy(graph);
    ASSERT_EQ(hierarchy.lvl(0).graph.numNodes, 1u) << "the test needs a graph that the decomposition keeps as one cluster";

    NormalizedCut normalizedCut{hierarchy};
    EXPECT_THROW(normalizedCut.compute(graph, 2), std::invalid_argument);
}
