//
// Chunk 4: the public interface (deco.h). Backend choice and graph types, and that the results are valid clusterings
// that match the underlying algorithms. The CUDA cases need a CUDA device and are skipped without one.
//

#include <gtest/gtest.h>

#include <vector>

#include "testUtils.h"
#include "deco.h"

using namespace testutil;

using RealWeighted = CsrGraph<frac_t>;

namespace {

/**
 * `result` is a valid clustering of `graph`: every node has a cluster in [0, numClusters), no cluster is empty,
 * cutEdges and cutWeight count the edges between clusters, and (if `connected`) every cluster is connected.
 */
template<class GraphT, class EdgeWeight>
void expectValidClustering(const GraphT& graph, const deco::Clustering<EdgeWeight>& result, bool connected = true) {
    ASSERT_EQ(result.clusterOf.size(), graph.numNodes);
    std::vector<NodeIx> size(result.numClusters, 0);
    for (const NodeIx c : result.clusterOf) {
        ASSERT_LT(c, result.numClusters);
        ++size[c];
    }
    for (NodeIx c = 0; c < result.numClusters; ++c) EXPECT_GT(size[c], 0u) << "cluster " << c << " is empty";

    EdgeIx cutEdges = 0;
    EdgeWeight cutWeight{};
    for (NodeIx v = 0; v < graph.numNodes; ++v) {
        for (EdgeIx eix = graph.startIndex(v); eix < graph.endIndex(v); ++eix) {
            if (result.clusterOf[v] != result.clusterOf[graph.edges[eix]]) {
                ++cutEdges;
                cutWeight += graph.edgeWeight(eix);
            }
        }
    }
    EXPECT_EQ(result.cutEdges, cutEdges / 2);   // every edge was counted from both sides
    EXPECT_EQ(result.cutWeight, cutWeight / 2);

    if (connected) {
        // partitionByLabels splits every cluster into its connected pieces, so a connected clustering stays the same
        auto copy = copyGraph(graph);
        const auto partition = partitionByLabels(copy, result.clusterOf, result.numClusters);
        EXPECT_EQ(partition.numClusters(), result.numClusters) << "some cluster is not connected";
    }
}

// Two components and an isolated node: planted clusters on nodes 0..29 and on 30..59, node 60 alone
template<class GraphT>
GraphT disconnectedGraph(unsigned seed) {
    auto edges = plantedClusters(3, 10, 2, seed);
    for (const auto& e : plantedClusters(3, 10, 2, seed + 1)) edges.push_back({e.u + 30, e.v + 30, e.w});
    return makeGraph<GraphT>(61, edges);
}

// The CPU, and CUDA if a device is available
std::vector<deco::Backend> availableBackends() {
    if (deco::cudaAvailable()) return {deco::Backend::Cpu, deco::Backend::Cuda};
    return {deco::Backend::Cpu};
}

const char* name(deco::Backend backend) {
    return backend == deco::Backend::Cuda ? "CUDA" : "CPU";
}

} // namespace


// ---------------------------------------------------------------------------------------------------------------------
// Backend choice
// ---------------------------------------------------------------------------------------------------------------------

TEST(BackendTest, CudaLimitations) {
    const auto edges = twoCliques(4, 1, 3, 1);
    EXPECT_STREQ(deco::cudaLimitation(makeGraph<Unweighted32>(8, edges)), nullptr);
    EXPECT_STREQ(deco::cudaLimitation(makeGraph<Weighted32>(8, edges)), nullptr);
    EXPECT_STREQ(deco::cudaLimitation(makeGraph<Weighted64>(8, edges)), nullptr);
    EXPECT_NE(deco::cudaLimitation(makeGraph<RealWeighted>(8, edges)), nullptr);

    // a path 0 - 1 - 2 with weights w and 1: volume 2 (w + 1)
    auto path = [](IntWeight w) {
        DynamicGraph<IntWeight> dg(3, 2);
        dg.addEdge(0, 1, w);
        dg.addEdge(1, 2, 1);
        return dg.finalize();
    };
    const IntWeight largestFitting = (IntWeight{1} << 31) - 2;   // volume 2^32 - 2
    EXPECT_STREQ(deco::cudaLimitation(path(largestFitting)), nullptr);
    EXPECT_NE(deco::cudaLimitation(path(largestFitting + 1)), nullptr);   // volume 2^32
    EXPECT_NE(deco::cudaLimitation(path(0)), nullptr);
    EXPECT_NE(deco::cudaLimitation(path(-1)), nullptr);
}

TEST(BackendTest, BackendChoice) {
    const auto edges = twoCliques(4, 1);
    const auto unweighted = makeGraph<Unweighted32>(8, edges);
    const auto real = makeGraph<RealWeighted>(8, edges);

    EXPECT_EQ(deco::chooseBackend(unweighted, deco::Backend::Cpu), deco::Backend::Cpu);
    EXPECT_EQ(deco::chooseBackend(real, deco::Backend::Auto), deco::Backend::Cpu);
    EXPECT_THROW(deco::chooseBackend(real, deco::Backend::Cuda), std::invalid_argument);

    if (deco::cudaAvailable()) {
        EXPECT_EQ(deco::chooseBackend(unweighted, deco::Backend::Auto), deco::Backend::Cuda);
        EXPECT_EQ(deco::chooseBackend(unweighted, deco::Backend::Cuda), deco::Backend::Cuda);
    } else {
        EXPECT_EQ(deco::chooseBackend(unweighted, deco::Backend::Auto), deco::Backend::Cpu);
        EXPECT_THROW(deco::chooseBackend(unweighted, deco::Backend::Cuda), std::invalid_argument);
    }
}


// ---------------------------------------------------------------------------------------------------------------------
// Expander decomposition
// ---------------------------------------------------------------------------------------------------------------------

template<class GraphT>
class InterfaceTest : public ::testing::Test {};

using AllGraphTypes = ::testing::Types<Unweighted32, Weighted32, Weighted64, RealWeighted>;
TYPED_TEST_SUITE(InterfaceTest, AllGraphTypes);

TYPED_TEST(InterfaceTest, CpuDecompositionMatchesTheAlgorithm) {
    auto graph = makeGraph<TypeParam>(60, plantedClusters(6, 10, 10, 3));
    auto copy = copyGraph(graph);

    const deco::Options options{.backend = deco::Backend::Cpu};
    const auto result = deco::expanderDecomposition(graph, options);
    const auto direct = expander_decomposition::compute(copy, options.config);

    EXPECT_EQ(result.backend, deco::Backend::Cpu);
    EXPECT_GT(result.stats.iterations, 0u);
    EXPECT_EQ(result.clusterOf, clusterLabels(direct));
    EXPECT_EQ(result.numClusters, direct.numClusters());
    EXPECT_EQ(result.cutEdges, direct.getNumCutEdges());
    EXPECT_EQ(result.cutWeight, direct.getCutSize());
    expectValidClustering(graph, result);
}

TYPED_TEST(InterfaceTest, DisconnectedGraphGivesConnectedClusters) {
    for (const deco::Backend backend : availableBackends()) {
        auto graph = disconnectedGraph<TypeParam>(5);
        if (backend == deco::Backend::Cuda && deco::cudaLimitation(graph) != nullptr) continue;   // real weights
        SCOPED_TRACE(name(backend));

        const auto result = deco::expanderDecomposition(graph, {.backend = backend});
        EXPECT_EQ(result.backend, backend);
        expectValidClustering(graph, result);   // connected clusters cannot span two components
    }
}

TEST(InterfaceCudaTest, Weighted64IsDecomposedLikeWeighted32) {
    if (!deco::cudaAvailable()) GTEST_SKIP() << "no CUDA device";
    const auto edges = plantedClusters(6, 10, 10, 7);
    auto narrow = makeGraph<Weighted32>(60, edges);
    auto wide = makeGraph<Weighted64>(60, edges);   // copied to 32-bit weights for the GPU

    const deco::Options options{.backend = deco::Backend::Cuda};
    const auto a = deco::expanderDecomposition(narrow, options);
    const auto b = deco::expanderDecomposition(wide, options);

    EXPECT_EQ(a.clusterOf, b.clusterOf);
    EXPECT_EQ(a.cutEdges, b.cutEdges);
    const IntWeight narrowCutWeight = a.cutWeight;
    EXPECT_EQ(narrowCutWeight, b.cutWeight);
}


// ---------------------------------------------------------------------------------------------------------------------
// Expander hierarchy and normalized cut
// ---------------------------------------------------------------------------------------------------------------------

template<class GraphT>
class InterfaceHierarchyTest : public ::testing::Test {};

using IntegerGraphTypes = ::testing::Types<Unweighted32, Weighted32, Weighted64>;
TYPED_TEST_SUITE(InterfaceHierarchyTest, IntegerGraphTypes);

TYPED_TEST(InterfaceHierarchyTest, HierarchyIsConsistent) {
    for (const deco::Backend backend : availableBackends()) {
        SCOPED_TRACE(name(backend));
        auto graph = makeGraph<TypeParam>(100, plantedClusters(10, 10, 15, 51));

        auto hierarchy = deco::expanderHierarchy(graph, {.backend = backend});
        expectHierarchyConsistent(hierarchy, graph);
    }
}

TYPED_TEST(InterfaceHierarchyTest, NormalizedCutGivesKValidClusters) {
    for (const deco::Backend backend : availableBackends()) {
        SCOPED_TRACE(name(backend));
        auto graph = makeGraph<TypeParam>(100, plantedClusters(10, 10, 15, 51));

        const auto result = deco::normalizedCut(graph, 3, {.backend = backend});
        EXPECT_EQ(result.numClusters, 3u);
        expectValidClustering(graph, result, false);   // normalized-cut clusters need not be connected
        EXPECT_DOUBLE_EQ(result.normalizedCut, deco::normalizedCutValue(graph, result));
    }
}

TYPED_TEST(InterfaceHierarchyTest, NormalizedCutReusesTheHierarchy) {
    for (const deco::Backend backend : availableBackends()) {
        SCOPED_TRACE(name(backend));
        auto graph = makeGraph<TypeParam>(100, plantedClusters(10, 10, 15, 51));
        auto hierarchy = deco::expanderHierarchy(graph, {.backend = backend});

        const auto first = deco::normalizedCut(graph, hierarchy, 3);
        EXPECT_EQ(deco::normalizedCut(graph, hierarchy, 5).numClusters, 5u);
        const auto again = deco::normalizedCut(graph, hierarchy, 3);
        EXPECT_EQ(again.clusterOf, first.clusterOf);
        EXPECT_EQ(again.normalizedCut, first.normalizedCut);
        expectHierarchyConsistent(hierarchy, graph);   // unchanged: volumes and cuts still match the graph

        if (backend == deco::Backend::Cpu) {   // the same as with a new hierarchy (the GPU need not be deterministic)
            auto fresh = makeGraph<TypeParam>(100, plantedClusters(10, 10, 15, 51));
            const auto direct = deco::normalizedCut(fresh, 3, {.backend = backend});
            EXPECT_EQ(direct.clusterOf, first.clusterOf);
            EXPECT_EQ(direct.normalizedCut, first.normalizedCut);
        }
    }
}

TEST(InterfaceNormalizedCutTest, ValueOfTwoCliques) {
    // two 4-cliques joined by one edge: each clique has volume 2 * 6 + 1 = 13, and one edge leaves it
    for (const deco::Backend backend : availableBackends()) {
        SCOPED_TRACE(name(backend));
        auto graph = makeGraph<Unweighted32>(8, twoCliques(4, 1));

        const auto result = deco::normalizedCut(graph, 2, {.backend = backend});
        EXPECT_EQ(result.numClusters, 2u);
        EXPECT_DOUBLE_EQ(result.normalizedCut, 2.0 / 13);
    }
}
