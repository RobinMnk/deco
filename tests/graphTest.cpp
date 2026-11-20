#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <variant>
#include <vector>

#include "core/graph.h"
#include "utils/graph_io.h"

TEST(GraphTest, UnweightedDynamicGraphFinalizesToCsr) {
    DynamicGraph<> graph{4, 3};
    graph.addEdge(0, 1);
    graph.addEdge(1, 2);
    graph.addEdge(2, 3);

    CsrGraph<> csr = graph.finalize();

    EXPECT_EQ(csr.numNodes, 4u);
    EXPECT_EQ(csr.numEdges, 3u);
    EXPECT_EQ(csr.degree(0), 1u);
    EXPECT_EQ(csr.degree(1), 2u);
    EXPECT_EQ(csr.weightedDegree(1), 2u);
    EXPECT_EQ(csr.edgeWeight(csr.startIndex(1)), 1u);
}

TEST(GraphTest, WeightedDynamicGraphComputesWeightedDegrees) {
    DynamicGraph<IntWeight> graph{3, 3};
    graph.addEdge(0, 1, 2);
    graph.addEdge(1, 2, 3);
    graph.addEdge(0, 2, 5);

    CsrGraph<IntWeight> csr = graph.finalize();

    EXPECT_EQ(csr.numNodes, 3u);
    EXPECT_EQ(csr.numEdges, 3u);
    EXPECT_EQ(csr.degree(0), 2u);
    EXPECT_EQ(csr.weightedDegree(0), 7);
    EXPECT_EQ(csr.weightedDegree(1), 5);
    EXPECT_EQ(csr.weightedDegree(2), 8);
}


// ---------------------------------------------------------------------------------------------------------------------
// Properties the partition and CUDA code rely on, for all graph types used in the code base
// ---------------------------------------------------------------------------------------------------------------------

#include <map>
#include <utility>

#include "testUtils.h"

using namespace testutil;

template<class GraphT>
class GraphTypedTest : public ::testing::Test {};

using GraphTypes = ::testing::Types<Unweighted32, Weighted32, Weighted64>;
TYPED_TEST_SUITE(GraphTypedTest, GraphTypes);

namespace {

// Random graph with distinct weights 1, 2, 3, ... (so that a weight attached to the wrong edge is detected).
// Not connected, so it also contains isolated nodes.
std::vector<WeightedEdge> distinctWeightEdges(NodeIx n, EdgeIx m, unsigned seed) {
    auto edges = randomEdges(n, m, 1, seed);
    for (uint32_t i = 0; i < edges.size(); ++i) edges[i].w = i + 1;
    return edges;
}

template<class GraphT>
uint64_t expectedWeight(const WeightedEdge& e) {
    return std::same_as<GraphT, Unweighted32> ? 1 : e.w;
}

} // namespace

TYPED_TEST(GraphTypedTest, FinalizeProducesSymmetricAdjacencyWithMatchingWeights) {
    constexpr NodeIx n = 50;
    const auto edges = distinctWeightEdges(n, 80, 3);
    const auto graph = makeGraph<TypeParam>(n, edges);

    std::map<std::pair<NodeIx, NodeIx>, uint64_t> weightOf;   // directed (u, v) -> weight
    std::vector<uint64_t> weightedDegree(n, 0);
    for (const auto& e : edges) {
        weightOf[{e.u, e.v}] = weightOf[{e.v, e.u}] = expectedWeight<TypeParam>(e);
        weightedDegree[e.u] += expectedWeight<TypeParam>(e);
        weightedDegree[e.v] += expectedWeight<TypeParam>(e);
    }

    ASSERT_EQ(graph.numNodes, n);
    ASSERT_EQ(graph.numEdges, edges.size());
    ASSERT_EQ(graph.ranges.size(), n + 1);
    ASSERT_EQ(graph.edges.size(), 2 * edges.size());
    EXPECT_EQ(graph.ranges[0], 0u);
    EXPECT_EQ(graph.ranges[n], 2 * edges.size());

    // every directed edge appears exactly once, at its source, with the weight of the undirected edge
    std::map<std::pair<NodeIx, NodeIx>, int> seen;
    for (NodeIx v = 0; v < n; ++v) {
        ASSERT_LE(graph.ranges[v], graph.ranges[v + 1]);
        EXPECT_EQ(graph.degree(v), graph.ranges[v + 1] - graph.ranges[v]);
        for (EdgeIx eix = graph.startIndex(v); eix < graph.endIndex(v); ++eix) {
            const NodeIx nb = graph.edges[eix];
            ASSERT_LT(nb, n);
            ASSERT_TRUE(weightOf.contains({v, nb})) << "edge " << v << " -> " << nb << " does not exist";
            EXPECT_EQ(static_cast<uint64_t>(graph.edgeWeight(eix)), weightOf.at({v, nb})) << "edge " << v << " -> " << nb;
            ++seen[{v, nb}];
        }
        EXPECT_EQ(static_cast<uint64_t>(graph.weightedDegree(v)), weightedDegree[v]) << "node " << v;
    }
    EXPECT_EQ(seen.size(), weightOf.size());
    for (const auto& [edge, count] : seen) EXPECT_EQ(count, 1) << "edge " << edge.first << " -> " << edge.second;
}

TYPED_TEST(GraphTypedTest, WeightSpansMatchPerEdgeAndPerNodeAccessors) {
    constexpr NodeIx n = 30;
    const auto graph = makeGraph<TypeParam>(n, distinctWeightEdges(n, 60, 4));

    const auto weights = graph.edgeWeights();
    const auto degrees = graph.weightedDegrees();

    if constexpr (std::same_as<TypeParam, Unweighted32>) {
        // unweighted graphs store no weights: the CUDA code relies on these being empty
        EXPECT_TRUE(weights.empty());
        EXPECT_TRUE(degrees.empty());
    } else {
        ASSERT_EQ(weights.size(), graph.edges.size());
        ASSERT_EQ(degrees.size(), n);
        for (EdgeIx eix = 0; eix < graph.edges.size(); ++eix) EXPECT_EQ(weights[eix], graph.edgeWeight(eix)) << "edge " << eix;
        for (NodeIx v = 0; v < n; ++v) EXPECT_EQ(degrees[v], graph.weightedDegree(v)) << "node " << v;
    }
}

TYPED_TEST(GraphTypedTest, SwapEdgesMovesWeightsAlong) {
    const auto edges = distinctWeightEdges(10, 30, 5);
    auto graph = makeGraph<TypeParam>(10, edges);

    // a node with at least two edges
    NodeIx v = 0;
    while (graph.degree(v) < 2) ++v;
    const EdgeIx a = graph.startIndex(v), b = graph.endIndex(v) - 1;

    const NodeIx nbA = graph.edges[a], nbB = graph.edges[b];
    const auto wA = graph.edgeWeight(a), wB = graph.edgeWeight(b);
    const auto degree = graph.weightedDegree(v);

    graph.swapEdges(a, b);

    EXPECT_EQ(graph.edges[a], nbB);
    EXPECT_EQ(graph.edges[b], nbA);
    EXPECT_EQ(graph.edgeWeight(a), wB);
    EXPECT_EQ(graph.edgeWeight(b), wA);
    EXPECT_EQ(graph.weightedDegree(v), degree);
}

TYPED_TEST(GraphTypedTest, GraphWithoutEdges) {
    const auto graph = makeGraph<TypeParam>(5, {});

    EXPECT_EQ(graph.numNodes, 5u);
    EXPECT_EQ(graph.numEdges, 0u);
    EXPECT_TRUE(graph.edges.empty());
    EXPECT_TRUE(graph.edgeWeights().empty());
    for (NodeIx v = 0; v < 5; ++v) {
        EXPECT_EQ(graph.degree(v), 0u);
        EXPECT_EQ(graph.weightedDegree(v), typename TypeParam::EdgeWeight{0});
    }
    if constexpr (!std::same_as<TypeParam, Unweighted32>) {
        EXPECT_EQ(graph.weightedDegrees().size(), 5u);
    }
}


// ---------------------------------------------------------------------------------------------------------------------
// Reading Matrix Market files
// ---------------------------------------------------------------------------------------------------------------------

namespace {

// Writes `content` to a temporary file, loads it, and removes the file again
AnyGraph loadMtx(const std::string& content) {
    static int counter = 0;
    const auto path = std::filesystem::temp_directory_path() / ("deco_graphTest_" + std::to_string(counter++) + ".mtx");
    std::ofstream(path) << content;
    struct Remove {
        std::filesystem::path path;
        ~Remove() { std::error_code ignored; std::filesystem::remove(path, ignored); }
    } remove{path};
    return load_mtx_dynamic(path.string());
}

template<class GraphT>
std::vector<NodeIx> neighborsOf(const GraphT& graph, NodeIx v) {
    return {graph.nbegin(v), graph.nend(v)};
}

} // namespace

TEST(MatrixMarketTest, SymmetricFileKeepsTheOrderOfTheEntries) {
    const AnyGraph loaded = loadMtx(
        "%%MatrixMarket matrix coordinate pattern symmetric\n"
        "% a comment\n"
        "4 4 4\n"
        "2 1\n"
        "4 2\n"
        "3 2\n"
        "3 3\n");   // self loop: skipped
    ASSERT_TRUE(std::holds_alternative<CsrGraph<>>(loaded));
    const auto& graph = std::get<CsrGraph<>>(loaded);

    EXPECT_EQ(graph.numNodes, 4u);
    EXPECT_EQ(graph.numEdges, 3u);
    EXPECT_EQ(neighborsOf(graph, 1), (std::vector<NodeIx>{0, 3, 2}));
}

TEST(MatrixMarketTest, GeneralFileGivesOneEdgePerEntry) {
    // a directed graph: the links 1 -> 2 and 2 -> 1 become two parallel edges (the graph A + A^T)
    const AnyGraph loaded = loadMtx(
        "%%MatrixMarket matrix coordinate pattern general\n"
        "3 3 3\n"
        "2 1\n"
        "1 2\n"
        "3 2\n");
    const auto& graph = std::get<CsrGraph<>>(loaded);

    EXPECT_EQ(graph.numEdges, 3u);
    EXPECT_EQ(neighborsOf(graph, 0), (std::vector<NodeIx>{1, 1}));
    EXPECT_EQ(neighborsOf(graph, 1), (std::vector<NodeIx>{0, 0, 2}));
    EXPECT_EQ(neighborsOf(graph, 2), (std::vector<NodeIx>{1}));
}

TEST(MatrixMarketTest, WeightsOfBothDirectionsAddUp) {
    const AnyGraph loaded = loadMtx(
        "%%MatrixMarket matrix coordinate integer general\n"
        "3 3 3\n"
        "1 2 5\n"
        "2 1 6\n"
        "2 3 7\n");
    ASSERT_TRUE(std::holds_alternative<CsrGraph<IntWeight>>(loaded));
    const auto& graph = std::get<CsrGraph<IntWeight>>(loaded);

    EXPECT_EQ(graph.numEdges, 3u);
    EXPECT_EQ(graph.weightedDegree(0), 11);
    EXPECT_EQ(graph.weightedDegree(1), 18);
    EXPECT_EQ(graph.weightedDegree(2), 7);
}
