#include <gtest/gtest.h>

#include "core/graph.h"

TEST(GraphTest, UnweightedDynamicGraphFinalizesToCsr) {
    DynamicGraph<> graph{4, 3};
    graph.addEdge(0, 1);
    graph.addEdge(1, 2);
    graph.addEdge(2, 3);

    CsrGraph<> csr = graph.finalize();

    EXPECT_EQ(csr.numNodes, 4);
    EXPECT_EQ(csr.numEdges, 3);
    EXPECT_EQ(csr.degree(0), 1);
    EXPECT_EQ(csr.degree(1), 2);
    EXPECT_EQ(csr.weightedDegree(1), 2);
    EXPECT_EQ(csr.edgeWeight(csr.startIndex(1)), 1);
}

TEST(GraphTest, WeightedDynamicGraphComputesWeightedDegrees) {
    DynamicGraph<IntWeight> graph{3, 3};
    graph.addEdge(0, 1, 2);
    graph.addEdge(1, 2, 3);
    graph.addEdge(0, 2, 5);

    CsrGraph<IntWeight> csr = graph.finalize();

    EXPECT_EQ(csr.numNodes, 3);
    EXPECT_EQ(csr.numEdges, 3);
    EXPECT_EQ(csr.degree(0), 2);
    EXPECT_EQ(csr.weightedDegree(0), 7);
    EXPECT_EQ(csr.weightedDegree(1), 5);
    EXPECT_EQ(csr.weightedDegree(2), 8);
}
