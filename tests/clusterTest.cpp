#include <gtest/gtest.h>

#include "core/partition.h"

template<MutableGraph Graph>
void expectClusterConsistent(const Partition<Graph>& partition, const Cluster<Graph>& cluster) {
    const Graph& graph = partition.getGraph();
    std::vector<uint8_t> inCluster(graph.numNodes);

    for (const ClusterVertex<Graph>& cv : cluster) {
        inCluster[cv.nix] = 1;
    }

    typename Graph::EdgeWeight volume{};
    typename Graph::EdgeWeight internalVolume{};

    for (const ClusterVertex<Graph>& cv : cluster) {
        EXPECT_EQ(partition.vertexFor(cv.nix).nix, cv.nix);
        EXPECT_EQ(cluster.edgeBeginIndex(cv), graph.startIndex(cv.nix));
        EXPECT_LE(cv.end, graph.endIndex(cv.nix));

        typename Graph::EdgeWeight internalDegree{};
        for (EdgeIx eix = graph.startIndex(cv.nix); eix < cv.end; ++eix) {
            EXPECT_TRUE(inCluster[graph.edges[eix]]);
            internalDegree += graph.edgeWeight(eix);
        }

        for (EdgeIx eix = cv.end; eix < graph.endIndex(cv.nix); ++eix) {
            EXPECT_FALSE(inCluster[graph.edges[eix]]);
        }

        EXPECT_EQ(cv.internalDegree, internalDegree);
        volume += graph.weightedDegree(cv.nix);
        internalVolume += internalDegree;
    }

    EXPECT_EQ(cluster.volume, volume);
    EXPECT_EQ(cluster.internalVolume, internalVolume);
}

TEST(PartitionTest, FullUnweightedPartitionIsConsistent) {
    DynamicGraph<> graph{4, 4};
    graph.addEdge(0, 1);
    graph.addEdge(1, 2);
    graph.addEdge(2, 3);
    graph.addEdge(0, 3);
    CsrGraph<> csr = graph.finalize();

    Partition partition{csr};

    ASSERT_EQ(partition.numClusters(), 1);
    EXPECT_EQ(partition.getCluster(0).volume, 2 * csr.numEdges);
    EXPECT_EQ(partition.getCluster(0).internalVolume, 2 * csr.numEdges);
    expectClusterConsistent(partition, partition.getCluster(0));
}

TEST(PartitionTest, WeightedSplitTracksCutSizeAndVolumes) {
    DynamicGraph<int_t> graph{4, 5};
    graph.addEdge(0, 1, 2);
    graph.addEdge(1, 2, 3);
    graph.addEdge(0, 2, 5);
    graph.addEdge(2, 3, 7);
    graph.addEdge(0, 3, 11);
    CsrGraph<int_t> csr = graph.finalize();

    Partition partition{csr};
    std::vector<NodeIx> labels{0, 0, 1, 1};
    std::vector<NodeIx> modified{0};

    EdgeIx cutEdges = partition.splitByIndices<true, false>(0, labels, 2, modified);

    ASSERT_EQ(modified.size(), 2);
    EXPECT_EQ(cutEdges, 3);
    EXPECT_EQ(partition.getNumCutEdges(), 3);
    EXPECT_EQ(partition.getCutSize(), 19);

    const Cluster<CsrGraph<int_t>>& left = partition.getCluster(modified[0]);
    const Cluster<CsrGraph<int_t>>& right = partition.getCluster(modified[1]);

    EXPECT_EQ(left.volume + right.volume, 56);
    EXPECT_EQ(left.internalVolume + right.internalVolume + 2 * partition.getCutSize(), 56);

    expectClusterConsistent(partition, left);
    expectClusterConsistent(partition, right);
}
