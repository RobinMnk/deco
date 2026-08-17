//
// Created by robin on 25.03.25.
//

#ifndef RCUT_PARTITION_H
#define RCUT_PARTITION_H

#include "graph.h"
//#include "../utils/skiplist.h"


struct SweepCut {
    frac_t sparsity;
    NodeIx offset;
};


/// Represents a Partition of a Graph, i.e., a decomposition of the graph into Clusters.
template<MutableGraph Graph>
class Partition {
public:
    using EdgeWeight = typename Graph::EdgeWeight;

    /// Contains the information of a node within its cluster. All values are restricted to be within the same cluster, no self-loops.
    struct ClusterVertex {
        NodeIx nix;
        EdgeIx end;
        EdgeWeight internalDegree;

        bool operator==(const ClusterVertex& other) const = default;
    };

    /// Represents a single part in a Partition of a Graph
    struct Cluster {
        EdgeWeight internalVolume{}, volume{};
        NodeIx nodeBegin{}, nodeEnd{};
        Partition* partition{};

        [[nodiscard]]
        NodeIx size() const {
            return nodeEnd - nodeBegin;
        }

        [[nodiscard]]
        std::vector<ClusterVertex>::iterator begin() const {
            return partition->clusterVertices.begin() + nodeBegin;
        }

        [[nodiscard]]
        std::vector<ClusterVertex>::iterator end() const {
            return partition->clusterVertices.begin() + nodeEnd;
        }

        [[nodiscard]]
        EdgeIx edgeBeginIndex(const ClusterVertex& cv) const {
            return partition->graph->startIndex(cv.nix);
        }

        [[nodiscard]]
        std::vector<NodeIx>::iterator edgeBegin(const ClusterVertex& cv) const {
            return partition->graph->nbegin(cv.nix);
        }

        [[nodiscard]]
        std::vector<NodeIx>::iterator edgeEnd(const ClusterVertex& cv) const {
            return partition->graph->edges.begin() + cv.end;
        }

        bool operator==(const Cluster& other) const = default;
    };


private:
    class DFS {
        std::vector<NodeIx> m_visited;
        std::vector<NodeIx> m_labels;
        std::vector<NodeIx> m_stack;
        NodeIx numCalls{0};
        const Partition* partition;

    public:
        DFS(const Partition* part, NodeIx n) : m_visited(n), m_labels(n), m_stack(n), partition{part} {}

        NodeIx computeLabels(const Cluster& cluster);

        [[nodiscard]]
        const std::vector<NodeIx>& labels() const {
            return m_labels;
        }
    };

    // guarantees that the index of any Cluster never changes (i.e., no deletions)
    std::vector<Cluster> clusters;
    std::vector<ClusterVertex> clusterVertices;
    std::vector<uint16_t> splitOracle;
    std::vector<NodeIx> lookup; // gives the index in permutation for a given NodeIx
    std::vector<uint8_t> sc_seen; // tracks seen nodes in sweepCut procedure. Note: vector<uint8_t> is faster than vector<bool> but uses 8x memory
    std::vector<EdgeIx> sc_removeAt; // tracks number of edges ending at given node, used in sweepCut procedure.
    EdgeIx numCutEdges;
    EdgeWeight cutSize;
    DFS dfs;
    Graph* graph;

public:
//    SkipList sl;

    explicit Partition(Graph& gr);

    /// splits Cluster with given id into its connected components and adds resulting clusters to partition and their id to given list
    void consolidate(NodeIx clusterId, std::vector<NodeIx>& modified);

    /// Cuts this cluster at the given offset index (wrt the permutation vector)
    template<bool requiresSort=true, bool consolidatePieces=true>
    void split(NodeIx clusterId, NodeIx offset, std::vector<NodeIx>& modified);

    /**
     * Splits the given cluster into numParts subclusters based on the values. The values are required to be indices in [0, numParts). Returns number of cut edges.
     * May modify the graph!
     */
    template<bool requiresSort=true, bool consolidatePieces=true, typename T>
    EdgeIx splitByIndices(NodeIx clusterId, const T& values, NodeIx numParts, std::vector<NodeIx>& modified);

    template<typename T>
    SweepCut sweepCut(NodeIx clusterId, const T& values);

    template<typename T>
    void sortCluster(const Cluster& cluster, const T& values);

    [[nodiscard]]
    NodeIx numClusters() const {
        return static_cast<NodeIx>(clusters.size());
    }

    [[nodiscard]]
    const Cluster& getCluster(NodeIx clusterId) const {
        return clusters[clusterId];
    }

    [[nodiscard]]
    EdgeIx getNumCutEdges() const {
        return numCutEdges;
    }

    [[nodiscard]]
    EdgeWeight getCutSize() const {
        return cutSize;
    }

    [[nodiscard]]
    const Graph& getGraph() const {
        return *graph;
    }

    [[nodiscard]]
    const ClusterVertex& vertexFor(NodeIx nix) const {
        return clusterVertices[lookup[nix]];
    }

    [[nodiscard]]
    std::vector<Cluster>::const_iterator begin() const {
        return clusters.begin();
    }

    [[nodiscard]]
    std::vector<Cluster>::const_iterator end() const {
        return clusters.end();
    }

    void updateSelfReferences() {
        for(Cluster& cluster: clusters) {
            cluster.partition = this;
        }
    }
}; // class Partition


template<MutableGraph Graph>
using Cluster = typename Partition<Graph>::Cluster;

template<MutableGraph Graph>
using ClusterVertex = typename Partition<Graph>::ClusterVertex;

template<MutableGraph Graph>
Partition<Graph>::Partition(Graph& gr) :
        clusters(1),
        clusterVertices(gr.numNodes),
        splitOracle(gr.numNodes),
        lookup(gr.numNodes),
        sc_seen(gr.numNodes),
        sc_removeAt(gr.numNodes),
        numCutEdges{0},
        cutSize{0},
        dfs(this, gr.numNodes),
        graph{&gr}
//        sl(2 * gr.numEdges)
{
    EdgeWeight volume{0};

    for(NodeIx nix = 0; nix < gr.numNodes; ++nix) {
        EdgeIx end = gr.ranges[nix+1];
        auto degree = static_cast<EdgeWeight>(end - gr.ranges[nix]);
        clusterVertices[nix] = {
            nix, end, degree
    };
        volume += degree;
        lookup[nix] = nix;
    }

    clusters[0] = {volume, volume, 0, gr.numNodes, this};

    std::vector<NodeIx> mod; // a bit ugly
    // consolidate(0, mod);
}

template<MutableGraph Graph>
void Partition<Graph>::consolidate(NodeIx clusterId, std::vector<NodeIx>& modified) {
    NodeIx subclusters = dfs.computeLabels(clusters[clusterId]);
    if(subclusters > 1) {
        INFO("Cluster consists of " << subclusters << " subclusters. Splitting.");
        splitByIndices<true, false>(clusterId, dfs.labels(), subclusters, modified);
    }
}

template<MutableGraph Graph>
template<typename T>
void Partition<Graph>::sortCluster(const Cluster &cluster, const T& values)  {
    // stable sort is actually preferred over regular sort to avoid unnecessary jumps in iterations later
    std::stable_sort(clusterVertices.begin() + cluster.nodeBegin, clusterVertices.begin() + cluster.nodeEnd,
                     [&values](const ClusterVertex& u, const ClusterVertex& v) { return values[u.nix] < values[v.nix]; }
    );
    for (NodeIx i = cluster.nodeBegin; i < cluster.nodeEnd; ++i) {
        lookup[clusterVertices[i].nix] = i;
    }
}

template<MutableGraph Graph>
template<bool requiresSort, bool consolidatePieces, typename T>
EdgeIx Partition<Graph>::splitByIndices(NodeIx clusterId, const T& values, NodeIx numParts, std::vector<NodeIx>& modified) {
    Cluster& current = clusters[clusterId];
    std::vector<Cluster> others(numParts, {0, 0, 0, 0, this});
    std::vector<NodeIx> counts(numParts);
    EdgeIx extraCutEdges = 0;
    EdgeWeight extraCutWeight{0};

    for(ClusterVertex& cv: current) {
        const NodeIx index = values[cv.nix];
        ++counts[index];
        others[index].volume += graph->weightedDegree(cv.nix);

        for (EdgeIx eix = current.edgeBeginIndex(cv); eix < cv.end; /*no increment of eix*/) {
            NodeIx nb = graph->edges[eix];
            EdgeWeight wgt = graph->edgeWeight(eix);

            // check if the neighbor is in some other cluster
            if (index != values[nb]) {
                // deactivate edge from cv.nix -> nb
                cv.internalDegree -= wgt;
                --cv.end;
//                sl.del(cv.end);

                graph->swapEdges(eix, cv.end);       // WARN: this modifies the graph (not semantically)!

                extraCutEdges++;  // counts every edge twice
                extraCutWeight += wgt;
                continue; // do not increment eix, due to swap
            } else {
                others[index].internalVolume += wgt;
                ++eix;
            }
        }
    }

    // every edge was counted twice
    extraCutEdges /= 2;
    extraCutWeight /= 2;

    numCutEdges += extraCutEdges;
    cutSize += extraCutWeight;

    if constexpr (requiresSort)
        sortCluster(current, values);

    // overwrite values for current cluster
    current.nodeEnd = current.nodeBegin + counts[0];
    current.volume = others[0].volume;
    current.internalVolume = others[0].internalVolume;

//    checkCluster(current, *graph);

    // insert new clusters
    NodeIx start = current.nodeEnd;
    for(NodeIx i = 1; i < numParts; ++i) {
        others[i].nodeBegin = start;
        start += counts[i];
        others[i].nodeEnd = start;

        auto nextId = static_cast<NodeIx>(clusters.size());
        modified.push_back(nextId);
        clusters.push_back(others[i]);

        if constexpr (consolidatePieces)
            consolidate(nextId, modified);
//        checkCluster(others[i], *graph);
    }

    if constexpr (consolidatePieces)
        consolidate(clusterId, modified);

    return extraCutEdges;
}

template<MutableGraph Graph>
template<bool requiresSort, bool consolidatePieces>
void Partition<Graph>::split(NodeIx clusterId, NodeIx offset, std::vector<NodeIx>& modified) {
    // prepare splitOracle to contain 0 for every node on one side and 1 for every node on the other side of the cut
    NodeIx ix = 0;
    for(const ClusterVertex& cv: clusters[clusterId]) {
        splitOracle[cv.nix] = ix > offset; // sets either 0 or 1
        ++ix;
    }
    
    splitByIndices<requiresSort, consolidatePieces>(clusterId, splitOracle, 2, modified);
}

template<MutableGraph Graph>
template<typename T>
SweepCut Partition<Graph>::sweepCut(NodeIx clusterId, const T& values) {
    const Cluster& current = clusters[clusterId];

    EdgeWeight cutEdges = 0;
    EdgeWeight cutVolume = 0;
    NodeIx offset = 0;

    SweepCut bestCut{1000, 0 };

    sortCluster(current, values);

    for(const ClusterVertex& cv: current) {
        cutVolume += cv.internalDegree; // graph->degree(cv.nix);

        for (EdgeIx eix = current.edgeBeginIndex(cv); eix < cv.end; ++eix) {
            NodeIx nb = graph->edges[eix];
            EdgeWeight wgt = graph->edgeWeight(eix);

            // sc_seen[nb] is either 0 or 1
            cutEdges += wgt * (1 - sc_seen[nb]);
            sc_removeAt[nb] += wgt * (1 - sc_seen[nb]);

//            Equivalent to this:
//            if(!sc_seen[nb]) {
//                cutEdges += wgt;
//                sc_removeAt[nb] += wgt;
//            }
        }

//         for(auto it = current.edgeBegin(cv); it != current.edgeEnd(cv); ++it) {
//             NodeIx nb = *it;
//             // branchless check if edge goes across cut
//             cutEdges += 1 - sc_seen[nb];
//             sc_removeAt[nb] += 1 - sc_seen[nb];
//
// //            Equivalent to this:
// //            if(!sc_seen[nb]) {
// //                cutEdges += 1;
// //                sc_removeAt[nb]++;
// //            }
//         }

        cutEdges -= sc_removeAt[cv.nix]; // edge ends at nix, it no longer crosses the cut
        sc_removeAt[cv.nix] = 0; // reset for next call

        // cut contains entire cluster -> no more cut, break (and account for numerical imprecision)
        if(cutVolume >= current.internalVolume - 1e-8) break;

        frac_t denom = static_cast<frac_t>(std::min(cutVolume, current.internalVolume - cutVolume));
        frac_t sparsity = static_cast<frac_t>(cutEdges) / denom;

        if(sparsity < bestCut.sparsity) {
            bestCut = { sparsity, offset };
        }

        offset++;
        sc_seen[cv.nix] = 1;
    }

    // reset for next call
    for(const ClusterVertex& cv: current) sc_seen[cv.nix] = 0;

    return {bestCut.sparsity, bestCut.offset};
}

template<MutableGraph Graph>
NodeIx Partition<Graph>::DFS::computeLabels(const Cluster& cluster) {
    NodeIx numComponents = 0;
    NodeIx qix = 0;

    numCalls++;

    for (const ClusterVertex& cv: cluster) {
        if (m_visited[cv.nix] == numCalls) {
            continue;
        }

        m_stack[qix++] = cv.nix;
        m_visited[cv.nix] = numCalls;
        m_labels[cv.nix] = numComponents;

        while (qix) {
            NodeIx u = m_stack[--qix];
            const ClusterVertex& cvx = partition->vertexFor(u);

            for (auto it = cluster.edgeBegin(cvx); it != cluster.edgeEnd(cvx); ++it) {
                NodeIx nb = *it;

                if (m_visited[nb] < numCalls) {
                    m_visited[nb] = numCalls;
                    m_labels[nb] = numComponents;
                    m_stack[qix++] = nb;
                }
            }
        }

        numComponents++;
    }

    return numComponents;
}

template<typename Graph>
void checkCluster(const Cluster<Graph>& cluster, const Graph& graph) {
    EdgeIx sumDegrees = 0, summedVol = 0;
    assert(cluster.nodeBegin <= cluster.nodeEnd);
    for(const ClusterVertex<Graph>& cv: cluster) {
        assert(cluster.edgeBegin(cv) == graph.nbegin(cv.nix));
        assert(cv.end == std::distance(graph.edges.begin(), graph.nbegin(cv.nix) + cv.internalDegree));
        assert(cv.end <= std::distance(graph.edges.begin(), graph.nend(cv.nix)));

        for(auto it = cluster.edgeBegin(cv); it != cluster.edgeEnd(cv); ++it) {
            NodeIx nb = *it;
            auto itx = std::find_if(graph.nbegin(cv.nix), graph.nend(cv.nix),
                                    [&nb](NodeIx otherNix) { return otherNix == nb; } );
            assert(itx != graph.nend(cv.nix));

            auto it2 = std::find_if(cluster.begin(), cluster.end(),
                                    [&nb](const ClusterVertex<Graph>& node) { return node.nix == nb; } );
            assert(it2 != cluster.end());
        }

        assert(cv.internalDegree == std::distance(cluster.edgeBegin(cv), cluster.edgeEnd(cv)));
        sumDegrees += cv.internalDegree;
        summedVol += graph.degree(cv.nix);
    }
    assert(summedVol == cluster.volume);
    assert(sumDegrees == cluster.internalVolume);
}

template<typename Graph>
frac_t compute_normalized_cut(const Partition<Graph>& part) {
    frac_t normalized_cut = 0.0;

    for(const Cluster<Graph>& cl: part) {
        if(cl.volume == 0) continue;
        normalized_cut += static_cast<frac_t>(cl.volume - cl.internalVolume) / cl.volume;
    }

    return normalized_cut;
}


#endif //RCUT_PARTITION_H
