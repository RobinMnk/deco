//
// Created by robin on 25.03.25.
//

#ifndef RCUT_PARTITION_H
#define RCUT_PARTITION_H

#include "graph.h"
//#include "../utils/skiplist.h"

// Contains the information of a node within its cluster. All values are restricted to be within the same cluster, no self-loops.
struct ClusterVertex {
    NodeIx nix;
    EdgeIx end; // redundant with internalDegree
    EdgeWeight internalDegree;

    bool operator==(const ClusterVertex& other) const = default;
};

struct PrefixValues {
    EdgeIx edgeDiff;
    EdgeIx vol;
};

struct SweepCut {
    frac_t sparsity;
    NodeIx offset;
    std::vector<PrefixValues> pS;
};

class Partition;

/// Represents a single part in a Partition of a Graph
struct Cluster {
    EdgeWeight internalVolume{}, volume{};
    NodeIx nodeBegin{}, nodeEnd{};
    Partition* partition{};

    [[nodiscard]] NodeIx size() const {
        return nodeEnd - nodeBegin;
    }

    [[nodiscard]] std::vector<ClusterVertex>::iterator begin() const;

    [[nodiscard]] std::vector<ClusterVertex>::iterator end() const;

    [[nodiscard]] std::vector<NodeIx>::iterator edgeBegin(const ClusterVertex& cv) const;

    [[nodiscard]] std::vector<NodeIx>::iterator edgeEnd(const ClusterVertex& cv) const;

    bool operator==(const Cluster& other) const = default;
};

class DFS {
    std::vector<NodeIx> m_visited;
    std::vector<NodeIx> m_labels;
    std::vector<NodeIx> m_stack;
    NodeIx numCalls{0};
    const Partition* partition;

public:
    DFS(const Partition* part, NodeIx n) : m_visited(n), m_labels(n), m_stack(n), partition{part} {}

    NodeIx computeLabels(const Cluster& cluster);

    [[nodiscard]] const std::vector<NodeIx>& labels() const {
        return m_labels;
    }
};


/// Represents a Partition of a Graph, i.e., a decomposition of the graph into Clusters.
class Partition {
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
    friend Cluster;

public:
//    SkipList sl;

    explicit Partition(Graph* gr);

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
    inline void sortCluster(const Cluster& cluster, const T& values);

    [[nodiscard]] NodeIx numClusters() const {
        return static_cast<NodeIx>(clusters.size());
    }

    [[nodiscard]] const Cluster& getCluster(NodeIx clusterId) const {
        return clusters[clusterId];
    }

    [[nodiscard]] EdgeIx getNumCutEdges() const {
        return numCutEdges;
    }

    [[nodiscard]] EdgeWeight getCutSize() const {
        return cutSize;
    }

    [[nodiscard]] const Graph& getGraph() const {
        return *graph;
    }

    [[nodiscard]] const ClusterVertex& vertexFor(NodeIx nix) const {
        return clusterVertices[lookup[nix]];
    }

    [[nodiscard]] inline std::vector<Cluster>::const_iterator begin() const {
        return clusters.begin();
    }

    [[nodiscard]] inline std::vector<Cluster>::const_iterator end() const {
        return clusters.end();
    }

    void updateSelfReferences() {
        for(Cluster& cluster: clusters) {
            cluster.partition = this;
        }
    }
};

template<typename T>
void Partition::sortCluster(const Cluster &cluster, const T& values)  {
    // stable sort is actually preferred over regular sort to avoid unnecessary jumps in iterations later
    std::stable_sort(clusterVertices.begin() + cluster.nodeBegin, clusterVertices.begin() + cluster.nodeEnd,
                     [&values](const ClusterVertex& u, const ClusterVertex& v) { return values[u.nix] < values[v.nix]; }
    );
    for (NodeIx i = cluster.nodeBegin; i < cluster.nodeEnd; ++i) {
        lookup[clusterVertices[i].nix] = i;
    }
}

template<bool requiresSort, bool consolidatePieces, typename T>
EdgeIx Partition::splitByIndices(NodeIx clusterId, const T& values, NodeIx numParts, std::vector<NodeIx>& modified) {
    Cluster& current = clusters[clusterId];
    std::vector<Cluster> others(numParts, {0, 0, 0, 0, this });
    std::vector<NodeIx> counts(numParts);
    EdgeIx extraCutEdges = 0;

    for(ClusterVertex& cv: current) {
        counts[values[cv.nix]]++;
        others[values[cv.nix]].volume += graph->degree(cv.nix);
        auto eit = current.edgeBegin(cv);
        while(eit != current.edgeEnd(cv)) {
            NodeIx nb = *eit;
//            EdgeWeight wgt = fullGraph. ;; //

            // check if the neighbor is in some other cluster
            if (values[cv.nix] != values[nb]) {
                // deactivate edge from cv.nix -> nb
                cv.internalDegree--;
                cv.end--;
//                sl.del(cv.end);
                std::iter_swap(eit, current.edgeEnd(cv)); // WARN: this modifies the graph!
                extraCutEdges++;  // counts every edge twice
                continue; // do not increment eit, due to swap
            } else {
                others[values[cv.nix]].internalVolume++;
            }
            ++eit;
        }
    }
    extraCutEdges /= 2; // every edge was counted twice
    numCutEdges += extraCutEdges;
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

template<bool requiresSort, bool consolidatePieces>
void Partition::split(NodeIx clusterId, NodeIx offset, std::vector<NodeIx>& modified) {
    // prepare splitOracle to contain 0 for every node on one side and 1 for every node on the other side of the cut
    NodeIx ix = 0;
    for(const ClusterVertex& cv: clusters[clusterId]) {
        splitOracle[cv.nix] = ix > offset; // sets either 0 or 1
        ++ix;
    }
    
    splitByIndices<requiresSort, consolidatePieces>(clusterId, splitOracle, 2, modified);
}

template<typename T>
SweepCut Partition::sweepCut(NodeIx clusterId, const T& values) {
    const Cluster& current = clusters[clusterId];

    EdgeIx cutEdges = 0;
    EdgeIx cutVolume = 0;
    NodeIx offset = 0;

    SweepCut bestCut{1000, 0 };

    sortCluster(current, values);

    std::vector<PrefixValues> prefixes(current.size());

    for(const ClusterVertex& cv: current) {
        cutVolume += cv.internalDegree; // graph->degree(cv.nix);

        prefixes[offset].vol = cutVolume;

        for(auto it = current.edgeBegin(cv); it != current.edgeEnd(cv); ++it) {
            NodeIx nb = *it;
            // branchless check if edge goes across cut
            cutEdges += 1 - sc_seen[nb];
            sc_removeAt[nb] += 1 - sc_seen[nb];

//            Equivalent to this:
//            if(!sc_seen[nb]) {
//                cutEdges += 1;
//                sc_removeAt[nb]++;
//            }
        }
        cutEdges -= sc_removeAt[cv.nix]; // edge ends at nix, it no longer crosses the cut
        sc_removeAt[cv.nix] = 0;

        prefixes[offset].edgeDiff = cutEdges;
//        prefixes[offset].nix = cv.nix;

        if(cutVolume == current.internalVolume) break;

        frac_t denom = static_cast<frac_t>(std::min(cutVolume, current.internalVolume - cutVolume));

        frac_t sparsity = static_cast<frac_t>(cutEdges) / denom;

        if(sparsity < bestCut.sparsity) {
            bestCut = { sparsity, offset };
        }
        offset++;
        sc_seen[cv.nix] = 1;
    }

    for(const ClusterVertex& cv: current) sc_seen[cv.nix] = 0;

    return {bestCut.sparsity, bestCut.offset, std::move(prefixes)};
}

void checkCluster(const Cluster& cluster, const Graph& graph);

frac_t compute_normalized_cut(const Partition& part);

#endif //RCUT_PARTITION_H
