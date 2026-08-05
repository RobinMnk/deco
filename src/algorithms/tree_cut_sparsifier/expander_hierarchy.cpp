#include "algorithms/tree_cut_sparsifier/expander_hierarchy.h"

ExpanderHierarchy::Level ExpanderHierarchy::contract(Partition& part) const {
    if(levels.empty()) return contract<true>(part);
    else return contract<false>(part);
}

template<bool first>
ExpanderHierarchy::Level ExpanderHierarchy::contract(Partition& part) const {
    const Graph& graph = part.getGraph();
    if constexpr (!first) {
        assert(graph.numNodes == levels.back().graph.numNodes);
    }
    std::vector<NodeIx> edges(2 * part.getNumCutEdges());
    std::vector<EdgeIx> range(part.numClusters() + 1);
    std::vector<EdgeIx> volumes(part.numClusters(), 0);
    std::vector<EdgeWeight> cutSizes(part.numClusters(), 0);
    range[0] = 0;

    NodeIx nix = 0;
    std::vector<NodeIx> clusterLookup(graph.numNodes, graph.numNodes + 1);
    for(const Cluster& cluster: part) {
        if constexpr (first) {
            volumes[nix] = cluster.volume;
        }
        for (const ClusterVertex& cv: cluster) {
            clusterLookup[cv.nix] = nix;
            if constexpr (!first) {
                volumes[nix] += levels.back().volumes[cv.nix];
            }
        }
        nix++;
    }

    EdgeIx eix = 0;
    nix = 0;
    for(const Cluster& cluster: part) {
        for (const ClusterVertex& cv: cluster) {
            for (auto it = cluster.edgeEnd(cv); it != graph.nend(cv.nix); ++it) {
                NodeIx nb = *it;
                assert(clusterLookup[nb] != nix);
                assert(clusterLookup[nb] != graph.numNodes + 1);
                edges[eix] = clusterLookup[nb];
                cutSizes[nix]++;
                eix++;
            }
        }
        assert(cutSizes[nix] == cluster.volume - cluster.internalVolume);
        nix++;
        range[nix] = eix;
    }

    return {
        Graph(std::move(edges), std::move(range), part.numClusters(), part.getNumCutEdges()),
        std::move(part),
        std::move(volumes),
        std::move(cutSizes),
        std::move(clusterLookup)
    };
}

ExpanderHierarchy::ExpanderHierarchy(Graph* gr) : fullGraph{gr} {}

void ExpanderHierarchy::build() {
    build(expanderDecomposition<>);
}

Graph* ExpanderHierarchy::getGraph() {
    return fullGraph;
}

NodeIx ExpanderHierarchy::height() {
    return static_cast<NodeIx>(levels.size());
}

const ExpanderHierarchy::Level& ExpanderHierarchy::lvl(NodeIx l) {
    return levels[l];
}

void ExpanderHierarchy::subtractVolume(NodeIx l, NodeIx nix, EdgeIx vol) {
    levels[l].volumes[nix] -= vol;
}

void ExpanderHierarchy::addCut(NodeIx l, NodeIx nix, EdgeIx cut) {
    levels[l].cutSizes[nix] += cut;
}
