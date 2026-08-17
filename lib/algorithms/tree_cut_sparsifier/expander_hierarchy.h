//
// Created by robin on 19.11.2025.
//

#ifndef RCUT_EXPANDER_HIERARCHY_H
#define RCUT_EXPANDER_HIERARCHY_H

#include "algorithms/expander_decomposition/cpu.h"
#include "utils/timer.h"

#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

template<MutableGraph Graph>
class ExpanderHierarchy {

    using EdgeWeight = typename Graph::EdgeWeight;

    struct Level {
        Graph graph;
        Partition<Graph> partition;
        std::vector<EdgeIx> volumes;        // size: number of clusters; volume of each subtree
        std::vector<EdgeWeight> cutSizes;   // size: number of clusters; value of cut leaving each cluster in the current graph
        std::vector<NodeIx> parents;        // size: numNodes of graph in level below; gives clusterId (= nix of parent) for each nix in the graph below
    };

    // levels[i].graph.numNodes == levels[i].volumes.size() == levels[i].cutSizes.size() == levels[i+1].parents.size()

    std::vector<Level> levels;
    Graph* fullGraph;
    bool complete = false;

    Level contract(Partition<Graph>& part) const;

    template<bool first>
    Level contract(Partition<Graph>& part) const;

public:

    explicit ExpanderHierarchy(Graph& gr) : fullGraph{&gr} {}

    void build() {
        build(expander_decomposition::compute);
    }

    template<class Decomposer>
    void build(Decomposer&& decompose);

    template<typename T>
    double predict(const T& demands) const;

    size_t height() {
        return levels.size();
    }

    const Level& lvl(NodeIx l) const {
        return levels[l];
    }

    void subtractVolume(NodeIx l, NodeIx nix, EdgeIx vol) {
        levels[l].volumes[nix] -= vol;
    }

    void addCut(NodeIx l, NodeIx nix, EdgeIx cut) {
        levels[l].cutSizes[nix] += cut;
    }

    Graph* getGraph() {
        return fullGraph;
    }
    const Graph* getGraph() const {
        return fullGraph;
    }
};

template<MutableGraph Graph>
ExpanderHierarchy<Graph>::Level ExpanderHierarchy<Graph>::contract(Partition<Graph>& part) const {
    if(levels.empty()) return contract<true>(part);
    else return contract<false>(part);
}


template<MutableGraph Graph>
template<bool first>
ExpanderHierarchy<Graph>::Level ExpanderHierarchy<Graph>::contract(Partition<Graph>& part) const {
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
    for(const Cluster<Graph>& cluster: part) {
        if constexpr (first) {
            volumes[nix] = cluster.volume;
        }
        for (const ClusterVertex<Graph>& cv: cluster) {
            clusterLookup[cv.nix] = nix;
            // assert(clusterLookup[cv.nix] == part.cvForNix(cv.nix)); // TODO: clusterLookup is redundant with Partition.lookup ?
            if constexpr (!first) {
                volumes[nix] += levels.back().volumes[cv.nix];
            }
        }
        nix++;
    }

    EdgeIx eix = 0;
    nix = 0;
    for(const Cluster<Graph>& cluster: part) {
        for (const ClusterVertex<Graph>& cv: cluster) {
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



template<MutableGraph Graph>
template<class Decomposer>
void ExpanderHierarchy<Graph>::build(Decomposer&& decompose) {
    Graph* current = fullGraph;
    Config conf{};
    Timer t;

    do {
        t.start();
        Partition ed = decompose(*current, conf);

        if(ed.getNumCutEdges() > 0.95 * current->numEdges) {
            conf.targetSparsity /= 2;
            WARN("\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t Adjusting targetSparsity to " << conf.targetSparsity);
            continue;
        }

        NodeIx numClusters = ed.numClusters();
        EdgeIx cutSize = ed.getNumCutEdges();

        Level lvl = contract(ed);
        levels.push_back(lvl); // warning: may invalidate Partition (contains self-pointers)!
        current = &levels.back().graph;

        auto timeSpent = t.timeSeconds();
        // WARN("Level " << levels.size() << " (" << timeSpent << "s)\t\t-> " << numClusters << " clusters with " << cutSize << " crossing edges");
    } while(current->numNodes > 1);


    // a bit ugly
    for(Level& lvl : levels) {
        lvl.partition.updateSelfReferences();
    }

    complete = true;

    // WARN("Completed Hierarchy");
}


template<MutableGraph Graph>
template<typename T>
double ExpanderHierarchy<Graph>::predict(const T& demands) const {
    assert(demands.size() == fullGraph->numNodes);
    assert(complete);

    std::vector<double> current(std::begin(demands), std::end(demands));
    std::vector<double> next(fullGraph->numNodes, 0.0);

    const double totalDemand = std::accumulate(current.begin(), current.end(), 0.0);
    if (std::abs(totalDemand) > 1e-8) {
        throw std::invalid_argument("Tree cut sparsifier prediction requires demand to sum to zero.");
    }

    double maxCongestion = 0.0;

    for (const Level& lvl : levels) {
        const NodeIx numClusters = lvl.graph.numNodes;
        std::fill_n(next.begin(), numClusters, 0.0);

        NodeIx clusterId = 0;
        for (const Cluster<Graph>& cluster : lvl.partition) {
            double clusterDemand = 0.0;
            for (const ClusterVertex<Graph>& cv : cluster) {
                clusterDemand += current[cv.nix];
            }

            if (lvl.cutSizes[clusterId] > 0) {
                const double congestion = std::abs(clusterDemand) / static_cast<double>(lvl.cutSizes[clusterId]);
                maxCongestion = std::max(maxCongestion, congestion);
            } else if (std::abs(clusterDemand) > 1e-8) {
                return std::numeric_limits<double>::infinity();
            }

            next[clusterId] = clusterDemand;
            clusterId++;
        }

        std::swap(current, next);
    }

    return maxCongestion;
}



#endif //RCUT_EXPANDER_HIERARCHY_H
