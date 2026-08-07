//
// Created by robin on 19.11.2025.
//

#ifndef RCUT_EXPANDER_HIERARCHY_H
#define RCUT_EXPANDER_HIERARCHY_H

#include "algorithms/expander_decomposition/cpu.h"

#include "utils/timer.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

class ExpanderHierarchy {

    Graph* fullGraph;
    bool complete = false;

    struct Level {
        Graph graph;
        Partition partition;
        std::vector<EdgeIx> volumes;        // size: number of clusters; volume of each subtree
        std::vector<EdgeWeight> cutSizes;   // size: number of clusters; value of cut leaving each cluster in the current graph
        std::vector<NodeIx> parents;        // size: numNodes of graph in level below; gives clusterId (= nix of parent) for each nix in the graph below
    };

    // levels[i].graph.numNodes == levels[i].volumes.size() == levels[i].cutSizes.size() == levels[i+1].parents.size()

    std::vector<Level> levels;

    Level contract(Partition& part) const;

    template<bool first>
    Level contract(Partition& part) const;

public:

    explicit ExpanderHierarchy(Graph* gr);

    void build();

    template<class Decomposer>
    void build(Decomposer&& decompose);

    template<typename T>
    double predict(const T& demands) const {
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
            for (const Cluster& cluster : lvl.partition) {
                double clusterDemand = 0.0;
                for (const ClusterVertex& cv : cluster) {
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

    Graph* getGraph();
    const Graph* getGraph() const;

    NodeIx height();

    const Level& lvl(NodeIx l);

    void subtractVolume(NodeIx l, NodeIx nix, EdgeIx vol);
    void addCut(NodeIx l, NodeIx nix, EdgeIx cut);
};


#endif //RCUT_EXPANDER_HIERARCHY_H
