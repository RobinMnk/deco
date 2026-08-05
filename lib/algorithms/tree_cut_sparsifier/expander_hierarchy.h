//
// Created by robin on 19.11.2025.
//

#ifndef RCUT_EXPANDER_HIERARCHY_H
#define RCUT_EXPANDER_HIERARCHY_H

#include "algorithms/expander_decomposition/cpu.h"

#include "utils/timer.h"
#include <iostream>

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
    frac_t predict(const T& demands) const {
        assert(demands.size() == fullGraph->numNodes);
        assert(complete);

        std::vector<frac_t> current(std::begin(demands), std::end(demands));
        std::vector<frac_t> next(fullGraph->numNodes, 0.0);

        frac_t max_congestion = 0.0;

        for (const Level& lvl : levels) {
            const NodeIx numClusters = lvl.graph.numNodes;
            std::fill_n(next.begin(), numClusters, static_cast<frac_t>(0.0));

            NodeIx clusterId = 0;
            for (const Cluster& cluster : lvl.partition) {
                frac_t cluster_demand = 0.0;
                for (const ClusterVertex& cv : cluster) {
                    cluster_demand += current[cv.nix];
                }

                if (lvl.cutSizes[clusterId] > 0) {
                    frac_t congestion = std::abs(cluster_demand) / lvl.cutSizes[clusterId];
                    if (congestion > max_congestion) {
                        max_congestion = congestion;
                    }
                }

                next[clusterId] = cluster_demand;
                clusterId++;
            }

            std::swap(current, next);
        }

        return max_congestion;
    }

    Graph* getGraph();

    NodeIx height();

    const Level& lvl(NodeIx l);

    void subtractVolume(NodeIx l, NodeIx nix, EdgeIx vol);
    void addCut(NodeIx l, NodeIx nix, EdgeIx cut);
};


#endif //RCUT_EXPANDER_HIERARCHY_H
