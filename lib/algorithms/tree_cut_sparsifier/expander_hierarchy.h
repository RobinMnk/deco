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
    void build(Decomposer&& decompose) {
        Graph* current = fullGraph;
        Config conf{};
        Timer t;

        do {
            t.start();
            Partition ed = decompose(*current, conf);

            if(ed.getNumCutEdges() > 0.95 * current->numEdges) {
                conf.targetSparsity *= 0.8;
                WARN("\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t Adjusting targetSparsity to " << conf.targetSparsity);
                continue;
            }

            NodeIx numClusters = ed.numClusters();
            EdgeIx cutSize = ed.getNumCutEdges();

            Level lvl = contract(ed);
            levels.push_back(lvl); // warning: may invalidate Partition (contains self-pointers)!
            current = &levels.back().graph;

            auto timeSpent = t.timeSeconds();
            WARN("Level " << levels.size() << " (" << timeSpent << "s)\t\t-> " << numClusters << " clusters with " << cutSize << " crossing edges");
        } while(current->numNodes > 1);


        // a bit ugly
        for(Level& lvl : levels) {
            lvl.partition.updateSelfReferences();
        }
        WARN("Completed Hierarchy");
    }

    Graph* getGraph();

    NodeIx height();

    const Level& lvl(NodeIx l);

    void subtractVolume(NodeIx l, NodeIx nix, EdgeIx vol);
    void addCut(NodeIx l, NodeIx nix, EdgeIx cut);
};


#endif //RCUT_EXPANDER_HIERARCHY_H
