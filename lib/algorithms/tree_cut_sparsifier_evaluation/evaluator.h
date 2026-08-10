//
// Created by robin on 05.08.2026.
//

#ifndef DECO_EVALUATOR_H
#define DECO_EVALUATOR_H

#include "algorithms/tree_cut_sparsifier/expander_hierarchy.h"


class Evaluator {


public:
    double estimate(const Graph& graph, const ExpanderHierarchy& eh, NodeIx numNodes) {
        RandomWalk rw(graph);
        double max_cong = 0.0;

        for (int i = 0; i < 50; i++) {
            auto& demand = rw.values();
            frac_t cong = eh.predict(demand);
            std::cout << "Congestion: " << cong << std::endl;
            if (cong > max_cong) max_cong = cong;
            rw.init_distribution(graph);
        }

        return max_cong;
    }
};


#endif //DECO_EVALUATOR_H