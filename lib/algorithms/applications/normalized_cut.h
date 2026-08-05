//
// Created by robin on 12.12.2025.
//

#ifndef RCUT_NORMALIZED_CUT_H
#define RCUT_NORMALIZED_CUT_H

#include "algorithms/tree_cut_sparsifier/expander_hierarchy.h"
#include <set>


class NormalizedCut {

    struct TreeNode {
        NodeIx level{0};
        NodeIx nix{0};
        auto operator<=>(const TreeNode&) const = default;
    };

    struct Candidate {
        TreeNode tn;
        frac_t penalty;
    };

    ExpanderHierarchy* hierarchy;
    Graph* graph;
    std::vector<NodeIx> labels;
    std::set<TreeNode> roots; // for some reason, hashing TreeNode is very annoying ... so using set instead

    [[nodiscard]] const Cluster& children(const TreeNode& tn) const;
    [[nodiscard]] EdgeIx volume(const TreeNode& tn) const;
    [[nodiscard]] EdgeIx cutSize(const TreeNode& tn) const;

    /**
     * Increase to normalized cut objective when cutting this TreeNode, assuming it belongs to a cluster with given cutSize and volume
     */
    [[nodiscard]] frac_t edgePenalty(const TreeNode& tn, EdgeIx clusterCutSize, EdgeIx clusterVolume) const;

    /**
     * Traverse hierarchy from tn downwards, find the edge minimizing the cutting-penalty, assign each graph node the correct label
     */
    Candidate getCandidate(const TreeNode& tn, NodeIx label, EdgeIx clusterCutSize, EdgeIx clusterVolume);

    void split(const TreeNode& tn);

public:
    explicit NormalizedCut(ExpanderHierarchy* eh);

    /**
     * Splits the Graph into k clusters, minimizing the normalized cut objective. Adjusts the sparsifier accordingly and returns the resulting partition.
     * @param k number of clusters
     */
    Partition compute(NodeIx k);
};


#endif //RCUT_NORMALIZED_CUT_H
