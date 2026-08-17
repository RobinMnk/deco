//
// Created by robin on 12.12.2025.
//

#ifndef RCUT_NORMALIZED_CUT_H
#define RCUT_NORMALIZED_CUT_H

#include "algorithms/tree_cut_sparsifier/expander_hierarchy.h"

#include <set>
#include <vector>

template<typename EdgeWeight>
class NormalizedCut {
    using Hierarchy = ExpanderHierarchy<EdgeWeight>;
    using Graph = typename Hierarchy::Graph;

    struct TreeNode {
        NodeIx level{0};
        NodeIx nix{0};
        auto operator<=>(const TreeNode&) const = default;
    };

    struct Candidate {
        TreeNode tn;
        frac_t penalty;
    };

    Hierarchy* hierarchy;
    Graph* graph;
    std::vector<NodeIx> labels;
    std::set<TreeNode> roots; // for some reason, hashing TreeNode is very annoying ... so using set instead

    [[nodiscard]]
    EdgeWeight volume(const TreeNode& tn) const {
        return hierarchy->lvl(tn.level).volumes[tn.nix];
    }

    [[nodiscard]]
    EdgeWeight cutSize(const TreeNode& tn) const {
        return hierarchy->lvl(tn.level).cutSizes[tn.nix];
    }

    /**
     * Increase to normalized cut objective when cutting this TreeNode, assuming it belongs to a cluster with given cutSize and volume
     */
    [[nodiscard]]
    frac_t edgePenalty(const TreeNode& tn, EdgeWeight clusterCutSize, EdgeWeight clusterVolume) const {
        const EdgeWeight cutA = cutSize(tn);
        const EdgeWeight volA = volume(tn);

        const EdgeWeight cutB = clusterCutSize + cutA;
        const EdgeWeight volB = clusterVolume - volA;

        return static_cast<frac_t>(cutA) / static_cast<frac_t>(volA)
             + static_cast<frac_t>(cutB) / static_cast<frac_t>(volB);
    }

    /**
     * Traverse hierarchy from tn downwards, find the edge minimizing the cutting-penalty, assign each graph node the correct label
     */
    Candidate getCandidate(const TreeNode& tn, NodeIx label, EdgeWeight clusterCutSize, EdgeWeight clusterVolume) {
        Candidate bestCandidate{TreeNode{0, 0}, 100000};

        for (NodeIx childIx : hierarchy->children(tn.level, tn.nix)) {
            if (tn.level == 0) {
                labels[childIx] = label;
            } else {
                const TreeNode child{tn.level - 1, childIx};

                if (roots.contains(child)) continue; // O(log k)

                if (volume(child) < clusterVolume) {
                    frac_t edgePen = edgePenalty(child, clusterCutSize, clusterVolume);
                    if (edgePen < bestCandidate.penalty) {
                        bestCandidate = {child, edgePen};
                    }
                }

                Candidate cd = getCandidate(child, label, clusterCutSize, clusterVolume);
                if (cd.penalty < bestCandidate.penalty) {
                    bestCandidate = cd;
                }
            }
        }

        return bestCandidate;
    }

    void split(const TreeNode& tn) {
        const EdgeWeight volA = volume(tn);
        const EdgeWeight cutA = cutSize(tn);

        NodeIx child = tn.nix;
        NodeIx lvlIdx = tn.level;

        while (lvlIdx + 1 < hierarchy->height()) {
            ++lvlIdx;
            NodeIx parent = hierarchy->lvl(lvlIdx).parents[child];
            hierarchy->subtractVolume(lvlIdx, parent, volA);
            hierarchy->addCut(lvlIdx, parent, cutA);

            child = parent;
        }
    }

public:
    explicit NormalizedCut(Hierarchy& eh)
        : hierarchy{&eh}, graph{eh.getGraph()}, labels(graph->numNodes, 0), roots() {}

    /**
     * Splits the Graph into k clusters, minimizing the normalized cut objective. Adjusts the sparsifier accordingly and returns the resulting partition.
     * @param k number of clusters
     */
    Partition<Graph> compute(NodeIx k) {
        roots.emplace(hierarchy->height() - 1, 0);

        for (NodeIx i = 1; i <= k; ++i) {
            Candidate bestCandidate{TreeNode{0, 0}, 1000};
            NodeIx rootIx = 0;
            frac_t bestDelta = 100000;

            for (const TreeNode& root : roots) {
                EdgeWeight rootVol = volume(root);
                EdgeWeight rootCutSize = cutSize(root);

                Candidate cd = getCandidate(root, rootIx, rootCutSize, rootVol);
                frac_t rootBias = static_cast<frac_t>(rootCutSize) / static_cast<frac_t>(rootVol);
                frac_t delta = cd.penalty - rootBias;

                if (delta < bestDelta) {
                    bestDelta = delta;
                    bestCandidate = cd;
                }

                ++rootIx;
            }

            split(bestCandidate.tn);
            roots.insert(bestCandidate.tn);
        }

        Partition<Graph> part{*graph};
        std::vector<NodeIx> mod;
        // This modifies the underlying graph, which invalidates the sparsifier.
        part.template splitByIndices<false, false>(0, labels, k, mod);
        return part;
    }
};

template<typename EdgeWeight>
NormalizedCut(ExpanderHierarchy<EdgeWeight>&) -> NormalizedCut<EdgeWeight>;

#endif //RCUT_NORMALIZED_CUT_H
