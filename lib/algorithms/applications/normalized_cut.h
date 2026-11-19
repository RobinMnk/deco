//
// Created by robin on 12.12.2025.
//

#ifndef RCUT_NORMALIZED_CUT_H
#define RCUT_NORMALIZED_CUT_H

#include "algorithms/tree_cut_sparsifier/expander_hierarchy.h"

#include <set>
#include <stdexcept>
#include <string>
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

    static constexpr frac_t NO_CANDIDATE = 100000;   // penalty of "no candidate found"

    struct Candidate {
        TreeNode tn;
        frac_t penalty;
    };

    Hierarchy* hierarchy;
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
     * Traverse hierarchy from tn downwards, find the edge minimizing the cutting-penalty
     */
    Candidate getCandidate(const TreeNode& tn, EdgeWeight clusterCutSize, EdgeWeight clusterVolume) {
        Candidate bestCandidate{TreeNode{0, 0}, NO_CANDIDATE};

        if (tn.level == 0) return bestCandidate;

        for (NodeIx childIx : hierarchy->children(tn.level, tn.nix)) {
            const TreeNode child{tn.level - 1, childIx};

            if (roots.contains(child)) continue; // O(log k)

            if (volume(child) < clusterVolume) {
                frac_t edgePen = edgePenalty(child, clusterCutSize, clusterVolume);
                if (edgePen < bestCandidate.penalty) {
                    bestCandidate = {child, edgePen};
                }
            }

            Candidate cd = getCandidate(child, clusterCutSize, clusterVolume);
            if (cd.penalty < bestCandidate.penalty) {
                bestCandidate = cd;
            }
        }

        return bestCandidate;
    }

    /**
     * Assign label to all graph nodes below tn, skipping subtrees that are roots of other clusters
     */
    void assignLabels(const TreeNode& tn, NodeIx label) {
        for (NodeIx childIx : hierarchy->children(tn.level, tn.nix)) {
            if (tn.level == 0) {
                labels[childIx] = label;
            } else {
                const TreeNode child{tn.level - 1, childIx};
                if (roots.contains(child)) continue; // O(log k)
                assignLabels(child, label);
            }
        }
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

            // tn belongs to the cluster of its nearest root ancestor. Ancestors above that root no longer
            // contain this cluster (its volume was already subtracted when the root was split off) -> stop here.
            // The top-level node is always a root, so the loop terminates there at the latest.
            if (roots.contains(TreeNode{lvlIdx, parent})) break;

            child = parent;
        }
    }

public:
    explicit NormalizedCut(Hierarchy& eh) : hierarchy{&eh}, labels(eh.getNumNodes(), 0), roots() {}

    /**
     * Splits the Graph into k clusters, minimizing the normalized cut objective. Adjusts the sparsifier accordingly and returns the resulting partition.
     * @param k number of clusters
     */
    template<MutableGraph Graph>
    Partition<Graph> compute(Graph& G, NodeIx k) {
        roots.emplace(hierarchy->height() - 1, 0);

        // every split adds one cluster -> k-1 splits for k clusters
        for (NodeIx i = 1; i < k; ++i) {
            Candidate bestCandidate{TreeNode{0, 0}, NO_CANDIDATE};
            frac_t bestDelta = 100000;

            for (const TreeNode& root : roots) {
                EdgeWeight rootVol = volume(root);
                EdgeWeight rootCutSize = cutSize(root);

                Candidate cd = getCandidate(root, rootCutSize, rootVol);
                frac_t rootBias = static_cast<frac_t>(rootCutSize) / static_cast<frac_t>(rootVol);
                frac_t delta = cd.penalty - rootBias;

                if (delta < bestDelta) {
                    bestDelta = delta;
                    bestCandidate = cd;
                }

            }

            if (bestCandidate.penalty >= NO_CANDIDATE) {
                throw std::invalid_argument("NormalizedCut: cannot split into " + std::to_string(k) +
                                            " clusters, the hierarchy has no tree node left to split off");
            }

            split(bestCandidate.tn);
            roots.insert(bestCandidate.tn);
        }

        NodeIx label = 0;
        for (const TreeNode& root : roots) {
            assignLabels(root, label++);
        }

        Partition<Graph> part{G};
        std::vector<NodeIx> mod;
        // This modifies the underlying graph, which invalidates the sparsifier.
        // labels are not grouped by node order -> the cluster must be sorted by label (requiresSort = true)
        part.template splitByIndices<true, false>(0, labels, k, mod);
        return part;
    }
};

template<typename EdgeWeight>
NormalizedCut(ExpanderHierarchy<EdgeWeight>&) -> NormalizedCut<EdgeWeight>;

#endif //RCUT_NORMALIZED_CUT_H
