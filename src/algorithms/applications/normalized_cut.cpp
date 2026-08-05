#include "algorithms/applications/normalized_cut.h"

const Cluster& NormalizedCut::children(const TreeNode& tn) const {
    return hierarchy->lvl(tn.level).partition.getCluster(tn.nix);
}

EdgeIx NormalizedCut::volume(const TreeNode& tn) const {
    return hierarchy->lvl(tn.level).volumes[tn.nix];
}

EdgeIx NormalizedCut::cutSize(const TreeNode& tn) const {
    return hierarchy->lvl(tn.level).cutSizes[tn.nix];
}

frac_t NormalizedCut::edgePenalty(const TreeNode& tn, EdgeIx clusterCutSize, EdgeIx clusterVolume) const {
    const EdgeWeight cutA = hierarchy->lvl(tn.level).cutSizes[tn.nix];
    const EdgeIx     volA = hierarchy->lvl(tn.level).volumes[tn.nix];

    const EdgeWeight cutB = clusterCutSize + cutA;
    const EdgeIx     volB = clusterVolume - volA;

    return static_cast<frac_t>(cutA) / volA + static_cast<frac_t>(cutB) / volB;
}

NormalizedCut::Candidate NormalizedCut::getCandidate(const TreeNode& tn, NodeIx label, EdgeIx clusterCutSize, EdgeIx clusterVolume) {
    Candidate bestCandidate{TreeNode{0, 0}, 100000};

    for(const ClusterVertex& cv: children(tn)) {
        if(tn.level == 0) {
            labels[cv.nix] = label;
        } else {
            const TreeNode child{tn.level - 1, cv.nix};

            if(roots.contains(child)) continue; // O(log k)

            if(volume(child) < clusterVolume) {
                frac_t edgePen = edgePenalty(child, clusterCutSize, clusterVolume);
                if(edgePen < bestCandidate.penalty) {
                    bestCandidate = { child, edgePen };
                }
            }

            Candidate cd = getCandidate(child, label, clusterCutSize, clusterVolume);
            if(cd.penalty < bestCandidate.penalty) {
                bestCandidate = cd;
            }
        }
    }
    return bestCandidate;
}

void NormalizedCut::split(const TreeNode& tn) {
    const EdgeIx volA = volume(tn);
    const EdgeWeight cutA = cutSize(tn);

    NodeIx child = tn.nix;
    NodeIx lvlIdx = tn.level;

    while (lvlIdx + 1 < hierarchy->height()) {
        lvlIdx++;
        NodeIx parent = hierarchy->lvl(lvlIdx).parents[child];
        hierarchy->subtractVolume(lvlIdx, parent, volA);
        hierarchy->addCut(lvlIdx, parent, cutA);

        child = parent;
    }
}

NormalizedCut::NormalizedCut(ExpanderHierarchy* eh) : hierarchy{eh}, graph{eh->getGraph()}, labels(graph->numNodes, 0), roots() {}

Partition NormalizedCut::compute(NodeIx k) {
    roots.emplace(hierarchy->height() - 1, 0);

    for(NodeIx i = 1; i <= k; i++) {
        Candidate bestCandidate{TreeNode{0, 0}, 1000};
        NodeIx rootIx = 0;
        frac_t bestDelta = 100000;
        for(const TreeNode& root: roots) {
            EdgeIx rootVol = volume(root);
            EdgeIx rootCutSize = hierarchy->lvl(root.level).cutSizes[root.nix];

            Candidate cd = getCandidate(root, rootIx, rootCutSize, rootVol);
            frac_t rootBias = static_cast<frac_t>(rootCutSize) / rootVol;
            frac_t delta = cd.penalty - rootBias;

            if(delta < bestDelta) {
                bestDelta = delta;
                bestCandidate = cd;
            }

            rootIx++;
        }
        split(bestCandidate.tn);
        roots.insert(bestCandidate.tn);
    }

    Partition part{graph};
    std::vector<NodeIx> mod;
    // This modifies the underlying graph, which invalidates the sparsifier.
    part.splitByIndices<false, false>(0, labels, k, mod);
    return part;
}
