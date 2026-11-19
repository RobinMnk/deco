//
// Created by robin on 05.08.2026.
//

#ifndef RCUT_EXPANDER_DECOMPOSITION_CUDA_TYPES_H
#define RCUT_EXPANDER_DECOMPOSITION_CUDA_TYPES_H

#include "core/definitions.h"
#include "core/graph.h"

#include <concepts>
#include <cstdint>
#include <vector>

// using Graph = CsrGraph<>;

using DevEdgeWeight = uint32_t;

// Graph types the CUDA implementation accepts
template<class G>
concept CudaGraph =
    std::same_as<G, CsrGraph<>> ||              // unweighted
    std::same_as<G, CsrGraph<DevEdgeWeight>>;   // weighted, e.g. contracted levels of an unweighted graph

using label_t = int32_t;

struct SweepCutData {
    label_t clusterId;
    float sparsity;
    NodeIx offset;
};

struct LabeledNode {
    NodeIx nix;
    label_t clusterId;
};

struct NodeData {
    NodeIx nix;
    EdgeIx prefixVolume;
    int32_t prefixEdgeDiff;
    NodeIx offsetInCluster;
};

struct ClusterData {
    float rwSum;
    float maxPotential;
    float minPotential;
    NodeIx totalElements;
};

struct AllSweepCuts {
    std::vector<label_t> clusterIds;
    std::vector<SweepCutData> cuts;
};

struct FinalPartition {
    std::vector<label_t> clusterIds;
    int numClusters;
};

#endif //RCUT_EXPANDER_DECOMPOSITION_CUDA_TYPES_H
