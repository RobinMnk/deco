//
// Created by robin on 05.08.2026.
//

#ifndef RCUT_EXPANDER_DECOMPOSITION_CUDA_H
#define RCUT_EXPANDER_DECOMPOSITION_CUDA_H

#include "config.h"
#include "core/partition.h"
#include "cuda/interface.h"

inline Partition cudaExpanderDecomposition(Graph& graph, const Config& = {}) {
    CudaDeviceManager cuda;
    cuda.initialize(graph);
    cuda.expanderDecomposition();

    FinalPartition finalPartition = cuda.getFinalPartition();
    Partition partition{&graph};
    std::vector<NodeIx> modified;
    partition.splitByIndices<false, false>(0, finalPartition.clusterIds, finalPartition.numClusters, modified);
    return partition;
}

#endif //RCUT_EXPANDER_DECOMPOSITION_CUDA_H
