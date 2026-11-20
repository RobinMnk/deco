//
// Created by robin on 05.08.2026.
//

#ifndef RCUT_EXPANDER_DECOMPOSITION_CUDA_H
#define RCUT_EXPANDER_DECOMPOSITION_CUDA_H

#include "config.h"
#include "core/partition.h"
#include "cuda/interface.h"
#include "utils/timer.h"


template<CudaGraph GraphT>
Partition<GraphT> cudaExpanderDecomposition(GraphT& graph, const Config& config = {}, DecompositionStats* stats = nullptr) {
    Timer t;
    t.start();
    CudaDeviceManager cuda;
    cuda.initialize(graph, config);
    cuda.expanderDecomposition();
    if (stats) {
        stats->seconds = t.timeSeconds();   // upload and decomposition on the GPU, without building the host partition
        stats->iterations = cuda.getNumIterations();
    }

    FinalPartition finalPartition = cuda.getFinalPartition();
    Partition<GraphT> partition{graph};
    std::vector<NodeIx> modified;
    // the GPU labels are not grouped by node order -> the cluster must be sorted by label (requiresSort = true)
    partition.template splitByIndices<true, false>(0, finalPartition.clusterIds, finalPartition.numClusters, modified);

    return partition;
}

#endif //RCUT_EXPANDER_DECOMPOSITION_CUDA_H
