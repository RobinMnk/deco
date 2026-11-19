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
Partition<GraphT> cudaExpanderDecomposition(GraphT& graph, const Config& = {}) {
    Timer t;
    t.start();
    CudaDeviceManager cuda;
    cuda.initialize(graph);
    cuda.expanderDecomposition();
    auto millis = t.timeMillis();

    FinalPartition finalPartition = cuda.getFinalPartition();
    Partition<GraphT> partition{graph};
    std::vector<NodeIx> modified;
    // the GPU labels are not grouped by node order -> the cluster must be sorted by label (requiresSort = true)
    partition.template splitByIndices<true, false>(0, finalPartition.clusterIds, finalPartition.numClusters, modified);

    printf("time: %lld\nnumClusters: %d\ncutEdges: %d\ncutWeight: %llu\nnodes: %d\nedges: %d\n", millis, partition.numClusters(), partition.getNumCutEdges(), static_cast<unsigned long long>(partition.getCutSize()), graph.numNodes, graph.numEdges);


    return partition;
}

#endif //RCUT_EXPANDER_DECOMPOSITION_CUDA_H
