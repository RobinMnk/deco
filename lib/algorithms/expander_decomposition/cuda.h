//
// Created by robin on 05.08.2026.
//

#ifndef RCUT_EXPANDER_DECOMPOSITION_CUDA_H
#define RCUT_EXPANDER_DECOMPOSITION_CUDA_H

#include "config.h"
#include "core/partition.h"
#include "cuda/interface.h"

// template<Strategy strategy = {}, MutableGraph Graph>
//     Partition<Graph> compute(Graph& graph,


inline Partition<Graph> cudaExpanderDecomposition(Graph& graph, const Config& = {}) {
    Timer t;
    t.start();
    CudaDeviceManager cuda;
    cuda.initialize(graph);
    cuda.expanderDecomposition();
    auto millis = t.timeMillis();

    FinalPartition finalPartition = cuda.getFinalPartition();
    Partition<Graph> partition{graph};
    std::vector<NodeIx> modified;
    partition.template splitByIndices<false, false>(0, finalPartition.clusterIds, finalPartition.numClusters, modified);

    printf("time: %lld\nnumClusters: %d\ncutEdges: %d\nnodes: %d\nedges: %d\n", millis, partition.numClusters(), partition.getNumCutEdges(), graph.numNodes, graph.numEdges);


    return partition;
}

#endif //RCUT_EXPANDER_DECOMPOSITION_CUDA_H
