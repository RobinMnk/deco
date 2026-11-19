//
// Created by robin on 19.01.2026.
//

#ifndef PAREX_DEVGRAPH_H
#define PAREX_DEVGRAPH_H

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/fill.h>
#include <thrust/scatter.h>
#include <thrust/scan.h>

#include "core/graph.h"
#include "types.h"

//struct DevGraph {
//    const NodeIx numNodes;
//    const EdgeIx numEdges;
//
//    // Graph
//    const NodeIx* neighbors;          // size: 2 * numEdges
//    const NodeIx* ranges;             // size: numNodes+1
//    const EdgeIx* active_degrees;     // size: numNodes
//    const NodeIx* labels;
//
//    // Buffers for Updates
//    const EdgeIx* edgeDeletionBuffer;
//    const NodeUpdate* nodeUpdateBuffer;
//
//    __host__ __device__
//    inline void deactivateEdge(EdgeIx idx) const;
//
//    __host__ __device__
//    inline void handleActiveDegrees(NodeIx idx) const;
//};

//__global__
//void deactivateEdgeKernel(DevGraph gr, EdgeIx numEdgeDeletions) {
//    NodeIx idx = blockIdx.x * blockDim.x + threadIdx.x;
//    if(idx < numEdgeDeletions) gr.deactivateEdge(idx);
//}
//
//__global__
//void degreeKernel(DevGraph gr, NodeIx numUpdates) {
//    NodeIx idx = blockIdx.x * blockDim.x + threadIdx.x;
//    if (idx < numUpdates) gr.handleActiveDegrees(idx);
//}


//__global__
//void updateActiveNodesKernel(
//        NodeIx numNodes,
//        const NodeIx* __restrict__ ranges,
//        const NodeIx* __restrict__ neighbors,
//        const NodeIx* __restrict__ labels,
//        EdgeIx* __restrict__ activeDegrees
//) {
//    NodeIx i = blockIdx.x * blockDim.x + threadIdx.x;
//    if (i >= numNodes) return;
//
//    // early exit for inactive nodes
//    const NodeIx clusterId = __ldg(&labels[i]);
//    if(clusterId == numNodes) {
//        activeDegrees[i] = 0;
//        return;
//    }
//
//    const NodeIx start = __ldg(&ranges[i]);
//    const NodeIx end   = __ldg(&ranges[i+1]);
//
//    EdgeIx degree = 0;
//
//    // The #pragma unroll hint tells the compiler to optimize the loop
//    // for small segments, which are common in many graphs.
//#pragma unroll 4
//    for (NodeIx j = start; j < end; ++j) {
//        const NodeIx neighbor = neighbors[j];
//        degree += (__ldg(&ranges[neighbor]) != clusterId);
//    }
//
//    activeDegrees[i] = degree;
//}


__global__
void scatterActiveDegreeCountsKernel(
    const NodeIx numActiveNodes,
    const LabeledNode* __restrict__ activeNodes,
    const EdgeIx* __restrict__ internalDegrees,
    EdgeIx* __restrict__ newRanges
) {
    const NodeIx i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numActiveNodes) return;

    const LabeledNode node = activeNodes[i];
    if (node.clusterId < 0) return;

    newRanges[node.nix + 1] = internalDegrees[node.nix];
}

__global__
void compactActiveEdgesKernel(
    const NodeIx numActiveNodes,
    const LabeledNode* __restrict__ activeNodes,
    const NodeIx* __restrict__ oldNeighbors,
    const EdgeIx* __restrict__ oldRanges,
    const EdgeIx* __restrict__ oldDegrees,
    const EdgeIx* __restrict__ newRanges,
    NodeIx* __restrict__ newNeighbors
) {
    const NodeIx i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numActiveNodes) return;

    const LabeledNode node = activeNodes[i];
    if (node.clusterId < 0) return;

    EdgeIx out = newRanges[node.nix];
    const EdgeIx begin = oldRanges[node.nix];
    const EdgeIx end = begin + oldDegrees[node.nix];

    for (EdgeIx eix = begin; eix < end; ++eix) {
        const NodeIx nb = oldNeighbors[eix];
        if (nb != INVALID_EDGE) {
            newNeighbors[out++] = nb;
        }
    }
}

__device__ __forceinline__
DevEdgeWeight edgeWeight(const DevEdgeWeight* __restrict__ weights, const EdgeIx eix) {
    return weights ? __ldg(&weights[eix]) : 1u;
}


class GraphManager {
    // Graph
    thrust::device_vector<NodeIx> neighbors;
    thrust::device_vector<EdgeIx> ranges;
    thrust::device_vector<EdgeIx> degrees;
    thrust::device_vector<DevEdgeWeight> weights;    // one per directed edge (same layout as neighbors), empty for unweighted graphs

    thrust::device_vector<NodeIx> nodeLookup;

    // Update buffers
//    thrust::device_vector<EdgeIx> edgeDeletionBuffer;
//    thrust::device_vector<NodeUpdate> nodeUpdateBuffer;
//    void updateVolumes();

    // void populateNodeLookup(NodeIx numNodes, EdgeIx totalEdges) {
    //     thrust::upper_bound(
    //         thrust::device,
    //         ranges.begin(),
    //         ranges.begin() + numNodes + 1,
    //         thrust::make_counting_iterator(static_cast<EdgeIx>(0)),
    //         thrust::make_counting_iterator(totalEdges),
    //         nodeLookup.begin()
    //     );
    //
    //     thrust::transform(
    //         thrust::device,
    //         nodeLookup.begin(),
    //         nodeLookup.end(),
    //         nodeLookup.begin(),
    //         thrust::placeholders::_1 - 1
    //     );
    // }

public:
    NodeIx n{0};
    EdgeIx m{0};

    NodeIx numClusters{1};

    template<CudaGraph GraphT>
    explicit GraphManager(const GraphT& graph) :
        neighbors(graph.edges),
        ranges(graph.ranges),
        degrees(graph.numNodes),
        weights(graph.edgeWeights().data(), graph.edgeWeights().data() + graph.edgeWeights().size()),  // empty for unweighted graphs
        // nodeLookup(2*graph.numEdges),
        n(graph.numNodes),
        m(graph.numEdges)
    //        edgeDeletionBuffer(2 * graph.numEdges),
    //        nodeUpdateBuffer(graph.numNodes)
    {
        thrust::transform(ranges.begin() + 1, ranges.end(), ranges.begin(), degrees.begin(), thrust::minus<EdgeIx>());

        // populateNodeLookup(graph.numNodes, 2 * graph.numEdges);
        // std::cout << "Copied Graph to GPU. \t" << neighbors.size() / 2 << " edges copied" << std::endl;
    }

    //    [[nodiscard]] DevGraph getView() const {
//        return DevGraph{
//            n, m,
//            thrust::raw_pointer_cast(neighbors.data()),
//            thrust::raw_pointer_cast(ranges.data()),
//        };
//    }

//    thrust::device_vector<NodeIx>& getLabels() {
//        return labels;
//    }

    const thrust::device_vector<EdgeIx>& getRanges() const {
        return ranges;
    }

    thrust::device_vector<NodeIx>& getNeighbors() {
        return neighbors;
    }

    thrust::device_vector<EdgeIx>& getDegrees() {
        return degrees;
    }

    // nullptr for unweighted graphs
    [[nodiscard]] const DevEdgeWeight* weightsPtr() const {
        return weights.empty() ? nullptr : thrust::raw_pointer_cast(weights.data());
    }

    [[nodiscard]] std::vector<DevEdgeWeight> downloadWeights() const {
        std::vector<DevEdgeWeight> w(weights.size());
        thrust::copy(weights.begin(), weights.end(), w.begin());
        return w;
    }

    [[nodiscard]] EdgeIx storedDirectedEdges() const {
        return static_cast<EdgeIx>(neighbors.size());
    }

    void compactActiveEdges(
        const thrust::device_vector<LabeledNode>& activeNodes,
        NodeIx numActiveNodes,
        const thrust::device_vector<EdgeIx>& internalDegrees
    ) {
        thrust::device_vector<EdgeIx> newRanges(n + 1, 0);

        const size_t blocks = getGridSize(numActiveNodes);
        scatterActiveDegreeCountsKernel<<<blocks, threads>>>(
            numActiveNodes,
            thrust::raw_pointer_cast(activeNodes.data()),
            thrust::raw_pointer_cast(internalDegrees.data()),
            thrust::raw_pointer_cast(newRanges.data())
        );

        thrust::inclusive_scan(
            thrust::device,
            newRanges.begin(),
            newRanges.end(),
            newRanges.begin()
        );

        thrust::device_vector<NodeIx> newNeighbors(neighbors.size(), INVALID_EDGE);
        compactActiveEdgesKernel<<<blocks, threads>>>(
            numActiveNodes,
            thrust::raw_pointer_cast(activeNodes.data()),
            thrust::raw_pointer_cast(neighbors.data()),
            thrust::raw_pointer_cast(ranges.data()),
            thrust::raw_pointer_cast(degrees.data()),
            thrust::raw_pointer_cast(newRanges.data()),
            thrust::raw_pointer_cast(newNeighbors.data())
        );

        neighbors.swap(newNeighbors);
        ranges.swap(newRanges);
        thrust::transform(ranges.begin() + 1, ranges.end(), ranges.begin(), degrees.begin(), thrust::minus<EdgeIx>());
    }

    // const thrust::device_vector<NodeIx> & getNodeLookup() const {
    //     return nodeLookup;
    // }

//    thrust::device_vector<EdgeIx>& getActiveDegrees() {
//        return active_degrees;
//    }
//
//    thrust::device_vector<EdgeIx>& getVolumes() {
//        return volumes;
//    }

//    std::vector<EdgeIx> downloadDegrees() {
//        std::vector<EdgeIx> degrees(n);
//        thrust::copy(active_degrees.begin(), active_degrees.end(), degrees.begin());
//        return degrees;
//    }

    CsrGraph<> downloadGraph() {
        CsrGraph g(n, m);
        thrust::copy(neighbors.begin(), neighbors.end(), g.edges.begin());
        thrust::copy(ranges.begin(), ranges.end(), g.ranges.begin());
        // thrust::copy(edgeMap.begin(), edgeMap.end(), g.map.begin());
        return g;
    }

    void updateLabels(std::vector<NodeIx>& nodeLabels, NodeIx activeClusters) {
//        labels = nodeLabels;
//        numActiveClusters = activeClusters;
//
//        unsigned int blocksPerGrid = (n + threads - 1) / threads;
//
//        updateActiveNodesKernel<<<blocksPerGrid, threads, 0, nullptr>>>(
//            n,
//            thrust::raw_pointer_cast(ranges.data()),
//            thrust::raw_pointer_cast(neighbors.data()),
//            thrust::raw_pointer_cast(labels.data()),
//            thrust::raw_pointer_cast(active_degrees.data())
//        );
//
//        updateVolumes();
    }
};

//void GraphManager::updateVolumes() {
//    // Output vector
//    volumes.resize(numActiveClusters);
//
//    // --- Make copies for sorting ---
//    thrust::device_vector<NodeIx> labels_sorted = labels;
//    thrust::device_vector<float> degrees_sorted = active_degrees;
//
//    // --- Radix sort by labels ---
//    thrust::sort_by_key(
//            labels_sorted.begin(),
//            labels_sorted.end(),
//            degrees_sorted.begin()
//    );
//
//    // --- Reduce by key ---
//    thrust::device_vector<NodeIx> unique_labels(numActiveClusters);
//    thrust::device_vector<float> summed_degrees(numActiveClusters);
//
//    auto end = thrust::reduce_by_key(
//            labels_sorted.begin(),
//            labels_sorted.end(),
//            degrees_sorted.begin(),
//            unique_labels.begin(),
//            summed_degrees.begin()
//    );
//
//    const size_t num_unique = end.first - unique_labels.begin();
//
//    // --- Scatter results into volumes ---
//    thrust::scatter(
//            summed_degrees.begin(),
//            summed_degrees.begin() + num_unique,
//            unique_labels.begin(),
//            volumes.begin()
//    );
//}





#endif //PAREX_DEVGRAPH_H
