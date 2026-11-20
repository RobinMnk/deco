//
// Created by robin on 16.01.2026.
//

#ifndef PAREX_INTERFACE_H
#define PAREX_INTERFACE_H

#include "algorithms/expander_decomposition/cuda/types.h"
#include "algorithms/expander_decomposition/config.h"
#include <memory>

struct NodeUpdate {
    NodeIx nix;
    EdgeIx diff;
};

class CudaDeviceManager {
public:
    CudaDeviceManager();
    ~CudaDeviceManager();

    CudaDeviceManager(const CudaDeviceManager&) = delete;
    CudaDeviceManager& operator=(const CudaDeviceManager&) = delete;

    // Precondition for weighted graphs: weights are positive and the total volume (2 * sum of weights) fits EdgeIx,
    // since all degrees, volumes and cuts are 32-bit on the GPU. Graphs contracted from an unweighted graph always
    // satisfy this, as contraction never increases the total weight.
    void initialize(const CsrGraph<>& graph, const Config& config = {});                // unweighted
    void initialize(const CsrGraph<DevEdgeWeight>& graph, const Config& config = {});   // weighted

    CsrGraph<> downloadGraph();                      // structure only
    std::vector<DevEdgeWeight> downloadWeights();    // one per directed edge, empty for unweighted graphs
    std::vector<EdgeIx> downloadDegrees();
    std::vector<NodeData> downloadPartition();

    void iterateRandomWalk();
    std::vector<frac_t> readRandomWalkValues();

    void computeSweepCuts();
//    void inspectSweepCut(std::vector<EdgeIx>& prefixSums, std::vector<EdgeIx>& cutVolumes);
    AllSweepCuts readSweepCuts();

    void cutClusters();

    EdgeIx getNumCutEdges();

    void expanderDecomposition();   // runs until every cluster is done
    size_t getNumIterations();      // of the last expanderDecomposition()

    FinalPartition getFinalPartition();

    std::vector<int> downloadActiveEdgeMap();
    std::vector<int> downloadLabels();
    int getNumClusters();



    void updateLabels(std::vector<NodeIx>& nodeLabels, NodeIx activeClusters);

//    void applyGraphUpdates(const std::vector<EdgeIx>& edgeDeletions, const std::vector<NodeUpdate>& updates);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};


int testCuda();

// Number of CUDA devices, 0 if there is none or no driver
int numCudaDevices();

#endif //PAREX_INTERFACE_H
