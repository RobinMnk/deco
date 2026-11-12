//
// Created by robin on 13.03.2025.
//

#ifndef RCUT_EXPANDER_DECOMPOSITION_CPU_H
#define RCUT_EXPANDER_DECOMPOSITION_CPU_H

#include "config.h"
#include "core/partition.h"
#include "random_walk.h"

#include <numeric>

namespace expander_decomposition {

    template<MutableGraph GraphT>
    bool isConverged(const Cluster<GraphT>& cluster, double potential, const Config& config) {
        return potential <= config.randomWalkThreshold; // 1 / std::sqrt(cluster.internalVolume);
    }

    template<MutableGraph GraphT>
    bool isSparse(const Cluster<GraphT>& cluster, double sparsity, const Config& config) {
        return sparsity < config.targetSparsity; // * std::sqrt(cluster.internalVolume);
    }


    template<Strategy strategy = {}, MutableGraph Graph>
    Partition<Graph> compute(Graph& graph, const Config& config = {}) {
        RandomWalk walk{graph};
        Partition decomposition{graph};

        std::vector<NodeIx> active, next(decomposition.numClusters());
        std::iota(next.begin(), next.end(), 0);

        // Graph consists of multiple connected components -> reset random walk in each
        if(decomposition.numClusters() > 1) {
            for(const Cluster<Graph>& cl: decomposition)
                walk.recenterCluster(cl);
        }

        size_t numSteps = config.initialNumSteps;
        size_t iterations = 0;

        while(!next.empty()) {
            std::swap(active, next);
            next.clear();

            iterations++;

            size_t rw = numSteps;
            while (rw--) walk.iterate(decomposition, active);
            auto& values = walk.values();

            // for (NodeIx cid: active) {
            //     const Cluster<Graph>& cluster = decomposition.getCluster(cid);
            //     printf("Cluster %d:\n  potential: %f\n", cid, walk.L2potential(cluster));
            // }

            while(!active.empty()) {
                NodeIx clusterId = active.back();
                active.pop_back();

                const Cluster<Graph>& current = decomposition.getCluster(clusterId);

                // walk.recenterCluster(current);

                if(std::abs(walk.averageOf(current)) > config.randomWalkThreshold) {
                    std::cout << "Cluster average: " << std::abs(walk.averageOf(current)) << std::endl;
                    WARN("\nCLUSTER CAN NEVER CONVERGE!!!\n");
                }

                LOG("  working on cluster " << clusterId << "\t (" << current.size() << " nodes, potential: " << walk.potential(current) << ")");

                if constexpr (!strategy.cutWhenConverged) {
                    // if(walk.potential(current) < config.randomWalkThreshold) {
                    if (isConverged<Graph>(current, walk.potential(current), config)) {
                        LOG("       -> potential sufficiently low -> certified expander");
                        continue;
                    }
                }

                auto sweepCut = decomposition.sweepCut(clusterId, values);

                LOG("     Sweep cut:   sparsity=" << sweepCut.sparsity << ", offset=" << sweepCut.offset);

                if(isSparse<Graph>(current, sweepCut.sparsity, config)) {
                // if(sweepCut.sparsity <= config.targetSparsity) {
                    LOG("       -> considered sparse -> cutting");
                    std::vector<NodeIx> modified{clusterId};
                    decomposition.template split<false, true>(clusterId, sweepCut.offset, modified);

                    numSteps = config.mainNumSteps;

                    for (NodeIx clx: modified) {
                        const Cluster<Graph> &cl = decomposition.getCluster(clx);
                        if (cl.size() < 2 || cl.internalVolume == 0) continue;
                        frac_t potential;
                        if constexpr (strategy.crs == Strategy::RECENTER)  potential = walk.recenterCluster(cl);
                        if constexpr (strategy.crs == Strategy::RESET)     potential = walk.resetCluster(cl);

                        if constexpr (strategy.cutWhenConverged) {
                            if constexpr (strategy.cs == Strategy::CASCADE) {
                                // cascade sweep cuts by adding cluster clx to active
                                active.push_back(clx);
                            } else if constexpr (strategy.cs == Strategy::ITERATE) {
                                // add clx directly to the next round, sweep cut will be considered after another step of the random walk
                                next.push_back(clx);
                            }
                        } else {
                            // if strategy.cutWhenConverged is set to false, we only consider cluster clx if its potential is above the threshold
                            if (potential > config.randomWalkThreshold) {
                                if constexpr (strategy.cs == Strategy::CASCADE) {
                                    active.push_back(clx);
                                } else if constexpr (strategy.cs == Strategy::ITERATE) {
                                    next.push_back(clx);
                                }
                            }
                        }
                    }
                } else {
                    if constexpr (strategy.cutWhenConverged) {
                        if(!isConverged<Graph>(current, walk.potential(current), config)) {
                        // if(walk.potential(current) > config.randomWalkThreshold) {
                            next.push_back(clusterId);
                        } else {
                            LOG("       -> potential sufficiently low -> certified expander");
                        }
                    } else {
                        next.push_back(clusterId);
                    }
                }
            }
        }

        printf("iterations: %llu\n", iterations);

        return decomposition;
    }
}

#endif //RCUT_EXPANDER_DECOMPOSITION_CPU_H
