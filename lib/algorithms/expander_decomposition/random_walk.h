//
// Created by robin on 13.03.2025.
//

#ifndef RCUT_RANDOM_WALK_H
#define RCUT_RANDOM_WALK_H

#include "core/partition.h"
#include <random>

template<MutableGraph Graph>
class RandomWalk {

    std::vector<frac_t> m_distribution;
    std::vector<frac_t> m_old_distribution;

    const Graph& G;
    std::mt19937 rng;   // random initial values

public:
    explicit RandomWalk(const Graph& gr, unsigned seed) : m_distribution(gr.numNodes), m_old_distribution(gr.numNodes), G(gr), rng(seed) {
        init_distribution(gr);
    }

    void setData(const std::vector<frac_t>& data) {
        assert(data.size() == m_distribution.size());
        for(NodeIx nix = 0; nix < data.size(); nix++) {
            m_distribution[nix] = data[nix];
        }
    }

    void init_distribution(const Graph& gr) {
        std::normal_distribution<> normal_distribution;
        frac_t avg = 0.0;

        for (size_t i = 0; i < m_distribution.size(); i++) {
            // generate a normal distributed number
            auto deg = gr.weightedDegree(i);
            std::normal_distribution<>::param_type stdev(0.0, std::sqrt(static_cast<frac_t>(deg)));
            frac_t x = normal_distribution(rng, stdev);
            m_distribution[i] = x;

            avg += (x - avg) / (static_cast<frac_t >(i) + 1.0);
        }

        for (frac_t& i : m_distribution) {
            i -= avg;
        }
    }

    frac_t resetCluster(const Cluster<Graph>& cluster) {
        std::normal_distribution<frac_t> normal_distribution;
        frac_t avg = 0.0;
        int i = 1;

        for (const ClusterVertex<Graph>& cn: cluster) {
            auto x = normal_distribution(rng);
            m_distribution[cn.nix] = x;
            avg += (x - avg) / static_cast<frac_t >(i);
            i++;
        }

        frac_t min = std::numeric_limits<frac_t>::max();
        frac_t max = std::numeric_limits<frac_t>::lowest();

        for (const ClusterVertex<Graph>& cn: cluster) {
            m_distribution[cn.nix] -= avg;
            if(m_distribution[cn.nix] < min) min = m_distribution[cn.nix];
            if(m_distribution[cn.nix] > max) max = m_distribution[cn.nix];
        }

        return max - min;
    }

    // TODO: idea: use skiplist on active nodes ( not edges)

    void iterate(const Partition<Graph>& part, const std::vector<NodeIx>& active) {
        std::swap(m_old_distribution, m_distribution);

        for(NodeIx clusterId: active) {
            const Cluster<Graph>& cluster = part.getCluster(clusterId);
            for(const ClusterVertex<Graph>& cv: cluster) {
                m_distribution[cv.nix] = rw_stay * m_old_distribution[cv.nix];
            }
        }

        for(NodeIx clusterId: active) {
            const Cluster<Graph>& cluster = part.getCluster(clusterId);
            for(const ClusterVertex<Graph>& cv: cluster) {
                if (cv.internalDegree == 0) continue;
                frac_t val = (1.0 - rw_stay) * m_old_distribution[cv.nix] / static_cast<frac_t>(cv.internalDegree);

                for (EdgeIx eix: cluster.internalEdgeIndices(cv)) {
                    NodeIx nb = G.edges[eix];
                    auto wgt = G.edgeWeight(eix);

                    m_distribution[nb] += val * static_cast<frac_t>(wgt);
                }
            }
        }
    }

    [[nodiscard]]
    const std::vector<frac_t>& values() const {
        return m_distribution;
    }

    frac_t potential(const Cluster<Graph>& cluster) {
        frac_t min = std::numeric_limits<frac_t>::max();
        frac_t max = std::numeric_limits<frac_t>::lowest();

        for (const ClusterVertex<Graph>& cn: cluster) {
            min = std::min(min, m_distribution[cn.nix]);
            max = std::max(max, m_distribution[cn.nix]);
        }

        return max - min;
    }

    frac_t L2potential(const Cluster<Graph>& cluster) {
        frac_t pot = 0.0;

        for (const ClusterVertex<Graph>& cv: cluster) {
            pot += m_distribution[cv.nix] * m_distribution[cv.nix];
        }

        return std::sqrt(pot);
    }

    void recenterClusterF(const Cluster<Graph>& cluster) {
        frac_t avg = 0.0;
        int i = 1;

        for (const ClusterVertex<Graph>& cn: cluster) {
            auto x = m_distribution[cn.nix];
            avg += (x - avg) / static_cast<frac_t >(i);
            i++;
        }

        for (const ClusterVertex<Graph>& cn: cluster) {
            m_distribution[cn.nix] -= avg;
        }
    }

    frac_t recenterCluster(const Cluster<Graph>& cluster) {
        frac_t avg = 0.0;
        int i = 1;

        for (const ClusterVertex<Graph>& cn: cluster) {
            auto x = m_distribution[cn.nix];
            avg += (x - avg) / static_cast<frac_t >(i);
            i++;
        }

        frac_t min = std::numeric_limits<frac_t>::max();
        frac_t max = std::numeric_limits<frac_t>::lowest();

        for (const ClusterVertex<Graph>& cn: cluster) {
            m_distribution[cn.nix] -= avg;
            if(m_distribution[cn.nix] < min) min = m_distribution[cn.nix];
            if(m_distribution[cn.nix] > max) max = m_distribution[cn.nix];
        }

        return max - min;
    }


    frac_t recenterClusterL2(const Cluster<Graph>& cluster) {
        frac_t avg = 0.0;
        int i = 1;

        for (const ClusterVertex<Graph>& cn: cluster) {
            auto x = m_distribution[cn.nix];
            avg += (x - avg) / static_cast<frac_t >(i);
            i++;
        }

        frac_t pot = 0.0;

        for (const ClusterVertex<Graph>& cv: cluster) {
            m_distribution[cv.nix] -= avg;
            pot += m_distribution[cv.nix] * m_distribution[cv.nix];
        }

        return std::sqrt(pot);
    }

    frac_t averageOf(const Cluster<Graph>& cluster) const {
        frac_t avg = 0.0;
        int i = 1;

        for(const ClusterVertex<Graph>& cv: cluster) {
            auto x = m_distribution[cv.nix];
            avg += (x - avg) / static_cast<frac_t >(i);
            i++;
        }

        return avg;
    }
};

#endif //RCUT_RANDOM_WALK_H
