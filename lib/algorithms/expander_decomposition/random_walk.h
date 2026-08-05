//
// Created by robin on 13.03.2025.
//

#ifndef RCUT_RANDOM_WALK_H
#define RCUT_RANDOM_WALK_H

#include "core/partition.h"
#include <valarray>
#include <random>

// const frac_t rw_stay = 0.1;
extern std::mt19937 random_source;

class RandomWalk {
    std::valarray<frac_t> m_distribution;
    std::valarray<frac_t> m_old_distribution;
//    std::vector<frac_t> val_prec;

public:
    explicit RandomWalk (size_t n);

    void setData(std::vector<frac_t>& data);

    void init_distribution();

    frac_t resetCluster(const Cluster& cluster);

    void iterate(const Partition& part, const std::vector<NodeIx>& active);

    const std::valarray<frac_t>& values();

    frac_t potential(const Cluster& cluster);

    void recenterClusterF(const Cluster& cluster);

    frac_t recenterCluster(const Cluster& cluster);

    frac_t averageOf(const Cluster& cluster);
};

#endif //RCUT_RANDOM_WALK_H
