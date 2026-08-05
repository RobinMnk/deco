//
// Created by robin on 26.01.2026.
//

#ifndef PAREX_TYPES_H
#define PAREX_TYPES_H

#include <thrust/device_vector.h>

#include "algorithms/expander_decomposition/cuda/types.h"

constexpr int threads = 256;
constexpr int WARP = 32;
constexpr int warpsPerBlock = threads / WARP;
static_assert(WARP > 0 && WARP <= 32 && (WARP & (WARP - 1)) == 0, "WARP must be a power of two in [1, 32]");
static_assert(threads % WARP == 0, "threads must be divisible by WARP");

constexpr int WARPMASK = WARP - 1;
constexpr unsigned int BASE_SUBWARP_MASK = WARP == 32 ? 0xFFFFFFFFU : static_cast<unsigned int>((1ULL << WARP) - 1ULL);

inline unsigned int getGridSize(unsigned int n) {
    return (n + threads - 1) / threads;
}

inline unsigned int getGridSizeWarpParallel(unsigned int n) {
    return (n + warpsPerBlock - 1) / warpsPerBlock;
}

template<typename T>
using dVec = thrust::device_vector<T>;

template<typename T>
void inspect(dVec<T> vec, NodeIx n) {
    std::vector<T> host_vec(n);
    thrust::copy(vec.begin(), vec.begin() + n, host_vec.begin());
    for (const T& t : host_vec) {
        std::cout << t << "  ";
    }
    std::cout << std::endl;
}

#define PRINT_DVEC(name, v) std::cout << name << ": "; inspect(v, v.size());

#endif //PAREX_TYPES_H
