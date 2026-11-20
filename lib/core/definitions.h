//
// Created by robin on 13.03.2025.
//

#ifndef RCUT_DEFINITIONS_H
#define RCUT_DEFINITIONS_H

#include <cassert>
#include <cstdint>
#include <bit>
#include <cmath>

// Diagnostics go to stderr, so that stdout only carries results
#ifdef DEBUG
#include <iostream>
#define INFO(msg) (std::cerr << "[INFO] " << msg << std::endl)
#define LOG(msg) (std::cerr << "[LOG] " << msg << std::endl)
#define WARN(msg) (std::cerr << "[WARNING] " << msg << "!" << std::endl)
#else
#include <iostream>
#define INFO(msg) ((void)0)
#define LOG(msg) ((void)0)
#define WARN(msg) (std::cerr << "[WARNING] " << msg << "!" << std::endl)
#endif

using NodeIx        = uint32_t;
using EdgeIx        = uint32_t;
using frac_t        = double;
using IntWeight     = int64_t;

constexpr double rw_stay = 0.1;   // the thresholds and the random seed are in Config (expander_decomposition/config.h)

constexpr NodeIx INVALID_EDGE = static_cast<NodeIx>(-1);

#endif //RCUT_DEFINITIONS_H
