//
// Created by robin on 13.03.2025.
//

#ifndef RCUT_EXPANDER_DECOMPOSITION_CONFIG_H
#define RCUT_EXPANDER_DECOMPOSITION_CONFIG_H

#include "core/definitions.h"

struct Strategy {
    enum ClusterResetStrategy   { RECENTER, RESET };
    enum CutStrategy            { CASCADE, ITERATE };

    ClusterResetStrategy    crs{RECENTER};
    CutStrategy             cs{ITERATE};
    bool                    cutWhenConverged{false};
};

// Configuration for Expander Decomposition, used by the CPU and the CUDA implementation
struct Config {
    frac_t targetSparsity = 0.3;        // a cluster is cut if its sweep cut has sparsity (cut / smaller volume) below this
    frac_t randomWalkThreshold = 1e-4;  // a cluster is done once its walk values differ by less than this
    unsigned seed = 11;                 // seed for the initial random-walk values
    size_t initialNumSteps = 1;         // CPU only: random-walk steps before the first sweep cut
    size_t mainNumSteps = 1;            // CPU only: random-walk steps between later sweep cuts (CUDA always does 1)
};

// Statistics of an expander decomposition run
struct DecompositionStats {
    size_t iterations = 0;   // rounds of random-walk steps and sweep cuts
    double seconds = 0;
};

#endif //RCUT_EXPANDER_DECOMPOSITION_CONFIG_H
