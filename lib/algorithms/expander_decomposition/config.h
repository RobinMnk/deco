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

// Configuration for Expander Decomposition
struct Config {
    frac_t targetSparsity = 0.01;
    frac_t randomWalkThreshold = 1e-6;
    size_t initialNumSteps = 6; // initial number of rw steps, before the first sweep cut is made
    size_t mainNumSteps = 2;
};

#endif //RCUT_EXPANDER_DECOMPOSITION_CONFIG_H
