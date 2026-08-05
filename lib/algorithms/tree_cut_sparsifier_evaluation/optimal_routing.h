//
// Created by robin on 05.08.2026.
//

#ifndef DECO_OPTIMAL_ROUTING_H
#define DECO_OPTIMAL_ROUTING_H

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <tuple>
#include <numeric>

// AMGCL Headers
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/cg.hpp>

// Define the AMG-preconditioned Conjugate Gradient Solver
typedef amgcl::backend::builtin<double> Backend;
typedef amgcl::make_solver<
    amgcl::amg<
        Backend,
        amgcl::coarsening::smoothed_aggregation,
        amgcl::relaxation::spai0
    >,
    amgcl::solver::cg<Backend>
> LaplacianSolver;

struct RoutingResult {
    double min_peak_congestion;       // The minimal optimal congestion factor lambda*
    std::vector<double> average_flow; // Average edge flow vector (size = number of directed CSR edges)
};

/**
 * Solves one electrical flow step L(r) * phi = demand using current resistances r.
 */
static std::vector<double> solve_electrical_potentials(
    int n,
    const std::vector<int>& ptr,
    const std::vector<int>& col,
    const std::vector<double>& resistances,
    const std::vector<double>& demand
) {
    std::vector<int> lap_ptr(n + 1, 0);
    std::vector<int> lap_col;
    std::vector<double> lap_val;

    lap_col.reserve(col.size() + n);
    lap_val.reserve(col.size() + n);

    for (int i = 0; i < n; ++i) {
        lap_ptr[i] = static_cast<int>(lap_col.size());
        double diag_val = 0.0;
        int diag_idx_in_row = -1;

        for (int idx = ptr[i]; idx < ptr[i + 1]; ++idx) {
            int j = col[idx];
            double R_e = resistances[idx];
            double conductance = 1.0 / R_e;

            if (i == j) continue;

            diag_val += conductance;

            if (j > i && diag_idx_in_row == -1) {
                diag_idx_in_row = static_cast<int>(lap_col.size());
                lap_col.push_back(i);
                lap_val.push_back(0.0);
            }

            lap_col.push_back(j);
            lap_val.push_back(-conductance);
        }

        if (diag_idx_in_row == -1) {
            lap_col.push_back(i);
            lap_val.push_back(diag_val);
        } else {
            lap_val[diag_idx_in_row] = diag_val;
        }
    }
    lap_ptr[n] = static_cast<int>(lap_col.size());

    std::vector<double> phi(n, 0.0);
    LaplacianSolver solve(std::tie(n, lap_ptr, lap_col, lap_val));
    solve(demand, phi);

    return phi;
}

/**
 * Computes the minimal peak congestion (lambda*) for a splittable single-commodity flow.
 *
 * @param n           Number of vertices (|V|)
 * @param ptr         CSR row pointers array (size n + 1)
 * @param col         CSR column indices array (size |E| * 2)
 * @param capacities  Base edge capacities c(e) (size |E| * 2)
 * @param demand      Demand vector (+1 for source, -1 for sink, 0 otherwise)
 * @param iterations  MWU outer iterations (15-25 is typically optimal for ~5% error)
 *
 * @return Struct containing minimal peak congestion and averaged valid optimal flow vector
 */
RoutingResult compute_optimal_splittable_congestion(
    int n,
    const std::vector<int>& ptr,
    const std::vector<int>& col,
    const std::vector<double>& capacities,
    const std::vector<double>& demand,
    int iterations = 20
) {
    size_t num_csr_edges = col.size();

    // 1. Initial edge resistances R_e = 1 / c(e)
    std::vector<double> R(num_csr_edges);
    for (size_t idx = 0; idx < num_csr_edges; ++idx) {
        R[idx] = 1.0 / std::max(capacities[idx], 1e-9);
    }

    std::vector<double> current_flows(num_csr_edges, 0.0);
    std::vector<double> avg_flows(num_csr_edges, 0.0);

    // Learning rate parameter for MWU
    double eta = 0.5;

    for (int iter = 0; iter < iterations; ++iter) {
        // Step A: Solve electrical node potentials
        std::vector<double> phi = solve_electrical_potentials(n, ptr, col, R, demand);

        // Step B: Calculate edge current f_ij = (phi_i - phi_j) / R_ij
        double max_iter_congestion = 0.0;

        for (int i = 0; i < n; ++i) {
            for (int idx = ptr[i]; idx < ptr[i + 1]; ++idx) {
                int j = col[idx];
                double flow = (phi[i] - phi[j]) / R[idx];
                current_flows[idx] = flow;

                double utilization = std::abs(flow) / capacities[idx];
                if (utilization > max_iter_congestion) {
                    max_iter_congestion = utilization;
                }
            }
        }

        // Accumulate average flow across all iterations
        for (size_t idx = 0; idx < num_csr_edges; ++idx) {
            avg_flows[idx] += current_flows[idx] / iterations;
        }

        // Step C: Multiplicative Weight Update on Edge Resistances
        // Increase resistance exponentially on heavily utilized edges
        if (max_iter_congestion > 1e-9) {
            for (size_t idx = 0; idx < num_csr_edges; ++idx) {
                double utilization = std::abs(current_flows[idx]) / capacities[idx];
                double normalized_utilization = utilization / max_iter_congestion;

                // R_e <- R_e * exp(eta * (utilization / max_utilization))
                R[idx] *= std::exp(eta * normalized_utilization);
            }
        }
    }

    // Calculate final peak congestion of the averaged flow vector
    double final_peak_congestion = 0.0;
    for (size_t idx = 0; idx < num_csr_edges; ++idx) {
        double congestion = std::abs(avg_flows[idx]) / capacities[idx];
        if (congestion > final_peak_congestion) {
            final_peak_congestion = congestion;
        }
    }

    return {final_peak_congestion, avg_flows};
}

#endif //DECO_OPTIMAL_ROUTING_H