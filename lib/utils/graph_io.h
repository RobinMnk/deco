//
// Created by robin on 13.03.2025.
//

#ifndef RCUT_IO_H
#define RCUT_IO_H

#include "core/graph.h"

#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <cstdint>
#include <cmath>
#include <variant>

using AnyGraph = std::variant<
    CsrGraph<>,
    CsrGraph<IntWeight>,
    CsrGraph<frac_t>
>;

enum class MtxValueType { Unweighted, Real, Integer };
enum class MtxSymmetry  { General, Symmetric, SkewSymmetric };

struct MtxHeader {
    uint32_t rows{0};
    uint32_t cols{0};
    uint64_t entries{0};
    MtxValueType value_type{MtxValueType::Unweighted};
    MtxSymmetry symmetry{MtxSymmetry::General};
};

inline MtxHeader parse_mtx_header(std::ifstream& file) {
    std::string line;
    MtxHeader header;

    // Read banner line: %%MatrixMarket matrix coordinate <field> <symmetry>
    if (!std::getline(file, line)) {
        throw std::runtime_error("Empty MTX file.");
    }

    std::stringstream banner_ss(line);
    std::string banner, object, format, field, symmetry;
    if (!(banner_ss >> banner >> object >> format >> field >> symmetry) || (banner != "%%MatrixMarket" && banner != "%MatrixMarket")) {
        throw std::runtime_error("Invalid Matrix Market header banner.");
    }

    if (object != "matrix" || format != "coordinate") {
        throw std::runtime_error("Only coordinate matrix market files are supported.");
    }

    // Determine weight field
    if (field == "pattern") {
        header.value_type = MtxValueType::Unweighted;
    } else if (field == "real" || field == "double") {
        header.value_type = MtxValueType::Real;
    } else if (field == "integer") {
        header.value_type = MtxValueType::Integer;
    } else {
        throw std::runtime_error("Unsupported field type: " + field);
    }

    // Determine symmetry
    if (symmetry == "general") {
        header.symmetry = MtxSymmetry::General;
    } else if (symmetry == "symmetric") {
        header.symmetry = MtxSymmetry::Symmetric;
    } else if (symmetry == "skew-symmetric") {
        header.symmetry = MtxSymmetry::SkewSymmetric;
    }

    // Skip comment lines starting with '%'
    while (std::getline(file, line)) {
        if (!line.empty() && line[0] != '%') break;
    }

    // Parse dimensions: num_rows num_cols num_entries
    std::stringstream dim_ss(line);
    if (!(dim_ss >> header.rows >> header.cols >> header.entries)) {
        throw std::runtime_error("Failed to parse matrix dimensions.");
    }

    return header;
}

template <typename Spec = Unweighted>
DynamicGraph<Spec> read_mtx_dynamic_impl(std::ifstream& file, const MtxHeader& header) {
    NodeIx n = std::max(header.cols, header.rows);
    EdgeIx m = header.entries;
    DynamicGraph<Spec> dg(n, m);

    using weight_type = typename DynamicGraph<Spec>::weight_type;
    using Policy = select_policy_t<Spec>;

    NodeIx u = 0, v = 0;

    while (file >> u >> v) {
        if (u == 0 || v == 0 || u > n || v > n) {
            throw std::runtime_error("Matrix Market node index out of range.");
        }

        weight_type w{1};
        if constexpr (is_weighted_policy<Policy>::value) {
            if (header.value_type != MtxValueType::Unweighted && !(file >> w)) {
                throw std::runtime_error("Failed to parse Matrix Market edge weight.");
            }
        }

        --u;
        --v;

        if (u == v) {
            continue;
        }

        if constexpr (is_weighted_policy<Policy>::value) {
            if constexpr (std::floating_point<weight_type>) {
                if (!std::isfinite(w)) {
                    throw std::runtime_error("Matrix Market edge weight must be finite.");
                }
            }

            if (w < weight_type{}) {
                throw std::runtime_error("Matrix Market edge weight must be nonnegative.");
            }

            if (w == weight_type{}) {
                continue;
            }

            dg.addEdge(u, v, w);
        } else {
            dg.addEdge(u, v);
        }
    }

    return dg;
}

inline CsrGraph<> load_mtx_unweighted(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filepath);
    }

    MtxHeader header = parse_mtx_header(file);
    if (header.value_type != MtxValueType::Unweighted) {
        throw std::runtime_error("MTX file is not unweighted");
    }

    DynamicGraph<> dg = read_mtx_dynamic_impl(file, header);
    return dg.finalize();
}



inline AnyGraph load_mtx_dynamic(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filepath);
    }

    MtxHeader header = parse_mtx_header(file);

    switch (header.value_type) {
        case MtxValueType::Unweighted: {
            DynamicGraph<> dg = read_mtx_dynamic_impl(file, header);
            return dg.finalize();
        }
        case MtxValueType::Real: {
            DynamicGraph<frac_t> dg = read_mtx_dynamic_impl<frac_t>(file, header);
            return dg.finalize();
        }
        case MtxValueType::Integer: {
            DynamicGraph<IntWeight> dg = read_mtx_dynamic_impl<IntWeight>(file, header);
            return dg.finalize();
        }
    }
    throw std::runtime_error("Unknown MTX weight format.");
}

#endif //RCUT_IO_H
