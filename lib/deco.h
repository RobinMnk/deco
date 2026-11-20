//
// Public interface of Deco: expander decomposition, expander hierarchy and normalized cut, on the CPU or with CUDA.
//
// All functions take the graph by reference and may reorder its adjacency lists (the graph itself does not change).
// The CUDA backend is available to programs that link DecoExpanderDecompositionCuda (built with DECO_ENABLE_CUDA=ON),
// which defines DECO_ENABLE_CUDA for them.
//

#ifndef DECO_H
#define DECO_H

#include "algorithms/applications/normalized_cut.h"
#include "algorithms/expander_decomposition/cpu.h"
#include "algorithms/expander_decomposition/cuda/types.h"
#include "algorithms/tree_cut_sparsifier/expander_hierarchy.h"
#ifdef DECO_ENABLE_CUDA
#include "algorithms/expander_decomposition/cuda.h"
#endif

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace deco {

enum class Backend { Auto, Cpu, Cuda };

struct Options {
    Backend backend = Backend::Auto;   // Auto: CUDA if a device is available and supports the graph, otherwise the CPU
    Config config{};
};

// A partition of the nodes into clusters
template<class EdgeWeight>
struct Clustering {
    std::vector<NodeIx> clusterOf;   // cluster of every node, numbered 0 .. numClusters - 1
    NodeIx numClusters = 0;
    EdgeIx cutEdges = 0;             // edges between different clusters
    EdgeWeight cutWeight{};          // their total weight
};

template<class EdgeWeight>
struct Decomposition : Clustering<EdgeWeight> {
    Backend backend = Backend::Cpu;  // where the decomposition ran
    DecompositionStats stats;
};


namespace detail {

inline const char* cudaUnavailableReason() {
#ifdef DECO_ENABLE_CUDA
    return numCudaDevices() > 0 ? nullptr : "no CUDA device found";
#else
    return "Deco was built without CUDA";
#endif
}

template<class EdgeWeight, MutableGraph GraphT>
void assign(Clustering<EdgeWeight>& result, const Partition<GraphT>& partition) {
    result.clusterOf.assign(partition.getGraph().numNodes, 0);
    NodeIx c = 0;
    for (const auto& cluster : partition) {
        if (cluster.size() == 0) continue;
        for (const auto& cv : cluster) result.clusterOf[cv.nix] = c;
        ++c;
    }
    result.numClusters = c;
    result.cutEdges = partition.getNumCutEdges();
    result.cutWeight = partition.getCutSize();
}

#ifdef DECO_ENABLE_CUDA
// Copy with the 32-bit weights of the CUDA backend. The weights must be in range (see cudaLimitation).
inline CsrGraph<DevEdgeWeight> toDeviceWeights(const CsrGraph<IntWeight>& graph) {
    std::vector<NodeIx> edges = graph.edges;
    std::vector<EdgeIx> ranges = graph.ranges;
    std::vector<DevEdgeWeight> weights(graph.edgeWeights().size());
    std::ranges::transform(graph.edgeWeights(), weights.begin(), [](IntWeight w) { return static_cast<DevEdgeWeight>(w); });
    return CsrGraph<DevEdgeWeight>(graph.numNodes, graph.numEdges, std::move(edges), std::move(ranges), std::move(weights));
}
#endif

// Expander decomposition on the GPU, for a graph that chooseBackend accepted for CUDA
template<MutableGraph GraphT>
Partition<GraphT> cudaPartition([[maybe_unused]] GraphT& graph, [[maybe_unused]] const Config& config,
                                [[maybe_unused]] DecompositionStats* stats) {
#ifdef DECO_ENABLE_CUDA
    if constexpr (CudaGraph<GraphT>) {
        return cudaExpanderDecomposition(graph, config, stats);
    } else if constexpr (std::same_as<GraphT, CsrGraph<IntWeight>>) {
        auto copy = toDeviceWeights(graph);
        Clustering<DevEdgeWeight> onCopy;
        assign(onCopy, cudaExpanderDecomposition(copy, config, stats));
        return partitionByLabels(graph, onCopy.clusterOf, onCopy.numClusters);
    }
#endif
    throw std::logic_error("deco: the CUDA backend cannot decompose this graph type");
}

} // namespace detail


// Whether the CUDA backend can run: Deco was built with CUDA and a CUDA device is present
inline bool cudaAvailable() {
    return detail::cudaUnavailableReason() == nullptr;
}

/**
 * Why the CUDA backend cannot decompose `graph`, or nullptr if it can. It takes unweighted graphs and integer weights
 * from 1 to 2^32 - 1, as long as the volume (twice the total weight) fits 32 bits. 64-bit integer weights are copied
 * to 32 bits for the GPU. Contraction never increases the total weight, so this also holds for every level of an
 * expander hierarchy of `graph`.
 */
template<MutableGraph GraphT>
const char* cudaLimitation(const GraphT& graph) {
    constexpr uint64_t maxVolume = std::numeric_limits<EdgeIx>::max();
    if constexpr (std::same_as<GraphT, CsrGraph<>>) {
        if (2 * uint64_t{graph.numEdges} > maxVolume) return "the volume (twice the number of edges) does not fit 32 bits";
        return nullptr;
    } else if constexpr (CudaGraph<GraphT> || std::same_as<GraphT, CsrGraph<IntWeight>>) {
        uint64_t volume = 0;
        for (const auto w : graph.edgeWeights()) {   // every edge appears twice, so this sums up to the volume
            if (std::cmp_less(w, 1) || std::cmp_greater(w, maxVolume)) return "edge weights must be integers from 1 to 2^32 - 1";
            volume += w;
            if (volume > maxVolume) return "the volume (twice the total edge weight) does not fit 32 bits";
        }
        return nullptr;
    } else {
        return "only unweighted graphs and integer edge weights are supported";
    }
}

/**
 * The backend that runs for `graph` when `requested` is asked for. Auto falls back to the CPU if CUDA is not available
 * or cannot take the graph; asking for Cuda explicitly throws std::invalid_argument with the reason instead.
 */
template<MutableGraph GraphT>
Backend chooseBackend(const GraphT& graph, Backend requested) {
    if (requested == Backend::Cpu) return Backend::Cpu;

    const char* reason = detail::cudaUnavailableReason();
    if (!reason) reason = cudaLimitation(graph);

    if (requested == Backend::Auto) return reason ? Backend::Cpu : Backend::Cuda;
    if (reason) throw std::invalid_argument(std::string("deco: cannot use the CUDA backend: ") + reason);
    return Backend::Cuda;
}

// Expander decomposition: clusters that are expanders (no sweep cut below config.targetSparsity), few edges between
template<MutableGraph GraphT>
Decomposition<typename GraphT::EdgeWeight> expanderDecomposition(GraphT& graph, const Options& options = {}) {
    Decomposition<typename GraphT::EdgeWeight> result;
    result.backend = chooseBackend(graph, options.backend);
    if (result.backend == Backend::Cuda) {
        detail::assign(result, detail::cudaPartition(graph, options.config, &result.stats));
    } else {
        detail::assign(result, expander_decomposition::compute(graph, options.config, &result.stats));
    }
    return result;
}

// Expander hierarchy (tree cut sparsifier): repeated expander decompositions of the contracted graph
template<MutableGraph GraphT>
ExpanderHierarchy<typename GraphT::EdgeWeight> expanderHierarchy(GraphT& graph, const Options& options = {}) {
    if (chooseBackend(graph, options.backend) == Backend::Cpu) {
        return buildExpanderHierarchy(graph, options.config);
    }
    return buildExpanderHierarchy(
        graph,
        [](auto& level, const Config& config) { return detail::cudaPartition(level, config, nullptr); },
        options.config
    );
}

// Partition into k clusters with a small normalized cut, derived from the expander hierarchy of `graph`
template<MutableGraph GraphT>
Clustering<typename GraphT::EdgeWeight> normalizedCut(GraphT& graph, NodeIx k, const Options& options = {}) {
    auto hierarchy = expanderHierarchy(graph, options);
    NormalizedCut ncut{hierarchy};

    Clustering<typename GraphT::EdgeWeight> result;
    detail::assign(result, ncut.compute(graph, k));
    return result;
}

} // namespace deco

#endif //DECO_H
