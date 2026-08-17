//
// Created by robin on 13.03.2025.
//

#ifndef RCUT_GRAPH_H
#define RCUT_GRAPH_H

#include <algorithm>
#include <concepts>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include "definitions.h"

#if defined(_MSC_VER) && !defined(__clang__)
    #define NO_UNIQUE_ADDRESS [[msvc::no_unique_address]]
#else
    #define NO_UNIQUE_ADDRESS [[no_unique_address]]
#endif


struct Unweighted {
    using weight_type = EdgeIx;

    constexpr Unweighted() noexcept = default;
    constexpr explicit Unweighted(EdgeIx) noexcept {}

    template <typename V>
    constexpr explicit Unweighted(const V&) noexcept {}

    [[nodiscard]]
    constexpr weight_type operator[](EdgeIx) const noexcept {
        return 1;
    }

    constexpr void swap_to_back(EdgeIx) noexcept {}
    constexpr void push_back(weight_type = 1) noexcept {}
    constexpr void pop_back() noexcept {}
    constexpr void reserve(EdgeIx) noexcept {}
    constexpr void swap(EdgeIx, EdgeIx) noexcept {}
};

template <typename T>
class Weighted {
    template <typename>
    friend class CsrGraph;

    std::vector<T> weights_;
    std::vector<T> degrees_;

    void computeDegrees(std::span<const EdgeIx> ranges) {
        degrees_.assign(ranges.size() - 1, T{});
        for (NodeIx node = 0; node + 1 < ranges.size(); ++node) {
            T degree{};
            for (EdgeIx eix = ranges[node]; eix < ranges[node + 1]; ++eix) {
                degree += weights_[eix];
            }
            degrees_[node] = degree;
        }
    }
public:
    using weight_type = T;

    Weighted() = default;
    explicit Weighted(EdgeIx numEdges) : weights_(numEdges, T{1}) {}
    explicit Weighted(std::vector<T>&& weights) : weights_(std::move(weights)) {}


    [[nodiscard]]
    const T& operator[](EdgeIx e) const noexcept {
        return weights_[e];
    }

    [[nodiscard]]
    T& operator[](EdgeIx e) noexcept {
        return weights_[e];
    }

    void push_back(T weight = T{1}) {
        weights_.push_back(std::move(weight));
    }

    void pop_back() noexcept {
        weights_.pop_back();
    }

    void swap_to_back(EdgeIx eix) {
        std::swap(weights_[eix], weights_.back());
    }

    void reserve(EdgeIx count) {
        weights_.reserve(count);
    }

    void swap(EdgeIx a, EdgeIx b) {
        std::swap(weights_[a], weights_[b]);
    }

    [[nodiscard]]
    const T& weightedDegree(NodeIx node) const noexcept {
        return degrees_[node];
    }
};


template <typename>
struct is_weighted_policy : std::false_type {};

template <typename T>
struct is_weighted_policy<Weighted<T>> : std::true_type {};

template <typename T>
concept ValidWeightPolicy = std::same_as<T, Unweighted> || is_weighted_policy<T>::value;

template <typename T>
using select_policy_t = std::conditional_t<
    ValidWeightPolicy<T>,
    T,
    Weighted<T>
>;

template <typename Spec = Unweighted>
class CsrGraph {

    using Policy = select_policy_t<Spec>;
    NO_UNIQUE_ADDRESS Policy weights{0};

public:
    using EdgeWeight = typename Policy::weight_type;

    NodeIx numNodes{0};
    EdgeIx numEdges{0};                 // number of undirected edges

    std::vector<NodeIx> edges;          // size: 2 * numEdges
    std::vector<EdgeIx> ranges;         // size: numNodes + 1

    CsrGraph() = default;

    CsrGraph(NodeIx n, EdgeIx m) :
        numNodes{n}, numEdges{m}, edges(2 * m), ranges(n + 1), weights(2 * m) {}

    CsrGraph(NodeIx n, EdgeIx m, std::vector<NodeIx>&& ed, std::vector<EdgeIx>&& rg) :
        numNodes{n}, numEdges{m}, edges(std::move(ed)), ranges(std::move(rg)), weights(2 * m) {
        if constexpr (!std::same_as<Policy, Unweighted>) {
            weights.computeDegrees(ranges);
        }
    }

    CsrGraph(NodeIx n, EdgeIx m, std::vector<NodeIx>&& ed, std::vector<EdgeIx>&& rg, std::vector<EdgeWeight>&& wgt)
        requires (!std::same_as<Policy, Unweighted>) :
        numNodes{n}, numEdges{m}, edges(std::move(ed)), ranges(std::move(rg)), weights(std::move(wgt)) {
        weights.computeDegrees(ranges);
    }

    // Forbid Copying
    CsrGraph(const CsrGraph&) = delete;
    CsrGraph& operator=(const CsrGraph&) = delete;

    // Allow Moving
    CsrGraph(CsrGraph&&) noexcept = default;
    CsrGraph& operator=(CsrGraph&&) noexcept = default;


    [[nodiscard]]
    std::span<NodeIx> neighbors(NodeIx node) noexcept {
        return {
            edges.data() + ranges[node],
            ranges[node + 1] - ranges[node]
        };
    }

    [[nodiscard]]
    std::span<const NodeIx> neighbors(NodeIx node) const noexcept {
        return {
            edges.data() + ranges[node],
            ranges[node + 1] - ranges[node]
        };
    }

    [[nodiscard]]
    auto nbegin(NodeIx node) noexcept {
        return edges.begin() + ranges[node];
    }

    [[nodiscard]]
    auto nend(NodeIx node) noexcept {
        return edges.begin() + ranges[node + 1];
    }

    [[nodiscard]]
    auto nbegin(NodeIx node) const noexcept {
        return edges.cbegin() + ranges[node];
    }

    [[nodiscard]]
    auto nend(NodeIx node) const noexcept {
        return edges.cbegin() + ranges[node + 1];
    }

    [[nodiscard]]
    EdgeIx startIndex(NodeIx node) const noexcept {
        return ranges[node];
    }

    [[nodiscard]]
    EdgeIx endIndex(NodeIx node) const noexcept {
        return ranges[node+1];
    }

    [[nodiscard]]
    EdgeIx degree(NodeIx node) const noexcept {
        return ranges[node + 1] - ranges[node];
    }

    [[nodiscard]]
    EdgeWeight edgeWeight(EdgeIx edge) const noexcept {
        return weights[edge];
    }

    [[nodiscard]]
    EdgeWeight weightedDegree(NodeIx node) const noexcept {
        if constexpr (std::same_as<Policy, Unweighted>) {
            return degree(node);
        } else {
            return weights.weightedDegree(node);
        }
    }

    void swapEdges(EdgeIx a, EdgeIx b) {
        std::swap(edges[a], edges[b]);
        weights.swap(a, b);
    }
};

template<typename G>
concept GraphLike = requires(G& g, const G& cg, NodeIx node, EdgeIx eix) {
    typename G::EdgeWeight;

    { g.numNodes } -> std::convertible_to<NodeIx>;
    { g.numEdges } -> std::convertible_to<EdgeIx>;

    { g.edges } -> std::same_as<std::vector<NodeIx>&>;
    { cg.edges } -> std::same_as<const std::vector<NodeIx>&>;

    { g.ranges } -> std::same_as<std::vector<EdgeIx>&>;
    { cg.ranges } -> std::same_as<const std::vector<EdgeIx>&>;

    { g.nbegin(node) };
    { g.nend(node) };
    { cg.nbegin(node) };
    { cg.nend(node) };

    { cg.degree(node) } -> std::same_as<EdgeIx>;
    { cg.weightedDegree(node) } -> std::convertible_to<typename G::EdgeWeight>;
    { cg.edgeWeight(eix) } -> std::convertible_to<typename G::EdgeWeight>;
};

template<typename G>
concept MutableGraph =
    GraphLike<G> && requires(G& g, EdgeIx a, EdgeIx b) {
        { g.swapEdges(a, b) } -> std::same_as<void>;
    };


template <typename Spec = Unweighted>
class DynamicGraph {

    using Policy = select_policy_t<Spec>;
    NO_UNIQUE_ADDRESS Policy weights{0};

public:
    using weight_type = typename Policy::weight_type;

    struct Edge {
        NodeIx fx{0};
        NodeIx tx{0};

        Edge() = default;
        Edge(NodeIx fx, NodeIx tx) : fx(std::min(fx, tx)), tx(std::max(fx, tx)) {}

        [[nodiscard]]
        bool operator==(const Edge& other) const noexcept {
            return fx == other.fx && tx == other.tx;
        }
    };

    NodeIx numNodes{0};
    EdgeIx numEdges{0};

    std::vector<Edge> edges;

    DynamicGraph() = default;

    // m is capacity, not the current number of edges.
    DynamicGraph(NodeIx n, EdgeIx m) : numNodes{n} {
        edges.reserve(m);
        weights.reserve(m);
    }

    DynamicGraph(NodeIx n, EdgeIx m, std::vector<Edge>&& eg )
        : numNodes{n}, numEdges{m}, edges(std::move(eg)), weights(m) {}

    DynamicGraph(NodeIx n, EdgeIx m, std::vector<Edge>&& eg, std::vector<weight_type>&& wgts)
        requires (!std::same_as<Policy, Unweighted>)
        : numNodes{n}, numEdges{m}, edges(std::move(eg)), weights(std::move(wgts)) {}

    // Forbid Copying
    DynamicGraph(const DynamicGraph&) = delete;
    DynamicGraph& operator=(const DynamicGraph&) = delete;

    // Allow Moving
    DynamicGraph(DynamicGraph&&) noexcept = default;
    DynamicGraph& operator=(DynamicGraph&&) noexcept = default;

    NodeIx addNode() noexcept {
        return numNodes++;
    }

    void addEdge(NodeIx fx, NodeIx tx, weight_type weight = weight_type{1}) {
        edges.emplace_back(fx, tx);
        weights.push_back(std::move(weight));
        ++numEdges;
    }

    void popEdge() noexcept {
        edges.pop_back();
        weights.pop_back();
        --numEdges;
    }

    void removeEdgeByIx(EdgeIx eix) {
        std::swap(edges[eix], edges.back());
        weights.swap_to_back(eix);
        popEdge();
    }

    [[nodiscard]]
    const Edge& edge(EdgeIx eix) const noexcept {
        return edges[eix];
    }

    [[nodiscard]]
    weight_type edgeWeight(EdgeIx eix) const noexcept {
        return weights[eix];
    }

    [[nodiscard]]
    CsrGraph<Spec> finalize() const {
        std::vector<NodeIx> csrEdges(2 * numEdges);
        std::vector<EdgeIx> ranges(numNodes + 1);

        for (const Edge& edge : edges) {
            ++ranges[edge.fx];
            ++ranges[edge.tx];
        }

        EdgeIx offset = 0;

        for (NodeIx node = 0; node < numNodes; ++node) {
            const EdgeIx degree = ranges[node];
            ranges[node] = offset;
            offset += degree;
        }

        ranges[numNodes] = offset;

        std::vector<EdgeIx> next = ranges;

        if constexpr (std::same_as<Policy, Unweighted>) {
            for (const Edge& edge : edges) {
                csrEdges[next[edge.fx]++] = edge.tx;
                csrEdges[next[edge.tx]++] = edge.fx;
            }

            return CsrGraph<Spec>(
                numNodes, numEdges, std::move(csrEdges), std::move(ranges)
            );

        } else {
            std::vector<weight_type> csrWeights(2 * numEdges);

            for (EdgeIx i = 0; i < numEdges; ++i) {
                const Edge& edge = edges[i];
                const weight_type weight = weights[i];

                csrEdges[next[edge.fx]] = edge.tx;
                csrWeights[next[edge.fx]] = weight;
                ++next[edge.fx];

                csrEdges[next[edge.tx]] = edge.fx;
                csrWeights[next[edge.tx]] = weight;
                ++next[edge.tx];
            }

            return CsrGraph<Spec>(
                numNodes, numEdges, std::move(csrEdges), std::move(ranges), std::move(csrWeights)
            );
        }
    }
};

#endif //RCUT_GRAPH_H
