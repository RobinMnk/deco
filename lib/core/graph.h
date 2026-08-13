//
// Created by robin on 13.03.2025.
//

#ifndef RCUT_GRAPH_H
#define RCUT_GRAPH_H

#include <type_traits>
#include <algorithm>
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

    constexpr weight_type operator[](EdgeIx) const noexcept {
        return 1;
    }

    constexpr void swap_to_back(EdgeIx) noexcept {}
    constexpr void push_back(weight_type = 1) noexcept {}
    constexpr void pop_back() noexcept {}
    constexpr void reserve(EdgeIx) noexcept {}
};

template <typename T>
class Weighted {
    std::vector<T> weights_;
public:
    using weight_type = T;

    Weighted() = default;
    explicit Weighted(EdgeIx numEdges) : weights_(numEdges, T{1}) {}
    explicit Weighted(std::vector<T>&& wgt) : weights_(std::move(wgt)) {}

    T operator[](EdgeIx e) const {
        return weights_[e];
    }

    T& operator[](EdgeIx e) {
        return weights_[e];
    }

    void push_back(T weight = T{1}) {
        weights_.push_back(std::move(weight));
    }

    void pop_back() {
        weights_.pop_back();
    }

    void swap_to_back(EdgeIx eix) {
        std::swap(weights_[eix], weights_.back());
    }

    void reserve(EdgeIx m) {
        weights_.reserve(m);
    }
};

template <typename>
struct is_weighted_policy : std::false_type {};

template <typename T>
struct is_weighted_policy<Weighted<T>> : std::true_type {};

template <typename T>
concept UnweightedPolicy = std::is_same_v<T, Unweighted>;

// If T is already a policy (Unweighted or Weighted<T>), use it directly.
// Otherwise, assume it's a scalar (like double) and wrap it in Weighted<T>.
template <typename T>
using select_policy_t = std::conditional_t<
    is_weighted_policy<T>::value || UnweightedPolicy<T>,
    T,
    Weighted<T>
>;

template <typename Spec = Unweighted>
class Graph {

    using Policy = select_policy_t<Spec>;
    NO_UNIQUE_ADDRESS Policy weights{0};

public:
    using weight_type = typename Policy::weight_type;

    NodeIx numNodes{0};
    EdgeIx numEdges{0};                 // number of undirected edges

    std::vector<NodeIx> edges;          // size: 2 * numEdges
    std::vector<EdgeIx> ranges;         // size: numNodes + 1

    Graph() = default;

    Graph(NodeIx n, EdgeIx m) :
        numNodes{n}, numEdges{m}, edges(2 * m), ranges(n + 1), weights(2 * m) {}

    Graph(NodeIx n, EdgeIx m, std::vector<NodeIx>&& ed, std::vector<EdgeIx>&& rg) :
        numNodes{n}, numEdges{m}, edges(std::move(ed)), ranges(std::move(rg)), weights(2 * m) {
    }
    Graph(NodeIx n, EdgeIx m, std::vector<NodeIx>&& ed, std::vector<EdgeIx>&& rg, std::vector<weight_type>&& wgt)
        requires (!UnweightedPolicy<Policy>) :
        numNodes{n}, numEdges{m}, edges(std::move(ed)), ranges(std::move(rg)), weights(std::move(wgt)) {
    }

    // Forbid Copying
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;

    // Allow Moving
    Graph(Graph&&) noexcept = default;
    Graph& operator=(Graph&&) noexcept = default;

    [[nodiscard]] std::vector<NodeIx>::iterator nbegin(NodeIx node) {
        return edges.begin() + ranges[node];
    }

    [[nodiscard]] std::vector<NodeIx>::iterator nend(NodeIx node) {
        return edges.begin() + ranges[node + 1];
    }

    [[nodiscard]] std::vector<NodeIx>::const_iterator nbegin(NodeIx node) const {
        return edges.cbegin() + ranges[node];
    }

    [[nodiscard]] std::vector<NodeIx>::const_iterator nend(NodeIx node) const {
        return edges.cbegin() + ranges[node + 1];
    }

    [[nodiscard]] EdgeIx startIndex(NodeIx node) const {
        return ranges[node];
    }

    [[nodiscard]] weight_type edgeWeight(EdgeIx e) const {
        return weights[e];
    }

    [[nodiscard]] EdgeIx degree(NodeIx node) const {
        return ranges[node + 1] - ranges[node];
    }

    // bool operator==(const Graph& other) const {
    //     return edges == other.edges
    //         && ranges == other.ranges
    //         && weights == other.weights
    //         && numNodes == other.numNodes
    //         && numEdges == other.numEdges;
    // }
};

template <typename Spec = Unweighted>
class DynamicGraph {

    using Policy = select_policy_t<Spec>;
    NO_UNIQUE_ADDRESS Policy weights{0};

public:
    struct Edge {
        NodeIx fx{0}, tx{0}; // from index, to index

        Edge() = default;

        Edge(NodeIx fx, NodeIx tx) : fx(fx), tx(tx) {}

        bool operator==(const Edge& other) const {
            return fx == other.fx && tx == other.tx;
        }
    };

    using weight_type = typename Policy::weight_type;

    NodeIx numNodes{0};
    EdgeIx numEdges{0};

    std::vector<Edge> edges{};

    DynamicGraph() = default;

    DynamicGraph(NodeIx n, EdgeIx m) : numNodes(n) {
        edges.reserve(m);
        weights.reserve(m);
    }

    DynamicGraph(NodeIx n, EdgeIx m, std::vector<Edge>&& eg) : numNodes(n), numEdges(m), edges(std::move(eg)), weights(m) {}

    DynamicGraph(NodeIx n, EdgeIx m, std::vector<Edge>&& eg, std::vector<weight_type>&& wgts)
        requires (!UnweightedPolicy<Policy>) :
        numNodes(n), numEdges(m), edges(std::move(eg)), weights(std::move(wgts)) {}

    // Forbid Copying
    DynamicGraph(const DynamicGraph&) = delete;
    DynamicGraph& operator=(const DynamicGraph&) = delete;

    // Allow Moving
    DynamicGraph(DynamicGraph&&) noexcept = default;
    DynamicGraph& operator=(DynamicGraph&&) noexcept = default;

    void addEdge(NodeIx fx, NodeIx tx, weight_type w = weight_type{1}) {
        edges.emplace_back(std::min(fx, tx), std::max(fx, tx));
        weights.push_back(std::move(w));
        numEdges++;
    }

    void popEdge() {
        edges.pop_back();
        weights.pop_back();
        numEdges--;
    }

    void removeEdgeByIx(EdgeIx eix) {
        std::swap(edges[eix], edges.back());
        weights.swap_to_back(eix);
        popEdge();
    }

    NodeIx addNode() {
        return numNodes++;
    }

    [[nodiscard]] weight_type edgeWeight(EdgeIx e) const { return weights[e]; }
    [[nodiscard]] const Edge& edge(EdgeIx e) const { return edges[e]; }

    /// Convert this DynamicGraph instance to a Graph in CSR format
    [[nodiscard]] Graph<Spec> finalize() const {
        std::vector<NodeIx> eg(2 * numEdges);
        std::vector<EdgeIx> rg(numNodes + 1, 0);
        std::vector<weight_type> wgt;

        // Allocate weights only if the policy actually uses them
        if constexpr (!UnweightedPolicy<Policy>) {
            wgt.resize(2 * numEdges);
        }

        // Temporary structure to hold directed edge + weight during sorting
        struct FullEdge {
            NodeIx fx;
            NodeIx tx;
            weight_type weight;
        };

        std::vector<FullEdge> fullEdges;
        fullEdges.reserve(2 * numEdges);

        for (EdgeIx i = 0; i < numEdges; ++i) {
            const auto& [fx, tx] = edges[i];
            weight_type w = edgeWeight(i);

            fullEdges.push_back({fx, tx, w});
            fullEdges.push_back({tx, fx, w});
        }

        // Sort edges by source node (fx), then target node (tx)
        std::sort(fullEdges.begin(), fullEdges.end(), [](const FullEdge& a, const FullEdge& b) {
            return a.fx == b.fx ? a.tx < b.tx : a.fx < b.fx;
        });

        // Build CSR ranges and edges (and weights if applicable)
        EdgeIx eix = 0;
        NodeIx nix = 0;
        while (nix < numNodes) {
            rg[nix] = eix;
            while (eix < fullEdges.size() && fullEdges[eix].fx == nix) {
                eg[eix] = fullEdges[eix].tx;
                if constexpr (!UnweightedPolicy<Policy>) {
                    wgt[eix] = fullEdges[eix].weight;
                }
                eix++;
            }
            nix++;
        }
        rg[numNodes] = eix;

        if constexpr (UnweightedPolicy<Policy>) {
            return Graph<Spec>(numNodes, numEdges, std::move(eg), std::move(rg));
        } else {
            return Graph<Spec>(numNodes, numEdges, std::move(eg), std::move(rg), std::move(wgt));
        }
    }
};

#endif //RCUT_GRAPH_H
