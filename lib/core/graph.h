//
// Created by robin on 13.03.2025.
//

#ifndef RCUT_GRAPH_H
#define RCUT_GRAPH_H

#include <type_traits>
#include <algorithm>

#include "definitions.h"
#include <vector>

#define NEIGHBORS(__g, __i) auto __end_it = __g.nend(__i); for(auto _it = __g.nbegin(__i); _it != __end_it; ++_it)
#define NEIGHBORSPTR(__g, __i) auto __end_it = __g->nend(__i); for(auto _it = __g->nbegin(__i); _it != __end_it; ++_it)
#define ITER_EIX_P(__g, __i, __eix) for(EdgeIx __eix = __g->ranges.at(__i); __eix < __g->ranges.at(__i+1); __eix++)
#define ITER_EIX(__g, __i, __eix) for(EdgeIx __eix = __g.ranges.at(__i); __eix < __g.ranges.at(__i+1); __eix++)


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
};

template <typename>
struct is_weighted_policy : std::false_type {};

template <typename T>
struct is_weighted_policy<Weighted<T>> : std::true_type {};

template <typename T>
concept WeightPolicy = is_weighted_policy<T>::value || std::is_same_v<T, Unweighted>;


template <typename Spec = Unweighted>
class Graph {

    // If Spec is already a policy (Unweighted or Weighted<T>), use it directly.
    // Otherwise, assume it's a scalar (like double) and wrap it in Weighted<Spec>.
    using Policy = std::conditional_t<
        WeightPolicy<Spec>,
        Spec,
        Weighted<Spec>
    >;

    [[no_unique_address]]
    Policy weights{0};

public:
    using weight_type = typename Policy::weight_type;

    NodeIx numNodes{0};
    EdgeIx numEdges{0};                 // number of undirected edges

    std::vector<NodeIx> edges;          // size: 2 * numEdges
    std::vector<EdgeIx> ranges;         // size: numNodes + 1


    // std::vector<EdgeIx> map;            // in an undirected graph: maps edge in one direction to its counterpart edge
//    std::vector<EdgeIx> edgeToNode;


    Graph() = default;

    Graph(NodeIx n, EdgeIx m) :
        numNodes{n}, numEdges{m}, edges(2 * m), ranges(n + 1), weights(2 * m) {}

    Graph(std::vector<NodeIx>&& ed, std::vector<EdgeIx>&& rg, NodeIx n, EdgeIx m) :
            numNodes{n}, numEdges{m}, edges(std::move(ed)), ranges(std::move(rg)), weights(2 * m) {
    }
    Graph(std::vector<NodeIx>&& ed, std::vector<EdgeIx>&& rg, std::vector<EdgeWeight>&& wgt, NodeIx n, EdgeIx m) :
            numNodes{n}, numEdges{m}, edges(std::move(ed)), ranges(std::move(rg)), weights(std::move(wgt)) {
    }

    // Forbid Copying
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;

    // Allow Moving
    Graph(Graph&&) noexcept = default;
    Graph& operator=(Graph&&) noexcept = default;

//    void populateEdgeLookup() {
//        NodeIx nix = 0;
//        for(EdgeIx eix = 0; eix < 2 * numEdges; eix++) {
//            while(ranges[nix+1] <= eix) nix++;
//            edgeToNode[eix] = nix;
//        }
//    }

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
        return ranges.at(node);
    }

    [[nodiscard]] weight_type edgeWeight(EdgeIx e) const {
        return weights[e];
    }

    [[nodiscard]] EdgeIx degree(NodeIx node) const {
        return static_cast<EdgeIx>(std::distance(nbegin(node), nend(node)));
    }

    // bool operator==(const Graph& other) const {
    //     return edges == other.edges
    //         && ranges == other.ranges
    //         && weights == other.weights
    //         && numNodes == other.numNodes
    //         && numEdges == other.numEdges;
    // }
};

template <typename WeightPolicy = Unweighted>
class DynamicGraph {
    struct Edge {
        NodeIx fx{0}, tx{0}; // from index, to index

        Edge() = default;

        Edge(NodeIx fx, NodeIx tx) : fx(fx), tx(tx) {}

        bool operator==(const Edge& other) const {
            return fx == other.fx && tx == other.tx;
        }
    };

public:
    using weight_type = typename WeightPolicy::weight_type;

    NodeIx numNodes{0};
    EdgeIx numEdges{0};

    std::vector<Edge> edges{};
    WeightPolicy weights{0};

    DynamicGraph(NodeIx n, EdgeIx m, std::vector<Edge>&& eg) : numNodes(n), numEdges(m), edges(std::move(eg)), weights(m) {}

    DynamicGraph(NodeIx n, EdgeIx m, std::vector<Edge>&& eg, std::vector<weight_type>&& wgts) : numNodes(n), numEdges(m), edges(std::move(eg)), weights(std::move(wgts)) {}

    void addEdge(NodeIx fx, NodeIx tx, weight_type w = weight_type{1}) {
        edges.emplace_back(std::min(fx, tx), std::max(fx, tx));
        weights.push_back(std::move(w));
        numEdges++;
    }

    void removeEdge(NodeIx fx, NodeIx tx) {
        if (fx > tx) std::swap(fx, tx);
        auto eit = std::find(edges.begin(), edges.end(), Edge{fx, tx});
        if (eit != edges.end()) {
            const auto ix = static_cast<EdgeIx>(std::distance(edges.begin(), eit));
            removeEdgeByIx(ix);
        }
    }

    void popEdge() {
        edges.pop_back();
        weights.pop_back();
        numEdges--;
    }

    void removeEdgeByIx(EdgeIx eix) {
        std::swap(edges.at(eix), edges.back());
        weights.swap_to_back(eix);
        popEdge();
    }

    NodeIx addNode() {
        return numNodes++;
    }

    /// Convert this DynamicGraph instance to a Graph in CSR format
    Graph<WeightPolicy> finalize() const {
        if(edges.empty()) return {};
        std::vector<Edge> fullEdges(2 * numEdges);
        std::vector<EdgeIx> rg(numNodes + 1);
        // std::vector<EdgeIx> map(2 * numEdges);

        EdgeIx eix = 0;
        for(const auto& [fx, tx]: edges) {
            fullEdges.at(eix++) = {fx, tx};
            fullEdges.at(eix++) = {tx, fx};
        }

        std::sort(fullEdges.begin(), fullEdges.end(), [](const Edge& a, const Edge& b){
            return a.fx == b.fx ? a.tx < b.tx : a.fx < b.fx;
        });

        eix = 0;
        NodeIx nix = 0;
        std::vector<NodeIx> eg(2 * numEdges);
        while(nix < numNodes) {
            rg.at(nix) = eix;
            while(eix < fullEdges.size() && fullEdges.at(eix).fx == nix) {
                eix++;
            }
            nix++;
        }
        rg.at(numNodes) = eix;

        std::vector<EdgeIx> tmp(rg);
        for(eix = 0; eix < 2 * numEdges; eix++) {
            Edge e = fullEdges.at(eix);
            // EdgeIx pix = tmp.at(e.tx)++;
            // map.at(eix) = pix;
            // map.at(pix) = eix;
            eg.at(eix) = e.tx;
        }

        return {std::move(eg), std::move(rg), numNodes, numEdges};
    }
};

#endif //RCUT_GRAPH_H
