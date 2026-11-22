//
// Created by robin on 19.11.2025.
//

#ifndef RCUT_EXPANDER_HIERARCHY_H
#define RCUT_EXPANDER_HIERARCHY_H

#include "algorithms/expander_decomposition/cpu.h"
#include "utils/timer.h"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

template<GraphLike Graph>
CsrGraph<typename Graph::EdgeWeight> contractGraph(const Graph& graph, NodeIx numClusters, const std::vector<NodeIx>& parents) {
    using EdgeWeight = typename Graph::EdgeWeight;

    struct ContractedEdge {
        NodeIx fx;
        NodeIx tx;
        EdgeWeight weight;
    };

    std::vector<ContractedEdge> edges;
    edges.reserve(graph.numEdges);

    for (NodeIx u = 0; u < graph.numNodes; ++u) {
        const NodeIx cu = parents[u];
        for (EdgeIx eix = graph.startIndex(u); eix < graph.endIndex(u); ++eix) {
            const NodeIx v = graph.edges[eix];
            if (u > v) continue;

            const NodeIx cv = parents[v];
            if (cu == cv) continue;

            edges.push_back({
                std::min(cu, cv),
                std::max(cu, cv),
                graph.edgeWeight(eix)
            });
        }
    }

    std::sort(edges.begin(), edges.end(), [](const ContractedEdge& a, const ContractedEdge& b) {
        return std::tie(a.fx, a.tx) < std::tie(b.fx, b.tx);
    });

    DynamicGraph<EdgeWeight> contracted(numClusters, static_cast<EdgeIx>(edges.size()));

    for (EdgeIx i = 0; i < edges.size(); /* incremented inside */) {
        ContractedEdge combined = edges[i++];
        while (i < edges.size() && edges[i].fx == combined.fx && edges[i].tx == combined.tx) {
            combined.weight += edges[i].weight;
            ++i;
        }

        contracted.addEdge(combined.fx, combined.tx, combined.weight);
    }

    return contracted.finalize();
}

// template<MutableGraph InputGraph>
// CsrGraph<typename InputGraph::EdgeWeight> makeWeightedCsrGraph(const InputGraph& graph) {
//     using EdgeWeight = typename InputGraph::EdgeWeight;
//
//     std::vector<NodeIx> edges = graph.edges;
//     std::vector<EdgeIx> ranges = graph.ranges;
//     std::vector<EdgeWeight> weights(edges.size());
//
//     for (EdgeIx eix = 0; eix < edges.size(); ++eix) {
//         weights[eix] = graph.edgeWeight(eix);
//     }
//
//     return CsrGraph<EdgeWeight>(
//         graph.numNodes,
//         graph.numEdges,
//         std::move(edges),
//         std::move(ranges),
//         std::move(weights)
//     );
// }


template<typename EdgeWeight>
class ExpanderHierarchy {

public:
    using Graph = CsrGraph<EdgeWeight>;

    struct Level {
        Graph graph;
        std::vector<EdgeWeight> volumes;    // size: number of clusters; volume of each subtree
        std::vector<EdgeWeight> cutSizes;   // size: number of clusters; value of cut leaving each cluster in the current graph
        std::vector<NodeIx> parents;        // size: numNodes of graph in level below; gives clusterId (= nix of parent) for each nix in the graph below
        std::vector<NodeIx> children;        // packed inverse of parents
        std::vector<EdgeIx> childrenStart;   // size: number of clusters + 1
        double seconds = 0;                  // time of the expander decomposition that produced this level

        [[nodiscard]]
        std::span<const NodeIx> childrenOf(NodeIx parent) const {
            return {
                children.data() + childrenStart[parent],
                childrenStart[parent + 1] - childrenStart[parent]
            };
        }
    };

    // levels[i].graph.numNodes == levels[i].volumes.size() == levels[i].cutSizes.size() == levels[i+1].parents.size()

private:
    std::vector<Level> levels;
    NodeIx originalNumNodes{0};
    bool complete = false;

    template<MutableGraph PartGraph>
    Level contract(Partition<PartGraph>& part) const;

    template<bool first, MutableGraph PartGraph>
    Level contract(Partition<PartGraph>& part) const;

public:
    explicit ExpanderHierarchy(NodeIx n) : originalNumNodes{n} {}

    ExpanderHierarchy(const ExpanderHierarchy&) = delete;
    ExpanderHierarchy& operator=(const ExpanderHierarchy&) = delete;
    ExpanderHierarchy(ExpanderHierarchy&&) noexcept = default;
    ExpanderHierarchy& operator=(ExpanderHierarchy&&) noexcept = default;

    void build() {
        build([](Graph& graph, const Config& config) {
            return expander_decomposition::compute(graph, config);
        });
    }

    template<class Decomposer>
    void build(Decomposer&& decompose);

    template<MutableGraph PartGraph>
    void appendLevel(Partition<PartGraph>& part, double seconds = 0) {
        levels.push_back(contract(part));
        levels.back().seconds = seconds;
    }

    template<class Decomposer>
    void buildUpperLevels(Decomposer&& decompose, Config conf = {});

    template<typename T>
    double predict(const T& demands) const;

    size_t height() const {
        return levels.size();
    }

    NodeIx getNumNodes() const {
        return originalNumNodes;
    }

    const Level& lvl(NodeIx l) const {
        return levels[l];
    }

    [[nodiscard]]
    std::span<const NodeIx> children(NodeIx l, NodeIx parent) const {
        return levels[l].childrenOf(parent);
    }
};

template<typename EdgeWeight>
template<MutableGraph PartGraph>
typename ExpanderHierarchy<EdgeWeight>::Level
ExpanderHierarchy<EdgeWeight>::contract(Partition<PartGraph>& part) const {
    static_assert(std::same_as<typename PartGraph::EdgeWeight, EdgeWeight>);
    if(levels.empty()) return contract<true>(part);
    else return contract<false>(part);
}


template<typename EdgeWeight>
template<bool first, MutableGraph PartGraph>
typename ExpanderHierarchy<EdgeWeight>::Level
ExpanderHierarchy<EdgeWeight>::contract(Partition<PartGraph>& part) const {
    static_assert(std::same_as<typename PartGraph::EdgeWeight, EdgeWeight>);
    const PartGraph& graph = part.getGraph();
    if constexpr (!first) {
        assert(graph.numNodes == levels.back().graph.numNodes);
    }

    std::vector<EdgeWeight> volumes(part.numClusters(), 0);
    std::vector<EdgeWeight> cutSizes(part.numClusters(), 0);

    NodeIx nix = 0;
    std::vector<NodeIx> clusterLookup(graph.numNodes, graph.numNodes + 1);
    for(const Cluster<PartGraph>& cluster: part) {
        if constexpr (first) {
            volumes[nix] = cluster.volume;
        }
        for (const ClusterVertex<PartGraph>& cv: cluster) {
            clusterLookup[cv.nix] = nix;
            if constexpr (!first) {
                volumes[nix] += levels.back().volumes[cv.nix];
            }
        }
        nix++;
    }

    nix = 0;
    for(const Cluster<PartGraph>& cluster: part) {
        for (const ClusterVertex<PartGraph>& cv: cluster) {
            for (EdgeIx eix = cv.end; eix < graph.endIndex(cv.nix); ++eix) {
                NodeIx nb = graph.edges[eix];
                assert(clusterLookup[nb] != nix);
                assert(clusterLookup[nb] != graph.numNodes + 1);
                cutSizes[nix] += graph.edgeWeight(eix);
            }
        }
        assert(cutSizes[nix] == cluster.volume - cluster.internalVolume);
        nix++;
    }

    std::vector<EdgeIx> childrenStart(part.numClusters() + 1);
    for (NodeIx parent : clusterLookup) {
        ++childrenStart[parent + 1];
    }
    for (NodeIx parent = 0; parent < part.numClusters(); ++parent) {
        childrenStart[parent + 1] += childrenStart[parent];
    }

    std::vector<NodeIx> children(graph.numNodes);
    std::vector<EdgeIx> nextChild = childrenStart;
    for (NodeIx child = 0; child < graph.numNodes; ++child) {
        const NodeIx parent = clusterLookup[child];
        children[nextChild[parent]++] = child;
    }

    return {
        contractGraph(graph, part.numClusters(), clusterLookup),
        std::move(volumes),
        std::move(cutSizes),
        std::move(clusterLookup),
        std::move(children),
        std::move(childrenStart)
    };
}

template<typename EdgeWeight>
template<class Decomposer>
void ExpanderHierarchy<EdgeWeight>::buildUpperLevels(Decomposer&& decompose, Config conf) {
    assert(!levels.empty());

    Graph* current = &levels.back().graph;
    Timer t;

    while(current->numNodes > 1) {
        if (current->numEdges == 0) break;

        t.start();
        Partition ed = decompose(*current, conf);

        if(ed.getNumCutEdges() > 0.95 * current->numEdges) {
            conf.targetSparsity /= 2;
            WARN("\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t Adjusting targetSparsity to " << conf.targetSparsity);
            continue;
        }

        const double seconds = t.timeSeconds();
        Level lvl = contract(ed);
        lvl.seconds = seconds;
        levels.push_back(std::move(lvl));
        current = &levels.back().graph;
        ++conf.seed;   // fresh random values on every level

        INFO("Level " << levels.size() << " (" << seconds << "s)\t\t-> " << ed.numClusters() << " clusters with " << ed.getNumCutEdges() << " crossing edges of weight " << ed.getCutSize());

        if (ed.getCutSize() == EdgeWeight{}) {
            // TODO: add single super node on top
            break;
        }
    }

    complete = true;
}


template<typename EdgeWeight>
template<typename T>
double ExpanderHierarchy<EdgeWeight>::predict(const T& demands) const {
    assert(demands.size() == originalNumNodes);
    assert(complete);

    std::vector<double> current(std::begin(demands), std::end(demands));
    std::vector<double> next(originalNumNodes, 0.0);

    const double totalDemand = std::accumulate(current.begin(), current.end(), 0.0);
    if (std::abs(totalDemand) > 1e-8) {
        throw std::invalid_argument("Tree cut sparsifier prediction requires demand to sum to zero.");
    }

    double maxCongestion = 0.0;

    for (const Level& lvl : levels) {
        const NodeIx numClusters = lvl.graph.numNodes;
        std::fill_n(next.begin(), numClusters, 0.0);

        for (NodeIx clusterId = 0; clusterId < numClusters; ++clusterId) {
            double clusterDemand = 0.0;
            for (NodeIx child : lvl.childrenOf(clusterId)) {
                clusterDemand += current[child];
            }

            if (lvl.cutSizes[clusterId] > 0) {
                const double congestion = std::abs(clusterDemand) / static_cast<double>(lvl.cutSizes[clusterId]);
                maxCongestion = std::max(maxCongestion, congestion);
            } else if (std::abs(clusterDemand) > 1e-8) {
                return std::numeric_limits<double>::infinity();
            }

            next[clusterId] = clusterDemand;
        }

        std::swap(current, next);
    }

    return maxCongestion;
}

template<MutableGraph InputGraph, class Decomposer>
    requires std::invocable<Decomposer&, InputGraph&, const Config&>
ExpanderHierarchy<typename InputGraph::EdgeWeight> buildExpanderHierarchy(InputGraph& graph, Decomposer&& decompose, Config conf = {}) {
    using EdgeWeight = typename InputGraph::EdgeWeight;

    ExpanderHierarchy<EdgeWeight> hierarchy{graph.numNodes};

    while (true) {
        Timer t;
        t.start();
        Partition first = decompose(graph, conf);

        if(first.getNumCutEdges() > 0.95 * graph.numEdges) {
            conf.targetSparsity /= 2;
            WARN("\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t Adjusting targetSparsity to " << conf.targetSparsity);
            continue;
        }

        const double seconds = t.timeSeconds();
        INFO("Level 1 (" << seconds << "s)\t\t-> " << first.numClusters() << " clusters with " << first.getNumCutEdges() << " crossing edges of weight " << first.getCutSize());

        hierarchy.appendLevel(first, seconds);
        break;
    }

    ++conf.seed;   // fresh random values on every level
    hierarchy.buildUpperLevels(std::forward<Decomposer>(decompose), conf);
    return hierarchy;
}

template<MutableGraph InputGraph>
ExpanderHierarchy<typename InputGraph::EdgeWeight> buildExpanderHierarchy(InputGraph& graph, const Config& config = {}) {
    return buildExpanderHierarchy(
        graph,
        [](auto& current, const Config& levelConfig) {
            return expander_decomposition::compute(current, levelConfig);
        },
        config
    );
}



#endif //RCUT_EXPANDER_HIERARCHY_H
