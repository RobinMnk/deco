//
// The module deco._core: the computations of the Python package. The public API is in python/deco/__init__.py, which
// checks the arguments, converts SciPy matrices and wraps the results in classes.
//

#include "deco.h"
#include "utils/graph_io.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>

#include <concepts>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

// A graph, converted once and reused by every computation. The computations reorder its adjacency lists and run
// without the GIL, so the mutex lets only one of them at a time work on the graph.
struct Graph {
    AnyGraph graph;
    std::mutex mutex;
};

using AnyHierarchy = std::variant<ExpanderHierarchy<EdgeIx>, ExpanderHierarchy<IntWeight>, ExpanderHierarchy<frac_t>>;

struct Sparsifier {
    AnyHierarchy hierarchy;
};

using Indices = nb::ndarray<const NodeIx, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
template<class T>
using Values = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

// A NumPy array that takes over `values`
template<class T>
nb::ndarray<nb::numpy, T, nb::ndim<1>> toNumpy(std::vector<T>&& values) {
    auto* owner = new std::vector<T>(std::move(values));
    nb::capsule deleter(owner, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
    return {owner->data(), {owner->size()}, deleter};
}

template<class GraphT>
const char* weightKind() {
    if constexpr (std::same_as<GraphT, CsrGraph<>>) return "unweighted";
    else if constexpr (std::floating_point<typename GraphT::EdgeWeight>) return "real";
    else return "integer";
}

std::unique_ptr<Graph> fromMtx(const std::string& path) {
    auto result = std::make_unique<Graph>();
    nb::gil_scoped_release release;
    try {
        result->graph = load_mtx_dynamic(path);
    } catch (const std::runtime_error& e) {
        throw std::invalid_argument(e.what());   // a file that cannot be read is invalid input: ValueError
    }
    return result;
}

// The graph with the edges {u[i], v[i]} of weight weights[i] (no weights: unweighted)
template<class Spec, class T = EdgeIx>
std::unique_ptr<Graph> fromEdges(NodeIx n, const Indices& u, const Indices& v, const Values<T>* weights = nullptr) {
    const size_t m = u.shape(0);
    if (v.shape(0) != m || (weights && weights->shape(0) != m)) throw std::invalid_argument("the edge arrays differ in length");
    if (n == std::numeric_limits<NodeIx>::max()) throw std::invalid_argument("too many nodes");
    if (2 * uint64_t{m} > std::numeric_limits<EdgeIx>::max()) throw std::invalid_argument("too many edges (at most 2^31 - 1)");

    auto result = std::make_unique<Graph>();
    nb::gil_scoped_release release;
    DynamicGraph<Spec> dynamic(n, static_cast<EdgeIx>(m));
    for (size_t i = 0; i < m; ++i) {
        if (u(i) >= n || v(i) >= n) throw std::invalid_argument("node index out of range");
        if constexpr (std::same_as<Spec, Unweighted>) dynamic.addEdge(u(i), v(i));
        else dynamic.addEdge(u(i), v(i), (*weights)(i));
    }
    result->graph = dynamic.finalize();
    return result;
}

/**
 * The graph with the adjacency lists adjacency[ranges[v] .. ranges[v + 1]) and their weights: a symmetric sparse
 * matrix without diagonal in CSR format, which stores every edge at both of its nodes. The caller checks the symmetry.
 */
template<class Spec, class T = EdgeIx>
std::unique_ptr<Graph> fromCsr(NodeIx n, const Indices& ranges, const Indices& adjacency, const Values<T>* weights = nullptr) {
    const size_t size = adjacency.shape(0);
    if (n == std::numeric_limits<NodeIx>::max() || ranges.shape(0) != size_t{n} + 1 || ranges(0) != 0 || ranges(n) != size
        || size % 2 != 0 || (weights && weights->shape(0) != size)) {
        throw std::invalid_argument("invalid CSR arrays");
    }

    auto result = std::make_unique<Graph>();
    nb::gil_scoped_release release;
    for (NodeIx v = 0; v < n; ++v) {
        if (ranges(v) > ranges(v + 1)) throw std::invalid_argument("invalid CSR arrays");
    }
    for (size_t i = 0; i < size; ++i) {
        if (adjacency(i) >= n) throw std::invalid_argument("node index out of range");
    }
    std::vector<EdgeIx> rg(ranges.data(), ranges.data() + ranges.shape(0));
    std::vector<NodeIx> ed(adjacency.data(), adjacency.data() + size);
    const auto m = static_cast<EdgeIx>(size / 2);
    if constexpr (std::same_as<Spec, Unweighted>) {
        result->graph = CsrGraph<>(n, m, std::move(ed), std::move(rg));
    } else {
        result->graph = CsrGraph<Spec>(n, m, std::move(ed), std::move(rg), std::vector<T>(weights->data(), weights->data() + size));
    }
    return result;
}

deco::Options makeOptions(double sparsity, double walkThreshold, unsigned seed, const std::string& backend) {
    deco::Options options;
    options.config.targetSparsity = sparsity;
    options.config.randomWalkThreshold = walkThreshold;
    options.config.seed = seed;
    if (backend == "auto") options.backend = deco::Backend::Auto;
    else if (backend == "cpu") options.backend = deco::Backend::Cpu;
    else if (backend == "cuda") options.backend = deco::Backend::Cuda;
    else throw std::invalid_argument("backend must be 'auto', 'cpu' or 'cuda'");
    return options;
}

// Runs compute(graph) on the graph without the GIL, and returns convert(result) with the GIL
template<class Compute, class Convert>
nb::object run(Graph& graph, Compute compute, Convert convert) {
    return std::visit([&](auto& g) -> nb::object {
        auto result = [&] {
            nb::gil_scoped_release release;
            std::lock_guard lock(graph.mutex);
            return compute(g);
        }();
        return convert(std::move(result));
    }, graph.graph);
}

template<class EdgeWeight>
nb::dict clusteringDict(deco::Clustering<EdgeWeight>&& clustering) {
    nb::dict result;
    result["cluster_of"] = toNumpy(std::move(clustering.clusterOf));
    result["num_clusters"] = clustering.numClusters;
    result["cut_edges"] = clustering.cutEdges;
    result["cut_weight"] = clustering.cutWeight;
    return result;
}

nb::object expanderDecomposition(Graph& graph, double sparsity, double walkThreshold, unsigned seed, const std::string& backend) {
    const auto options = makeOptions(sparsity, walkThreshold, seed, backend);
    return run(graph,
        [&](auto& g) { return deco::expanderDecomposition(g, options); },
        [](auto&& decomposition) -> nb::object {
            const deco::Backend ran = decomposition.backend;
            const DecompositionStats stats = decomposition.stats;
            nb::dict result = clusteringDict(std::move(decomposition));
            result["backend"] = ran == deco::Backend::Cuda ? "cuda" : "cpu";
            result["iterations"] = stats.iterations;
            result["seconds"] = stats.seconds;
            return result;
        });
}

nb::object sparsifier(Graph& graph, double sparsity, double walkThreshold, unsigned seed, const std::string& backend) {
    const auto options = makeOptions(sparsity, walkThreshold, seed, backend);
    return run(graph,
        [&](auto& g) { return deco::expanderHierarchy(g, options); },
        [](auto&& hierarchy) -> nb::object { return nb::cast(std::make_unique<Sparsifier>(Sparsifier{std::move(hierarchy)})); });
}

// Normalized cut of `graph` from `sparsifier`, which was built from this graph
nb::object normalizedCut(Graph& graph, const Sparsifier& sparsifier, NodeIx k) {
    return run(graph,
        [&](auto& g) {
            using Hierarchy = ExpanderHierarchy<typename std::decay_t<decltype(g)>::EdgeWeight>;
            return deco::normalizedCut(g, std::get<Hierarchy>(sparsifier.hierarchy), k);
        },
        [](auto&& clustering) -> nb::object {
            const double value = clustering.normalizedCut;
            nb::dict result = clusteringDict(std::move(clustering));
            result["normalized_cut"] = value;
            return result;
        });
}

NodeIx sparsifierNumNodes(const Sparsifier& sparsifier) {
    return std::visit([](const auto& h) { return h.getNumNodes(); }, sparsifier.hierarchy);
}

// Every level as a dict of parents, volumes, cut_weights and seconds
nb::list sparsifierLevels(Sparsifier& sparsifier) {
    return std::visit([](auto& h) {
        nb::list levels;
        for (NodeIx l = 0; l < h.height(); ++l) {
            const auto& level = h.lvl(l);
            nb::dict item;
            item["parents"] = toNumpy(std::vector(level.parents));
            item["volumes"] = toNumpy(std::vector(level.volumes));
            item["cut_weights"] = toNumpy(std::vector(level.cutSizes));
            item["seconds"] = level.seconds;
            levels.append(item);
        }
        return levels;
    }, sparsifier.hierarchy);
}

double predict(const Sparsifier& sparsifier, const Values<double>& demand) {
    if (demand.shape(0) != sparsifierNumNodes(sparsifier)) throw std::invalid_argument("the demand needs one value per node");
    const std::span<const double> values(demand.data(), demand.shape(0));
    nb::gil_scoped_release release;
    return std::visit([&](const auto& h) { return h.predict(values); }, sparsifier.hierarchy);
}

} // namespace


NB_MODULE(_core, m) {
    const Config defaults;
    m.attr("DEFAULT_SPARSITY") = defaults.targetSparsity;
    m.attr("DEFAULT_WALK_THRESHOLD") = defaults.randomWalkThreshold;
    m.attr("DEFAULT_SEED") = defaults.seed;

    nb::class_<Graph>(m, "Graph")
        .def_static("from_mtx", &fromMtx, "path"_a)
        .def_static("from_unweighted_edges", [](NodeIx n, const Indices& u, const Indices& v) {
            return fromEdges<Unweighted>(n, u, v);
        }, "n"_a, "u"_a, "v"_a)
        .def_static("from_integer_edges", [](NodeIx n, const Indices& u, const Indices& v, const Values<IntWeight>& w) {
            return fromEdges<IntWeight>(n, u, v, &w);
        }, "n"_a, "u"_a, "v"_a, "weights"_a)
        .def_static("from_real_edges", [](NodeIx n, const Indices& u, const Indices& v, const Values<frac_t>& w) {
            return fromEdges<frac_t>(n, u, v, &w);
        }, "n"_a, "u"_a, "v"_a, "weights"_a)
        .def_static("from_unweighted_csr", [](NodeIx n, const Indices& ranges, const Indices& adjacency) {
            return fromCsr<Unweighted>(n, ranges, adjacency);
        }, "n"_a, "ranges"_a, "adjacency"_a)
        .def_static("from_integer_csr", [](NodeIx n, const Indices& ranges, const Indices& adjacency, const Values<IntWeight>& w) {
            return fromCsr<IntWeight>(n, ranges, adjacency, &w);
        }, "n"_a, "ranges"_a, "adjacency"_a, "weights"_a)
        .def_static("from_real_csr", [](NodeIx n, const Indices& ranges, const Indices& adjacency, const Values<frac_t>& w) {
            return fromCsr<frac_t>(n, ranges, adjacency, &w);
        }, "n"_a, "ranges"_a, "adjacency"_a, "weights"_a)
        .def_prop_ro("num_nodes", [](const Graph& g) { return std::visit([](const auto& x) { return x.numNodes; }, g.graph); })
        .def_prop_ro("num_edges", [](const Graph& g) { return std::visit([](const auto& x) { return x.numEdges; }, g.graph); })
        .def_prop_ro("weights", [](const Graph& g) {
            return std::visit([](const auto& x) { return weightKind<std::decay_t<decltype(x)>>(); }, g.graph);
        });

    nb::class_<Sparsifier>(m, "Sparsifier")
        .def_prop_ro("num_nodes", &sparsifierNumNodes)
        .def("levels", &sparsifierLevels)
        .def("predict", &predict, "demand"_a);

    m.def("cuda_available", &deco::cudaAvailable);
    m.def("expander_decomposition", &expanderDecomposition,
          "graph"_a, "sparsity"_a, "walk_threshold"_a, "seed"_a, "backend"_a);
    m.def("sparsifier", &sparsifier, "graph"_a, "sparsity"_a, "walk_threshold"_a, "seed"_a, "backend"_a);
    m.def("normalized_cut", &normalizedCut, "graph"_a, "sparsifier"_a, "k"_a);
}
