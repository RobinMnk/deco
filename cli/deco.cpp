
#include "algorithms/applications/normalized_cut.h"
#include "algorithms/tree_cut_sparsifier/expander_hierarchy.h"
#ifdef DECO_ENABLE_CUDA
#include "algorithms/expander_decomposition/cuda.h"
#endif

#include "algorithms/expander_decomposition/cpu.h"
#include "utils/graph_io.h"
#include "utils/timer.h"

#include <type_traits>

void checkBuildMode() {
    #ifdef NDEBUG
        std::cout << "Running in RELEASE mode!" << std::endl;
    #else
        std::cout << "Running in DEBUG mode" << std::endl;
    #endif
}

template<typename T>
void report(Partition<T>& part, const T& G, const DecompositionStats& stats) {
    std::cout << "iterations: " << stats.iterations << "\n"
              << "time: " << std::llround(stats.seconds * 1000) << "\n"
              << "numClusters: " << part.numClusters() << "\n"
              << "cutEdges: " << part.getNumCutEdges() << "\n"
              << "cutWeight: " << part.getCutSize() << "\n"
              << "nodes: " << G.numNodes << "\n"
              << "edges: " << G.numEdges << std::endl;
}

template<MutableGraph GraphT>
void run(GraphT& G) {
    using EdgeWeight = typename GraphT::EdgeWeight;
    constexpr std::string_view weightType =
        std::same_as<GraphT, CsrGraph<>> ? "unweighted" :
        std::same_as<EdgeWeight, IntWeight> ? "IntWeight" :
        std::same_as<EdgeWeight, frac_t> ? "frac_t" :
                                           "unknown";

    std::cout << "Graph: " << G.numNodes << " nodes, " << G.numEdges
              << " edges, weight type: " << weightType << std::endl;

    // auto expDec = buildExpanderHierarchy(G);
    DecompositionStats stats;
    auto expDec = expander_decomposition::compute(G, Config{}, &stats);

    report(expDec, G, stats);

    // printf("time: %lldms\nnodes: %d\nedges: %d\n", millis, G.numNodes, G.numEdges);

    // t.start();
    // auto NC = NormalizedCut(expDec);
    // auto kPart = NC.compute(G, 16);
    // double ncVal = compute_normalized_cut(kPart);
    // millis = t.timeMillis();
    //
    // printf("NC time: %lldms\nNC: %f\n", millis, ncVal);

    // printf("time: %lld\nnumClusters: %d\ncutEdges: %d\nnodes: %d\nedges: %d\n", millis, expDec.numClusters(), expDec.getNumCutEdges(), G.numNodes, G.numEdges);
    // std::cout << "Cut size: " << expDec.getCutSize() << std::endl;
}

void test(MutableGraph auto& G) {
    using GraphT = std::remove_reference_t<decltype(G)>;
    Partition<GraphT> decomposition{G};
}

int main(int argc, char* argv[]) {
    checkBuildMode();

    std::string filename = argv[1];

    auto G2 = load_mtx_dynamic(filename);

    // std::visit([](auto& graph) {
    //     run(graph);
    // }, G2);

    #ifdef DECO_ENABLE_CUDA
    auto G = load_mtx_unweighted(filename);
    DecompositionStats stats;
    auto part = cudaExpanderDecomposition(G, Config{}, &stats);
    report(part, G, stats);
    #endif
}
