//
// Command-line interface of Deco, see usage() below.
// The summary goes to stdout (as "key: value" lines or one JSON object), errors and warnings to stderr.
//

#include "deco.h"
#include "utils/graph_io.h"
#include "utils/timer.h"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace {

// ---------------------------------------------------------------------------------------------------------------------
// Arguments
// ---------------------------------------------------------------------------------------------------------------------

template<class T>
std::string number(T value) {
    char buffer[32];
    const auto result = std::to_chars(buffer, buffer + sizeof buffer, value);   // doubles: shortest exact form
    return {buffer, result.ptr};
}

std::string usage() {
    const Config defaults;
    return "Usage: Deco <command> <graph.mtx> [options]\n"
           "\n"
           "Commands:\n"
           "  decompose            expander decomposition\n"
           "  sparsifier           congestion approximator (tree cut sparsifier)\n"
           "  ncut                 partition into k clusters with a small normalized cut (needs -k)\n"
           "\n"
           "Options:\n"
           "  --backend B          auto, cpu or cuda (default: auto, which uses CUDA if it can)\n"
           "  --sparsity X         cut a cluster if it has a sweep cut sparser than X (default: " + number(defaults.targetSparsity) + ")\n"
           "  --walk-threshold X   a cluster is done once its walk values differ by less than X (default: " + number(defaults.randomWalkThreshold) + ")\n"
           "  --seed N             seed for the random walks (default: " + number(defaults.seed) + ")\n"
           "  -k N                 number of clusters for ncut\n"
           "  -o, --output FILE    write the cluster of every node to FILE, one line per node\n"
           "                       (sparsifier: its clusters on all levels, separated by spaces)\n"
           "  --json               print the summary as one JSON object\n"
           "  -h, --help           show this help\n"
           "\n"
           "Times are in milliseconds.\n";
}

struct UsageError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct Args {
    std::string command;
    std::string graph;
    deco::Options options;
    NodeIx k = 0;
    std::string output;
    bool json = false;
    bool help = false;
};

template<class T>
T parseNumber(std::string_view option, std::string_view text) {
    T value{};
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        throw UsageError("invalid value for " + std::string(option) + ": '" + std::string(text) + "'");
    }
    return value;
}

deco::Backend parseBackend(std::string_view text) {
    if (text == "auto") return deco::Backend::Auto;
    if (text == "cpu") return deco::Backend::Cpu;
    if (text == "cuda") return deco::Backend::Cuda;
    throw UsageError("invalid value for --backend: '" + std::string(text) + "' (use auto, cpu or cuda)");
}

Args parseArgs(int argc, char* argv[]) {
    Args args;
    std::vector<std::string_view> positional;

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        auto value = [&]() -> std::string_view {
            if (i + 1 >= argc) throw UsageError(std::string(arg) + " needs a value");
            return argv[++i];
        };

        if (arg == "-h" || arg == "--help") args.help = true;
        else if (arg == "--backend") args.options.backend = parseBackend(value());
        else if (arg == "--sparsity") args.options.config.targetSparsity = parseNumber<frac_t>(arg, value());
        else if (arg == "--walk-threshold") args.options.config.randomWalkThreshold = parseNumber<frac_t>(arg, value());
        else if (arg == "--seed") args.options.config.seed = parseNumber<unsigned>(arg, value());
        else if (arg == "-k") args.k = parseNumber<NodeIx>(arg, value());
        else if (arg == "-o" || arg == "--output") args.output = value();
        else if (arg == "--json") args.json = true;
        else if (arg.starts_with('-')) throw UsageError("unknown option " + std::string(arg));
        else positional.push_back(arg);
    }
    if (args.help) return args;

    if (positional.size() != 2) throw UsageError("expected a command and a graph file");
    args.command = positional[0];
    args.graph = positional[1];

    if (args.command != "decompose" && args.command != "sparsifier" && args.command != "ncut") {
        throw UsageError("unknown command '" + args.command + "'");
    }
    if ((args.command == "ncut") != (args.k > 0)) {
        throw UsageError(args.command == "ncut" ? "ncut needs -k with at least 1 cluster" : "-k is only for ncut");
    }
    if (args.options.config.targetSparsity <= 0) throw UsageError("--sparsity must be positive");
    if (args.options.config.randomWalkThreshold <= 0) throw UsageError("--walk-threshold must be positive");
    return args;
}


// ---------------------------------------------------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------------------------------------------------

std::string jsonString(std::string_view text) {
    std::string out = "\"";
    for (const char c : text) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += c;
        } else if (static_cast<unsigned char>(c) < 0x20) {
            char escaped[8];
            std::snprintf(escaped, sizeof escaped, "\\u%04x", static_cast<unsigned>(c));
            out += escaped;
        } else {
            out += c;
        }
    }
    return out + "\"";
}

/**
 * The summary of a run: printed as "key: value" lines, or as one JSON object. A list (the levels of a hierarchy)
 * becomes one line per item, or a JSON array of objects.
 */
class Summary {
    struct Field {
        std::string key;
        std::string value;          // as printed
        bool isString = false;      // quoted in JSON
        std::vector<Summary> list;  // if not empty, the field is a list of items
    };
    std::vector<Field> fields;

public:
    void add(std::string key, std::string_view text) {
        fields.push_back({std::move(key), std::string(text), true, {}});
    }

    template<class T> requires std::is_arithmetic_v<T>
    void add(std::string key, T value) {
        fields.push_back({std::move(key), number(value), false, {}});
    }

    void addList(std::string key, std::vector<Summary> items) {
        fields.push_back({std::move(key), "", false, std::move(items)});
    }

    void printText(std::ostream& out) const {
        for (const Field& field : fields) {
            if (field.list.empty()) {
                out << field.key << ": " << field.value << "\n";
                continue;
            }
            for (size_t i = 0; i < field.list.size(); ++i) {
                out << field.key << " " << i + 1 << ":";
                const char* separator = " ";
                for (const Field& item : field.list[i].fields) {
                    out << separator << item.key << " " << item.value;
                    separator = ", ";
                }
                out << "\n";
            }
        }
    }

    void printJson(std::ostream& out) const {
        out << "{";
        const char* separator = "";
        for (const Field& field : fields) {
            out << separator << jsonString(field.key) << ":";
            separator = ",";
            if (!field.list.empty()) {
                out << "[";
                for (size_t i = 0; i < field.list.size(); ++i) {
                    if (i > 0) out << ",";
                    field.list[i].printJson(out);
                }
                out << "]";
            } else {
                out << (field.isString ? jsonString(field.value) : field.value);
            }
        }
        out << "}";
    }
};

// Milliseconds, to the microsecond
double milliseconds(double seconds) {
    return std::round(seconds * 1e6) / 1000;
}

// Writes one line per node, given by writeLine(out, node)
template<class WriteLine>
void writeNodeLines(std::ostream& out, size_t numNodes, WriteLine writeLine) {
    for (NodeIx v = 0; v < numNodes; ++v) {
        writeLine(out, v);
        out << "\n";
    }
    out.flush();
    if (!out) throw std::runtime_error("cannot write the output file");
}

template<class EdgeWeight>
void writeClusters(std::ostream& out, const deco::Clustering<EdgeWeight>& clustering) {
    writeNodeLines(out, clustering.clusterOf.size(), [&](std::ostream& line, NodeIx v) {
        line << clustering.clusterOf[v];
    });
}

template<class EdgeWeight>
void addClustering(Summary& summary, const deco::Clustering<EdgeWeight>& clustering) {
    summary.add("numClusters", clustering.numClusters);
    summary.add("cutEdges", clustering.cutEdges);
    summary.add("cutWeight", clustering.cutWeight);
}

const char* backendName(deco::Backend backend) {
    switch (backend) {
        case deco::Backend::Cpu: return "cpu";
        case deco::Backend::Cuda: return "cuda";
        default: return "auto";
    }
}

template<class GraphT>
const char* weightKind() {
    if constexpr (std::same_as<GraphT, CsrGraph<>>) return "unweighted";
    else if constexpr (std::floating_point<typename GraphT::EdgeWeight>) return "real";
    else return "integer";
}


// ---------------------------------------------------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------------------------------------------------

// Each command runs on `graph`, adds its results to `summary`, and writes the clusters to `output` (if not null)

template<MutableGraph GraphT>
void decompose(const Args& args, GraphT& graph, Summary& summary, std::ostream* output) {
    const auto result = deco::expanderDecomposition(graph, args.options);

    summary.add("backend", backendName(result.backend));
    summary.add("iterations", result.stats.iterations);
    summary.add("time", milliseconds(result.stats.seconds));
    addClustering(summary, result);

    if (output) writeClusters(*output, result);
}

template<MutableGraph GraphT>
void sparsifier(const Args& args, GraphT& graph, Summary& summary, std::ostream* output) {
    summary.add("backend", backendName(deco::chooseBackend(graph, args.options.backend)));

    Timer timer;
    timer.start();
    auto result = deco::expanderHierarchy(graph, args.options);
    summary.add("time", milliseconds(timer.timeSeconds()));
    summary.add("levels", result.height());

    std::vector<Summary> levels;
    for (NodeIx l = 0; l < result.height(); ++l) {
        const auto& level = result.lvl(l);
        typename GraphT::EdgeWeight cutWeight{};   // of the clusters on this level: the weight of the contracted graph
        for (const auto w : level.graph.edgeWeights()) cutWeight += w;

        Summary item;
        item.add("clusters", level.graph.numNodes);
        item.add("edges", level.graph.numEdges);
        item.add("cutWeight", cutWeight / 2);   // every edge is stored twice
        item.add("time", milliseconds(level.seconds));
        levels.push_back(std::move(item));
    }
    summary.addList("level", std::move(levels));

    if (output) {
        writeNodeLines(*output, graph.numNodes, [&](std::ostream& line, NodeIx v) {
            NodeIx node = v;
            for (NodeIx l = 0; l < result.height(); ++l) {
                node = result.lvl(l).parents[node];
                line << (l > 0 ? " " : "") << node;
            }
        });
    }
}

template<MutableGraph GraphT>
void ncut(const Args& args, GraphT& graph, Summary& summary, std::ostream* output) {
    summary.add("backend", backendName(deco::chooseBackend(graph, args.options.backend)));
    summary.add("k", args.k);

    Timer timer;
    timer.start();
    const auto result = deco::normalizedCut(graph, args.k, args.options);
    summary.add("time", milliseconds(timer.timeSeconds()));
    addClustering(summary, result);
    summary.add("normalizedCut", result.normalizedCut);

    if (output) writeClusters(*output, result);
}

template<MutableGraph GraphT>
void run(const Args& args, GraphT& graph) {
    std::ofstream outputFile;   // opened before the computation, so that a wrong path fails right away
    if (!args.output.empty()) {
        outputFile.open(args.output);
        if (!outputFile) throw std::runtime_error("cannot open " + args.output + " for writing");
    }
    std::ostream* output = args.output.empty() ? nullptr : &outputFile;

    Summary summary;
    summary.add("command", args.command);
    summary.add("graph", args.graph);
    summary.add("nodes", graph.numNodes);
    summary.add("edges", graph.numEdges);
    summary.add("weights", weightKind<GraphT>());

    if (args.command == "decompose") decompose(args, graph, summary, output);
    else if (args.command == "sparsifier") sparsifier(args, graph, summary, output);
    else ncut(args, graph, summary, output);

    if (args.json) {
        summary.printJson(std::cout);
        std::cout << std::endl;
    } else {
        summary.printText(std::cout);
    }
}

} // namespace


int main(int argc, char* argv[]) {
    Args args;
    try {
        args = parseArgs(argc, argv);
    } catch (const UsageError& e) {
        std::cerr << "error: " << e.what() << "\n\n" << usage();
        return 2;
    }
    if (args.help) {
        std::cout << usage();
        return 0;
    }

#ifndef NDEBUG
    std::cerr << "warning: this is a debug build, the times are not representative" << std::endl;
#endif

    try {
        AnyGraph graph = load_mtx_dynamic(args.graph);
        std::visit([&](auto& g) { run(args, g); }, graph);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
