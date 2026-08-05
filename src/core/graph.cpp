#include "core/graph.h"

void DynamicGraph::removeEdge(NodeIx fx, NodeIx tx) {
    if(fx > tx) std::swap(fx, tx);
    auto eit = std::find(edges.begin(), edges.end(), Edge{fx, tx});
    std::swap(*eit, edges.back());
    popEdge();
}

Graph DynamicGraph::finalize() const {
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
    while(eix < fullEdges.size()) {
        assert(nix <= numNodes);
        rg.at(nix) = eix;
        while(eix < fullEdges.size() && fullEdges.at(eix).fx == nix) eix++;
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
