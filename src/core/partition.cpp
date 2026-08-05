#include "core/partition.h"

Partition::Partition(Graph* gr) :
        clusters(1),
        clusterVertices(gr->numNodes),
        splitOracle(gr->numNodes),
        lookup(gr->numNodes),
        sc_seen(gr->numNodes),
        sc_removeAt(gr->numNodes),
        numCutEdges{0},
        cutSize{0},
        dfs(this, gr->numNodes),
        graph{gr}
//        sl(2 * gr.numEdges)
{
    EdgeIx volume{0};

    for(NodeIx nix = 0; nix < gr->numNodes; ++nix) {
        EdgeIx degree = gr->ranges[nix+1] - gr->ranges[nix];
        clusterVertices[nix] = {
                nix, gr->ranges[nix+1], degree
        };
        volume += gr->ranges[nix+1] - gr->ranges[nix];
        lookup[nix] = nix;
    }

    clusters[0] = {volume, volume, 0, gr->numNodes, this};

    std::vector<NodeIx> mod; // a bit ugly
    // consolidate(0, mod);
}

void Partition::consolidate(NodeIx clusterId, std::vector<NodeIx>& modified) {
    NodeIx subclusters = dfs.computeLabels(clusters[clusterId]);
    if(subclusters > 1) {
        INFO("Cluster consists of " << subclusters << " subclusters. Splitting.");
        splitByIndices<true, false>(clusterId, dfs.labels(), subclusters, modified);
    }
}

std::vector<ClusterVertex>::iterator Cluster::begin() const {
    return partition->clusterVertices.begin() + nodeBegin;
}

std::vector<ClusterVertex>::iterator Cluster::end() const {
    return partition->clusterVertices.begin() + nodeEnd;
}

std::vector<NodeIx>::iterator Cluster::edgeBegin(const ClusterVertex& cv) const {
    return partition->graph->nbegin(cv.nix);
}

std::vector<NodeIx>::iterator Cluster::edgeEnd(const ClusterVertex& cv) const {
    return partition->graph->edges.begin() + cv.end;
}

NodeIx DFS::computeLabels(const Cluster& cluster) {
    NodeIx numComponents = 0;
    NodeIx qix = 0;

    numCalls++;

    for (const ClusterVertex& cv: cluster) {
        if (m_visited[cv.nix] == numCalls) {
            continue;
        }

        m_stack[qix++] = cv.nix;
        m_visited[cv.nix] = numCalls;
        m_labels[cv.nix] = numComponents;

        // Run DFS starting from cv.nix
        while (qix) {
            NodeIx u = m_stack[--qix];
            const ClusterVertex& cvx = partition->vertexFor(u);

            for (auto it = cluster.edgeBegin(cvx); it != cluster.edgeEnd(cvx); ++it) {
                NodeIx nb = *it;

                if (m_visited[nb] < numCalls) {
                    m_visited[nb] = numCalls;
                    m_labels[nb] = numComponents;
                    m_stack[qix++] = nb;
                }
            }
        }

        numComponents++;
    }

    return numComponents;
}

void checkCluster(const Cluster& cluster, const Graph& graph) {
    EdgeIx sumDegrees = 0, summedVol = 0;
    assert(cluster.nodeBegin <= cluster.nodeEnd);
    for(const ClusterVertex& cv: cluster) {
        assert(cluster.edgeBegin(cv) == graph.nbegin(cv.nix));
        assert(cv.end == std::distance(graph.edges.begin(), graph.nbegin(cv.nix) + cv.internalDegree));
        assert(cv.end <= std::distance(graph.edges.begin(), graph.nend(cv.nix)));

        for(auto it = cluster.edgeBegin(cv); it != cluster.edgeEnd(cv); ++it) {
            NodeIx nb = *it;
            auto itx = std::find_if(graph.nbegin(cv.nix), graph.nend(cv.nix),
                                    [&nb](NodeIx otherNix) { return otherNix == nb; } );
            assert(itx != graph.nend(cv.nix));

            auto it2 = std::find_if(cluster.begin(), cluster.end(),
                                    [&nb](const ClusterVertex& node) { return node.nix == nb; } );
            assert(it2 != cluster.end());
        }

        assert(cv.internalDegree == std::distance(cluster.edgeBegin(cv), cluster.edgeEnd(cv)));
        sumDegrees += cv.internalDegree;
        summedVol += graph.degree(cv.nix);
    }
    assert(summedVol == cluster.volume);
    assert(sumDegrees == cluster.internalVolume);
}

frac_t compute_normalized_cut(const Partition& part) {
    frac_t normalized_cut = 0.0;

    for(const Cluster& cl: part) {
        if(cl.volume == 0) continue;
        normalized_cut += static_cast<frac_t>(cl.volume - cl.internalVolume) / cl.volume;
    }

    return normalized_cut;
}
