import networkx as nx


def create_directed_graph(adjacency_matrix, a_length, developer_names, last_activity_per_developer):
    graph = nx.DiGraph()
    graph.add_nodes_from(range(0, a_length))

    activity_map = {developer_names[k]: v for k, v in last_activity_per_developer.items()}

    nx.set_node_attributes(graph, 'active', activity_map)

    nr_topics = len(adjacency_matrix)

    for t1 in range(a_length):
        for t2 in range(a_length):
            sm1 = adjacency_matrix[t1].sum()
            if sm1 == 0:
                wgt = 0
            else:
                wgt = adjacency_matrix[t1, t2] / adjacency_matrix[t1].sum()

            param = {'weight': wgt, 'components': []}
            graph.add_edge(t1, t2, **param)

    all_edges = graph.edges()
    edge_occurrence = {}
    for t1 in range(0, a_length):
        edge_occurrence[t1] = 0

    for edge in all_edges:
        edge_occurrence[edge[0]] += 1  # outgoing direction
        edge_occurrence[edge[1]] += 1  # incoming direciton

    for n in range(0, a_length):
        if edge_occurrence[n] == 0:
            graph.remove_node(n)

    # getting weight   goal_g[0][14]

    node_sizes = []
    sm = 0.0
    for topic in range(nr_topics):
        sm += adjacency_matrix[topic][:].sum()

    for topic in range(nr_topics):
        for n in graph.nodes():
            incoming = adjacency_matrix[topic][n].sum()
            size = 300.0 + (float(incoming) / sm) * sm * 6
            node_sizes.append(size)

    glabels = {}
    relabel = {}
    for n in graph.nodes():
        lbl = [k for k, v in developer_names.items() if v == n][0]
        relabel[n] = lbl
        glabels[n] = lbl[:lbl.find('@')]

    # using the developer ids as labels is a lot more useful than integers
    graph = nx.relabel_nodes(graph, relabel)

    return graph, node_sizes, glabels
