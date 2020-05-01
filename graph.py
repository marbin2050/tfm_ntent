__author__ = '{Alfonso Aguado Bustillo}'

from itertools import combinations
import networkx as nx


def create_graph(docs_by_query):

    # create the set of edges -- ideally you would dump this to a file rather than keep it in memory
    edges = set()
    for documents in docs_by_query.values():
        edges.update(set(combinations(documents, 2)))

    # create graph
    graph = nx.Graph()

    # add nodes
    for doc_id in docs_by_query.keys():
        graph.add_node(doc_id)

    # add edges
    for edge in edges:
        graph.add_edge(edge[0], edge[1])

    # list(map(lambda edge: graph.add_edge(edge[0], edge[1]), edges))

    # nx.draw(graph, with_labels=True, font_weight='bold')
    # plt.show()

    # return graph