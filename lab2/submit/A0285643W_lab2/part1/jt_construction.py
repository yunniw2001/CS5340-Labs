import numpy as np
import networkx as nx
from networkx.algorithms import tree
from factor import Factor
from factor_utils import factor_product


""" ADD HELPER FUNCTIONS HERE (IF NEEDED) """
def build_clique_graph_and_weight(jt_cliques):
    jt_cliques_idx = range(len(jt_cliques))
    G = nx.Graph()
    for tmp_clique_idx in jt_cliques_idx:
        # build nodes
        G.add_node(tmp_clique_idx)
    # check union
    for i in range(len(jt_cliques_idx)):
        for j in range(i+1,len(jt_cliques_idx)):
            # check intersect
            s_ij = np.intersect1d(jt_cliques[i],jt_cliques[j])
            # add weight if exist
            weight_ij = len(s_ij)
            if weight_ij>0:
                G.add_edge(i,j,weight = weight_ij)
    return G

""" END ADD HELPER FUNCTIONS HERE """


def _get_clique_factors(jt_cliques, factors):
    """
    Assign node factors to cliques in the junction tree and derive the clique factors.

    Args:
        jt_cliques: list of junction tree maximal cliques e.g. [[x1, x2, x3], [x2, x3], ... ]
        factors: list of factors from the original graph

    Returns:
        list of clique factors where the factor(jt_cliques[i]) = clique_factors[i]
    """
    clique_factors = [Factor() for _ in jt_cliques]

    """ YOUR CODE HERE """
    for idx in range(len(jt_cliques)):
        cur_clique = jt_cliques[idx]
        clique_var = set(cur_clique)
        cur_factors = factors.copy()
        for tmp_factor in cur_factors:
            if set(tmp_factor.var).issubset(clique_var):
                clique_factors[idx] = factor_product(clique_factors[idx],tmp_factor)
                factors.remove(tmp_factor)
    """ END YOUR CODE HERE """

    assert len(clique_factors) == len(jt_cliques), 'there should be equal number of cliques and clique factors'
    return clique_factors


def _get_jt_clique_and_edges(nodes, edges):
    """
    Construct the structure of the junction tree and return the list of cliques (nodes) in the junction tree and
    the list of edges between cliques in the junction tree. [i, j] in jt_edges means that cliques[j] is a neighbor
    of cliques[i] and vice versa. [j, i] should also be included in the numpy array of edges if [i, j] is present.
    You can use nx.Graph() and nx.find_cliques().

    Args:
        nodes: numpy array of nodes [x1, ..., xN]
        edges: numpy array of edges e.g. [x1, x2] implies that x1 and x2 are neighbors.

    Returns:
        list of junction tree cliques. each clique should be a maximal clique. e.g. [[X1, X2], ...]
        numpy array of junction tree edges e.g. [[0,1], ...], [i,j] means that cliques[i]
            and cliques[j] are neighbors.
    """
    jt_cliques = []
    jt_edges = np.array(edges)  # dummy value

    """ YOUR CODE HERE """
    new_graph = nx.Graph()
    new_graph.add_nodes_from(nodes)
    new_graph.add_edges_from(edges)

    # get cliques
    jt_cliques = list(nx.find_cliques(new_graph))

    # find the maximum spanning tree
    clique_graph_with_weight = build_clique_graph_and_weight(jt_cliques)

    # maximum_spanning_tree
    maximum_spanning_tree = nx.maximum_spanning_tree(clique_graph_with_weight)

    # get edge
    jt_edges = [list(edge) for edge in maximum_spanning_tree.edges()]

    """ END YOUR CODE HERE """

    return jt_cliques, jt_edges


def construct_junction_tree(nodes, edges, factors):
    """
    Constructs the junction tree and returns its the cliques, edges and clique factors in the junction tree.
    DO NOT EDIT THIS FUNCTION.

    Args:
        nodes: numpy array of random variables e.g. [X1, X2, ..., Xv]
        edges: numpy array of edges e.g. [[X1,X2], [X2,X1], ...]
        factors: list of factors in the graph

    Returns:
        list of cliques e.g. [[X1, X2], ...]
        numpy array of edges e.g. [[0,1], ...], [i,j] means that cliques[i] and cliques[j] are neighbors.
        list of clique factors where jt_cliques[i] has factor jt_factors[i] where i is an index
    """
    jt_cliques, jt_edges = _get_jt_clique_and_edges(nodes=nodes, edges=edges)
    jt_factors = _get_clique_factors(jt_cliques=jt_cliques, factors=factors)
    return jt_cliques, jt_edges, jt_factors
