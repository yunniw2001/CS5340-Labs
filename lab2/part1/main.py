""" CS5340 Lab 2 Part 1: Junction Tree Algorithm
See accompanying PDF for instructions.

Name: WU YIHANG
Email: e1216288@u.nus.edu
Student ID: A0285643W
"""

import os
import numpy as np
import json
import networkx as nx
from argparse import ArgumentParser

from factor import Factor
from jt_construction import construct_junction_tree
from factor_utils import factor_product, factor_evidence, factor_marginalize

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
INPUT_DIR = os.path.join(DATA_DIR, 'inputs')  # we will store the input data files here!
PREDICTION_DIR = os.path.join(DATA_DIR, 'predictions')  # we will store the prediction files here!


""" ADD HELPER FUNCTIONS HERE """
def collect(g,i,j,msg,cliques):
    N_j = np.array(list(g.neighbors(j)))
    N_j_no_i =np.setdiff1d(N_j,np.array([i]))
    for k in N_j_no_i:
        collect(g,j,k,msg,cliques)
    sendmessage(g,j,i,msg,cliques)
    return

def sendmessage(g,j,i,msg,cliques):
    N_j = np.array(list(g.neighbors(j)))
    N_j_no_i =np.setdiff1d(N_j,np.array([i]))
    # calculate product
    m_ki = Factor()
    if len(N_j_no_i) == 0:
        m_ki = g.nodes[j]['factor']
    else:
        for k in range(len(N_j_no_i)):
            k_var = N_j_no_i[k]
            if k == 0:
                m_ki = msg[k_var][j]
            else:
                m_ki = factor_product(m_ki,msg[k_var][j])
        m_ki = factor_product(g.nodes[j]['factor'],m_ki)
    # calculate sum
    margin_vars = np.setdiff1d(cliques[j],cliques[i])
    msg[j][i] = factor_marginalize(m_ki,margin_vars)
    return msg[j][i]

def distribute(g,i,j,msg,cliques):
    sendmessage(g,i,j,msg,cliques)
    N_j = np.array(list(g.neighbors(j)))
    N_j_no_i =np.setdiff1d(N_j,np.array([i]))

    for k in N_j_no_i:
        distribute(g,j,k,msg,cliques)

def computeMarginal(g,i,msg):
    n_i = np.array(list(g.neighbors(i)))
    product = Factor()
    for j in range(len(n_i)):
        if j == 0:
            product = msg[n_i[j]][i]
        else:
            product = factor_product(product,msg[n_i[j]][i])
    if g.nodes[i] != {}:
        product = factor_product(product, g.nodes[i]['factor'])
    product_sum = np.sum(product.val)
    product.val = product.val/product_sum
    return product



""" END HELPER FUNCTIONS HERE """


def _update_mrf_w_evidence(all_nodes, evidence, edges, factors):
    """
    Update the MRF graph structure from observing the evidence

    Args:
        all_nodes: numpy array of nodes in the MRF
        evidence: dictionary of node:observation pairs where evidence[x1] returns the observed value of x1
        edges: numpy array of edges in the MRF
        factors: list of Factors in teh MRF

    Returns:
        numpy array of query nodes
        numpy array of updated edges (after observing evidence)
        list of Factors (after observing evidence; empty factors should be removed)
    """

    query_nodes = all_nodes
    updated_edges = edges
    updated_factors = factors

    """ YOUR CODE HERE """
    factor_after_evidence = []
    for tmp_factor in factors:
        factor_after_evidence.append(factor_evidence(tmp_factor,evidence))
    updated_factors = factor_after_evidence

    # remove evidence vars aka get query nodes
    observe_vars = evidence.keys()
    query_nodes = np.setdiff1d(query_nodes,list(observe_vars))

    # remove edges aka get updated edges
    updated_edges = []
    for edge in edges:
        if edge[0] in observe_vars or edge[1] in observe_vars:
            continue
        else:
            updated_edges.append(edge)

    """ END YOUR CODE HERE """

    return query_nodes, updated_edges, updated_factors


def _get_clique_potentials(jt_cliques, jt_edges, jt_clique_factors):
    """
    Returns the list of clique potentials after performing the sum-product algorithm on the junction tree

    Args:
        jt_cliques: list of junction tree nodes e.g. [[x1, x2], ...]
        jt_edges: numpy array of junction tree edges e.g. [i,j] implies that jt_cliques[i] and jt_cliques[j] are
                neighbors
        jt_clique_factors: list of clique factors where jt_clique_factors[i] is the factor for cliques[i]

    Returns:
        list of clique potentials computed from the sum-product algorithm
    """
    clique_potentials = jt_clique_factors

    """ YOUR CODE HERE """
    root = 0

    # Create structure to hold messages
    num_nodes = len(jt_cliques)
    messages = [[None] * num_nodes for _ in range(num_nodes)]

    graph = nx.Graph()
    for idx,clique in enumerate(jt_cliques):
        graph.add_node(idx,factor = jt_clique_factors[idx])
    graph.add_edges_from(jt_edges)
    root = 0
    n_f = graph.neighbors(root)
    # print(list(n_f))
    for e in n_f:
        collect(graph, root, e, messages,jt_cliques)
    n_f = graph.neighbors(root)
    # print(list(n_f))
    for e in n_f:
        distribute(graph, root, e, messages,jt_cliques)
    # print(messages[:])
    # print(V)
    clique_potentials = []
    for i in range(len(jt_cliques)):
        res = computeMarginal(graph, i, messages)
        clique_potentials.append(res)

    """ END YOUR CODE HERE """

    assert len(clique_potentials) == len(jt_cliques)
    return clique_potentials


def _get_node_marginal_probabilities(query_nodes, cliques, clique_potentials):
    """
    Returns the marginal probability for each query node from the clique potentials.

    Args:
        query_nodes: numpy array of query nodes e.g. [x1, x2, ..., xN]
        cliques: list of cliques e.g. [[x1, x2], ... [x2, x3, .., xN]]
        clique_potentials: list of clique potentials (Factor class)

    Returns:
        list of node marginal probabilities (Factor class)

    """
    query_marginal_probabilities = []

    """ YOUR CODE HERE """
    marginals_table = {}
    query_marginal_dict = {}

    related_clique_for_query_node={}
    for node in query_nodes:
        # find smallest cliques related to query nodes
        smallest_clique_idx = 2**32-1
        smallest_clique_length = 2**32-1
        for idx,clique in enumerate(cliques):
            if node in clique and len(clique)<smallest_clique_length:
                smallest_clique_idx = idx
        smallest_clique = cliques[smallest_clique_idx]
        smallest_potential = clique_potentials[smallest_clique_idx]
        vars_need_marginalize = np.setdiff1d(smallest_potential.var,[node])
        query_marginal_probabilities.append(factor_marginalize(smallest_potential,vars_need_marginalize))
    """ END YOUR CODE HERE """

    return query_marginal_probabilities


def get_conditional_probabilities(all_nodes, evidence, edges, factors):
    """
    Returns query nodes and query Factors representing the conditional probability of each query node
    given the evidence e.g. p(xf|Xe) where xf is a single query node and Xe is the set of evidence nodes.

    Args:
        all_nodes: numpy array of all nodes (random variables) in the graph
        evidence: dictionary of node:evidence pairs e.g. evidence[x1] returns the observed value for x1
        edges: numpy array of all edges in the graph e.g. [[x1, x2],...] implies that x1 is a neighbor of x2
        factors: list of factors in the MRF.

    Returns:
        numpy array of query nodes
        list of Factor
    """
    query_nodes, updated_edges, updated_node_factors = _update_mrf_w_evidence(all_nodes=all_nodes, evidence=evidence,
                                                                              edges=edges, factors=factors)

    jt_cliques, jt_edges, jt_factors = construct_junction_tree(nodes=query_nodes, edges=updated_edges,
                                                               factors=updated_node_factors)

    clique_potentials = _get_clique_potentials(jt_cliques=jt_cliques, jt_edges=jt_edges, jt_clique_factors=jt_factors)

    query_node_marginals = _get_node_marginal_probabilities(query_nodes=query_nodes, cliques=jt_cliques,
                                                            clique_potentials=clique_potentials)

    return query_nodes, query_node_marginals


def parse_input_file(input_file: str):
    """ Reads the input file and parses it. DO NOT EDIT THIS FUNCTION. """
    with open(input_file, 'r') as f:
        input_config = json.load(f)

    nodes = np.array(input_config['nodes'])
    edges = np.array(input_config['edges'])

    # parse evidence
    raw_evidence = input_config['evidence']
    evidence = {}
    for k, v in raw_evidence.items():
        evidence[int(k)] = v

    # parse factors
    raw_factors = input_config['factors']
    factors = []
    for raw_factor in raw_factors:
        factor = Factor(var=np.array(raw_factor['var']), card=np.array(raw_factor['card']),
                        val=np.array(raw_factor['val']))
        factors.append(factor)
    return nodes, edges, evidence, factors


def main():
    """ Entry function to handle loading inputs and saving outputs. DO NOT EDIT THIS FUNCTION. """
    argparser = ArgumentParser()
    argparser.add_argument('--case', type=int, required=True,
                           help='case number to create observations e.g. 1 if 1.json')
    args = argparser.parse_args()

    case = args.case
    input_file = os.path.join(INPUT_DIR, '{}.json'.format(case))
    nodes, edges, evidence, factors = parse_input_file(input_file=input_file)

    # solution part:
    query_nodes, query_conditional_probabilities = get_conditional_probabilities(all_nodes=nodes, edges=edges,
                                                                                 factors=factors, evidence=evidence)

    predictions = {}
    for i, node in enumerate(query_nodes):
        probability = query_conditional_probabilities[i].val
        predictions[int(node)] = list(np.array(probability, dtype=float))

    if not os.path.exists(PREDICTION_DIR):
        os.makedirs(PREDICTION_DIR)
    prediction_file = os.path.join(PREDICTION_DIR, '{}.json'.format(case))
    with open(prediction_file, 'w') as f:
        json.dump(predictions, f, indent=1)
    print('INFO: Results for test case {} are stored in {}'.format(case, prediction_file))


if __name__ == '__main__':
    main()
