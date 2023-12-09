""" CS5340 Lab 4 Part 1: Importance Sampling
See accompanying PDF for instructions.

Name: WU YIHANG
Email: e1216288@u.nus.edu
Student ID: A0285643W
"""

import os
import json
import numpy as np
import networkx as nx
from factor_utils import factor_evidence, factor_product, assignment_to_index
from factor import Factor
from argparse import ArgumentParser
# from tqdm import tqdm

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
INPUT_DIR = os.path.join(DATA_DIR, 'inputs')
PREDICTION_DIR = os.path.join(DATA_DIR, 'predictions')

""" ADD HELPER FUNCTIONS HERE """


def find_order(proposal_factors):
    G = nx.DiGraph()
    for index, factor in proposal_factors.items():
        vars = factor.var
        if len(vars) > 1:
            child = vars[-1]
            parents = vars[:-1]
            G.add_edges_from((parent, child) for parent in parents)
        else:
            G.add_node(vars[0])
    nodes = list(nx.topological_sort(G))
    return nodes

def get_probability_from_factor(factor, sample):
    assignment = [sample[var] for var in factor.var]
    index = assignment_to_index(assignment, factor.card)
    return factor.val[index]

def find_factor_for_node(node,factors):
    for item in factors:
        if node == factors[item].var[-1]:
            return factors[item]

def calculate_weight(sample, evidence,target_factors_pos,proposal_factors_pos):
    weight = 1.0

    for node, value in sample.items():
        target_factor = target_factors_pos[node]
        proposal_factor = proposal_factors_pos[node]
        tmp = sample.copy()
        tmp.update(evidence)
        target_prob = get_probability_from_factor(target_factor, tmp)
        proposal_prob = get_probability_from_factor(proposal_factor, sample)

        if proposal_prob == 0:
            return 0

        weight *= target_prob / proposal_prob

    return weight

""" END HELPER FUNCTIONS HERE """


def _sample_step(nodes, proposal_factors):
    """
    Performs one iteration of importance sampling where it should sample a sample for each node. The sampling should
    be done in topological order.

    Args:
        nodes: numpy array of nodes. nodes are sampled in the order specified in nodes
        proposal_factors: dictionary of proposal factors where proposal_factors[1] returns the
                sample distribution for node 1

    Returns:
        dictionary of node samples where samples[1] return the scalar sample for node 1.
    """
    samples = {}

    """ YOUR CODE HERE: Use np.random.choice """
    sve = proposal_factors.copy()
    for node in nodes:
        samples[node] = np.random.choice(a=np.arange(proposal_factors[node].card[0]), p=proposal_factors[node].val)
        # evidence by parent node
        for factor in proposal_factors:
            proposal_factors[factor] = factor_evidence(proposal_factors[factor], samples)
    proposal_factors = sve
    """ END YOUR CODE HERE """

    assert len(samples.keys()) == len(nodes)
    return samples


def _get_conditional_probability(target_factors, proposal_factors, evidence, num_iterations):
    """
    Performs multiple iterations of importance sampling and returns the conditional distribution p(Xf | Xe) where
    Xe are the evidence nodes and Xf are the query nodes (unobserved).

    Args:
        target_factors: dictionary of node:Factor pair where Factor is the target distribution of the node.
                        Other nodes in the Factor are parent nodes of the node. The product of the target
                        distribution gives our joint target distribution.
        proposal_factors: dictionary of node:Factor pair where Factor is the proposal distribution to sample node
                        observations. Other nodes in the Factor are parent nodes of the node
        evidence: dictionary of node:val pair where node is an evidence node while val is the evidence for the node.
        num_iterations: number of importance sampling iterations

    Returns:
        Approximate conditional distribution of p(Xf | Xe) where Xf is the set of query nodes (not observed) and
        Xe is the set of evidence nodes. Return result as a Factor
    """
    out = Factor()

    """ YOUR CODE HERE """
    samples = []
    weights = []
    # reorder into topological order and remove evidence
    topo_order_nodes = find_order(proposal_factors)
    evidence_nodes = list(evidence.keys())
    nodes_no_evidence = [node for node in topo_order_nodes if node not in evidence_nodes]
    weights_ap = {}
    # evidence
    for item in proposal_factors:
        proposal_factors[item] = factor_evidence(proposal_factors[item],evidence)

    # prepare some dict to save time
    target_factors_pos = {}
    proposal_factors_pos = {}
    for node in nodes_no_evidence:
        target_factors_pos[node] = find_factor_for_node(node,target_factors)
        proposal_factors_pos[node] = find_factor_for_node(node,proposal_factors)

    weight_detail = {}
    for _ in range(num_iterations):
        sve = proposal_factors.copy()
        sample = _sample_step(nodes_no_evidence,proposal_factors)
        sample_res = sorted(sample.items(), key=lambda d: d[0])
        sample= {key: value for key, value in sample_res}
        sample_assignment = [value for key,value in sample_res]
        proposal_factors = sve

        if tuple(sample.items()) in weight_detail:
            weight = weight_detail[tuple(sample.items())]
            # print('1')
        else:
            weight = calculate_weight(sample,evidence,target_factors_pos,proposal_factors_pos)
            weight_detail[tuple(sample.items())] = weight
        # weight = 1
        # for node in nodes_no_evidence:
        #     # Calculate weight as ratio of target to proposal probabilities for the sampled value
        #
        #     idx = assignment_to_index([sample[node]], target_factors[node].card)
        #     target_prob = target_factors[node].val[idx] if idx < len(target_factors[node].val) else 0
        #     proposal_factors[node] = factor_evidence(proposal_factors[node],evidence)
        #     proposal_prob = proposal_factors[node].val[idx] if idx < len(proposal_factors[node].val) else 0
        #     weight *= (target_prob / proposal_prob) if proposal_prob > 0 else 0

        samples.append(sample)
        weights.append(weight)
        # record appearance
        if not tuple(sample.items()) in weights_ap:
            weights_ap[tuple(sample.items())] = 0
        weights_ap[tuple(sample.items())]+=weight

    denominator = sum(weights_ap.values())
    for item in weights_ap:
        weights_ap[item]/=denominator

    upper_sorted_nodes = sorted(nodes_no_evidence)
    out.var = np.array([node for node in upper_sorted_nodes])
    out.card = []
    for node in upper_sorted_nodes:
        factor = find_factor_for_node(node,target_factors)
        out.card.append(factor.card[-1])
    out.card = np.array(out.card)
    num_assignments = int(np.prod(out.card))
    out.val = np.zeros(num_assignments)

    for key,value in weights_ap.items():
        sample = dict(key)
        assignment = [sample[node] for node in upper_sorted_nodes]
        idx = assignment_to_index([assignment], out.card)
        out.val[idx] = value


    """ END YOUR CODE HERE """

    return out


def load_input_file(input_file: str) -> (Factor, dict, dict, int):
    """
    Returns the target factor, proposal factors for each node and evidence. DO NOT EDIT THIS FUNCTION

    Args:
        input_file: input file to open

    Returns:
        Factor of the target factor which is the target joint distribution of all nodes in the Bayesian network
        dictionary of node:Factor pair where Factor is the proposal distribution to sample node observations. Other
                    nodes in the Factor are parent nodes of the node
        dictionary of node:val pair where node is an evidence node while val is the evidence for the node.
    """
    with open(input_file, 'r') as f:
        input_config = json.load(f)
    target_factors_dict = input_config['target-factors']
    proposal_factors_dict = input_config['proposal-factors']
    assert isinstance(target_factors_dict, dict) and isinstance(proposal_factors_dict, dict)

    def parse_factor_dict(factor_dict):
        var = np.array(factor_dict['var'])
        card = np.array(factor_dict['card'])
        val = np.array(factor_dict['val'])
        return Factor(var=var, card=card, val=val)

    target_factors = {int(node): parse_factor_dict(factor_dict=target_factor) for
                      node, target_factor in target_factors_dict.items()}
    proposal_factors = {int(node): parse_factor_dict(factor_dict=proposal_factor_dict) for
                        node, proposal_factor_dict in proposal_factors_dict.items()}
    evidence = input_config['evidence']
    evidence = {int(node): ev for node, ev in evidence.items()}
    num_iterations = input_config['num-iterations']
    return target_factors, proposal_factors, evidence, num_iterations


def main():
    """
    Helper function to load the observations, call your parameter learning function and save your results.
    DO NOT EDIT THIS FUNCTION.
    """
    argparser = ArgumentParser()
    argparser.add_argument('--case', type=int, required=True,
                           help='case number to create observations e.g. 1 if 1.json')
    args = argparser.parse_args()
    # np.random.seed(0)

    case = args.case
    input_file = os.path.join(INPUT_DIR, '{}.json'.format(case))
    target_factors, proposal_factors, evidence, num_iterations = load_input_file(input_file=input_file)

    # solution part
    conditional_probability = _get_conditional_probability(target_factors=target_factors,
                                                           proposal_factors=proposal_factors,
                                                           evidence=evidence, num_iterations=num_iterations)
    print(conditional_probability)
    # end solution part

    # json only recognises floats, not np.float, so we need to cast the values into floats.
    save__dict = {
        'var': np.array(conditional_probability.var).astype(int).tolist(),
        'card': np.array(conditional_probability.card).astype(int).tolist(),
        'val': np.array(conditional_probability.val).astype(float).tolist()
    }

    if not os.path.exists(PREDICTION_DIR):
        os.makedirs(PREDICTION_DIR)
    prediction_file = os.path.join(PREDICTION_DIR, '{}.json'.format(case))

    with open(prediction_file, 'w') as f:
        json.dump(save__dict, f, indent=1)
    print('INFO: Results for test case {} are stored in {}'.format(case, prediction_file))


if __name__ == '__main__':
    main()
