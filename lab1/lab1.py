""" CS5340 Lab 1: Belief Propagation and Maximal Probability
See accompanying PDF for instructions.

Name: WU YIHANG
Email: e1216288@u.nus.edu
Student ID: A0285643W
"""

import copy
from typing import List

import numpy as np

from factor import Factor, index_to_assignment, assignment_to_index, generate_graph_from_factors, \
    visualize_graph


"""For sum product message passing"""
def factor_product(A, B):
    """Compute product of two factors.

    Suppose A = phi(X_1, X_2), B = phi(X_2, X_3), the function should return
    phi(X_1, X_2, X_3)
    """
    if A.is_empty():
        return B
    if B.is_empty():
        return A

    # Create output factor. Variables should be the union between of the
    # variables contained in the two input factors
    out = Factor()
    out.var = np.union1d(A.var, B.var) # find the union, find all x_i

    # Compute mapping between the variable ordering between the two factors
    # and the output to set the cardinality
    out.card = np.zeros(len(out.var), np.int64)
    mapA = np.argmax(out.var[None, :] == A.var[:, None], axis=-1)
    # mapA is an index array, refers to A.var's index in out.var
    mapB = np.argmax(out.var[None, :] == B.var[:, None], axis=-1)
    out.card[mapA] = A.card
    out.card[mapB] = B.card

    # For each assignment in the output, compute which row of the input factors
    # it comes from
    out.val = np.zeros(np.prod(out.card))
    assignments = out.get_all_assignments()
    idxA = assignment_to_index(assignments[:, mapA], A.card)
    idxB = assignment_to_index(assignments[:, mapB], B.card)

    """ YOUR CODE HERE
    You should populate the .val field with the factor product
    Hint: The code for this function should be very short (~1 line). Try to
      understand what the above lines are doing, in order to implement
      subsequent parts.
    """
    out.val = A.val[idxA] * B.val[idxB]
    return out


def factor_marginalize(factor, var):
    """Sums over a list of variables.

    Args:
        factor (Factor): Input factor
        var (List): Variables to marginalize out

    Returns:
        out: Factor with variables in 'var' marginalized out.
    """
    out = Factor()

    """ YOUR CODE HERE
    Marginalize out the variables given in var
    """
    # print(factor.var)
    # print(var)
    out.var = np.setdiff1d(factor.var, var)
    # print(out.var)
    # print(factor.var)
    out.card = factor.card[~np.isin(factor.var, var)]
    # print(out.card)
    # print(factor.card)
    # out.card = factor.card[out.var]
    # print(out.card)
    var = np.array(var)
    # print(var[:, None])
    var_axis = tuple(np.where(~np.isin(factor.var, var))[0])
    merge_axis = tuple(np.where(np.isin(factor.var, var))[0])
    # print(merge_axis)

    # print(out.card)
    # merge_axis = tuple(np.where(factor.var!=var)[0])
    # var_axis = tuple(np.where(factor.var == var)[0])
    # print(merge_axis)
    # print(var_axis)
    # print(factor.val)
    out.val = np.zeros(np.prod(out.card))
    # print(out.val)
    all_assignments = factor.get_all_assignments()
    # print(all_assignments.shape)
    first_ap_row, indices = np.unique(all_assignments[:, var_axis], axis=0, return_inverse=True)
    # print(var_axis)
    # print(indices.shape)
    # print(factor.val.shape)
    # print(indices)
    out.val = np.bincount(indices, weights=factor.val)
    # print(all_assignments)
    # print(sums)
    # print(out.var)
    # print(out.card)
    # print(out.val)
    # result = np.column_stack((first_ap_row, out.val))
    # print(result)
    # print(out.var)
    # print(out.card)
    # print(out.val)
    return out


def observe_evidence(factors, evidence=None):
    """Modify a set of factors given some evidence

    Args:
        factors (List[Factor]): List of input factors
        evidence (Dict): Dictionary, where the keys are the observed variables
          and the values are the observed values.

    Returns:
        List of factors after observing evidence
    """
    if evidence is None:
        return factors
    out = copy.deepcopy(factors)

    """ YOUR CODE HERE
    Set the probabilities of assignments which are inconsistent with the
    evidence to zero.
    """
    # 找所有的assignment
    # 找出对应位置的索引
    # 把索引处概率修改为0.
    # print(out)
    for cur_factor in out:
        cur_all_assignment = cur_factor.get_all_assignments()
        observe_vars = np.array(list(evidence.keys()))
        observe_vals = np.array(list(evidence.values()))
        # print(observe_vars)
        # print(observe_vals)
        intersection = np.intersect1d(cur_factor.var, observe_vars)
        if not len(intersection)>0:
            # print('yes')
            continue

        # print(intersection)
        # print(observe_vars)
        # print(observe_vars == intersection)
        observe_vals = [evidence.get(k) for k in intersection]
        # print(observe_vals)
        # print(observe_vals)
        observe_vars = intersection
        observe_vars_idx =  np.argmax(cur_factor.var[None, :] == observe_vars[:, None], axis=-1)
        # print(observe_vars_idx)
        # print(cur_all_assignment[:,observe_vars_idx])
        # if observe_vars[0] == 8:
        #     print('yes')
        # print(cur_all_assignment[:,observe_vars_idx])
        # print(observe_vals)
        # print(np.all(cur_all_assignment[:,observe_vars_idx]==observe_vals,axis=1))
        unrelated_assignment_mask = ~np.all(cur_all_assignment[:,observe_vars_idx]==observe_vals,axis=1)
        # print(unrelated_assignment_mask)
        cur_factor.val[unrelated_assignment_mask] = 0.0
        # print(out)

    return out


"""For max sum meessage passing (for MAP)"""
def factor_sum(A, B):
    """Same as factor_product, but sums instead of multiplies
    """
    if A.is_empty():
        return B
    if B.is_empty():
        return A

    # Create output factor. Variables should be the union between of the
    # variables contained in the two input factors
    out = Factor()
    out.var = np.union1d(A.var, B.var)

    # Compute mapping between the variable ordering between the two factors
    # and the output to set the cardinality
    out.card = np.zeros(len(out.var), np.int64)
    mapA = np.argmax(out.var[None, :] == A.var[:, None], axis=-1)
    mapB = np.argmax(out.var[None, :] == B.var[:, None], axis=-1)
    out.card[mapA] = A.card
    out.card[mapB] = B.card

    # For each assignment in the output, compute which row of the input factors
    # it comes from
    out.val = np.zeros(np.prod(out.card))
    assignments = out.get_all_assignments()
    idxA = assignment_to_index(assignments[:, mapA], A.card)
    idxB = assignment_to_index(assignments[:, mapB], B.card)

    """ YOUR CODE HERE
    You should populate the .val field with the factor sum. The code for this
    should be very similar to the factor_product().
    """
    out.val = A.val[idxA] + B.val[idxB]
    return out


def factor_max_marginalize(factor, var):
    """Marginalize over a list of variables by taking the max.

    Args:
        factor (Factor): Input factor
        var (List): Variable to marginalize out.

    Returns:
        out: Factor with variables in 'var' marginalized out. The factor's
          .val_argmax field should be a list of dictionary that keep track
          of the maximizing values of the marginalized variables.
          e.g. when out.val_argmax[i][j] = k, this means that
            when assignments of out is index_to_assignment[i],
            variable j has a maximizing value of k.
          See test_lab1.py::test_factor_max_marginalize() for an example.
    """
    out = Factor()

    """ YOUR CODE HERE
    Marginalize out the variables given in var. 
    You should make use of val_argmax to keep track of the location with the
    maximum probability.
    """
    out.var = np.setdiff1d(factor.var, var)
    out.card = factor.card[~np.isin(factor.var, var)]
    out.val_argmax = []
    out.val = np.zeros(np.product(out.card))
    marginalize_axis = list(np.where(np.isin(factor.var, var))[0])
    out_axis = list(np.where(~np.isin(factor.var, var))[0])
    factor_assignments = factor.get_all_assignments()
    out_assignment, out_assignment_axis, inverse_idx = np.unique(factor_assignments[:, out_axis], axis=0,
                                                                 return_index=True, return_inverse=True)
    for idx in range(len(out_assignment)):
        # print(idx)
        same_assignment_idx = np.where(inverse_idx == idx)[0]
        max_prob = factor.val[same_assignment_idx].max()
        max_prob_in_same_assignment_idx = same_assignment_idx[np.argmax(factor.val[same_assignment_idx])]
        # print(out.val)
        out.val[idx] = max_prob
        out.val_argmax.append(dict(zip(var, factor_assignments[max_prob_in_same_assignment_idx][marginalize_axis])))
    # print(out)
    # print(out.val_argmax)
    return out


def compute_joint_distribution(factors):
    """Computes the joint distribution defined by a list of given factors

    Args:
        factors (List[Factor]): List of factors

    Returns:
        Factor containing the joint distribution of the input factor list
    """
    joint = Factor()

    """ YOUR CODE HERE
    Compute the joint distribution from the list of factors. You may assume
    that the input factors are valid so no input checking is required.
    """
    for i in range(len(factors)):
        # print(factors[i])
        if i == 0:
            joint = copy.deepcopy(factors[i])
        else:
            joint = factor_product(joint, factors[i])
        # print(joint)

    return joint


def compute_marginals_naive(V, factors, evidence):
    """Computes the marginal over a set of given variables

    Args:
        V (int): Single Variable to perform inference on
        factors (List[Factor]): List of factors representing the graphical model
        evidence (Dict): Observed evidence. evidence[k] = v indicates that
          variable k has the value v.

    Returns:
        Factor representing the marginals
    """

    output = Factor()

    """ YOUR CODE HERE
    Compute the marginal. Output should be a factor.
    Remember to normalize the probabilities!
    """
    # Compute the joint distribution over all variables
    joint_distribution = compute_joint_distribution(factors)
    # print(joint_distribution)
    # Reduce the joint distribution by the evidence
    reduce_after_distribution = observe_evidence([joint_distribution], evidence)[0]
    # print(reduce_after_distribution)
    # Marginalizing out irrelevant variables
    observe_vars = np.array(list(evidence.keys()))
    var_need_to_marginalize = np.setdiff1d(joint_distribution.var, [V])
    # var_need_to_marginalize = np.setdiff1d(var_need_to_marginalize,observe_vars)
    # print(var_need_to_marginalize)
    distribution_after_reduce = factor_marginalize(reduce_after_distribution, var_need_to_marginalize)
    # print(distribution_after_reduce)
    # normalize the final probability distribution
    distribution_after_reduce.val = distribution_after_reduce.val / np.sum(distribution_after_reduce.val)
    output = distribution_after_reduce
    output.var = np.array([V])
    # print(output.var)
    # print(output.card)
    # print(output)
    return output

def collect(g,i,j,msg):
    N_j = np.array(list(g.neighbors(j)))
    N_j_no_i =np.setdiff1d(N_j,np.array([i]))
    for k in N_j_no_i:
        collect(g,j,k,msg)
    sendmessage(g,j,i,msg)
    return

def sendmessage(g,j,i,msg):
    N_j = np.array(list(g.neighbors(j)))
    # print()
    N_j_no_i =np.setdiff1d(N_j,np.array([i]))
    # print(N_j_no_i)
    # calculate product
    m_ki = None
    if len(N_j_no_i) == 0:
        m_ki = g.edges[i, j]['factor']
        # print(m_ki)
    else:
        for k in range(len(N_j_no_i)):
            k_var = N_j_no_i[k]
            # print(k_var)
            if k == 0:
                m_ki = msg[k_var][j]
            else:
                m_ki = factor_product(m_ki,msg[k_var][j])
        m_ki = factor_product(g.edges[i, j]['factor'],m_ki)
    # calculate sum
    # for root when disribute
    if g.nodes[j] != {}:
        # print('yes')
        m_ki = factor_product(m_ki, g.nodes[j]['factor'])
    msg[j][i] = factor_marginalize(m_ki,[j])
    # print(msg[j][i])
    return msg[j][i]

def distribute(g,i,j,msg):
    sendmessage(g,i,j,msg)

    # N_j = g.neighbors(j)
    # N_j_no_i =np.setdiff1d(N_j,g.nodes[i])
    N_j = np.array(list(g.neighbors(j)))
    # print()
    N_j_no_i =np.setdiff1d(N_j,np.array([i]))

    for k in N_j_no_i:
        distribute(g,j,k,msg)
def computeMarginal(g,i,msg):
    n_i = np.array(list(g.neighbors(i)))
    product = None
    for j in range(len(n_i)):
        if j == 0:
            product = msg[n_i[j]][i]
            if g.nodes[i] != {}:
                # print('yes')
                product= factor_product(product, g.nodes[i]['factor'])
        else:
            product = factor_product(product,msg[n_i[j]][i])
    product_sum = np.sum(product.val)
    product.val = product.val/product_sum
    return product


def compute_marginals_bp(V, factors, evidence):
    """Compute single node marginals for multiple variables
    using sum-product belief propagation algorithm

    Args:
        V (List): Variables to infer single node marginals for
        factors (List[Factor]): List of factors representing the grpahical model
        evidence (Dict): Observed evidence. evidence[k]=v denotes that the
          variable k is assigned to value v.

    Returns:
        marginals: List of factors. The ordering of the factors should follow
          that of V, i.e. marginals[i] should be the factor for variable V[i].
    """
    # Dummy outputs, you should overwrite this with the correct factors
    marginals = []

    # Setting up messages which will be passed
    factors = observe_evidence(factors, evidence)
    graph = generate_graph_from_factors(factors)

    # Uncomment the following line to visualize the graph. Note that we create
    # an undirected graph regardless of the input graph since 1) this
    # facilitates graph traversal, and 2) the algorithm for undirected and
    # directed graphs is essentially the same for tree-like graphs.
    # visualize_graph(graph)

    # You can use any node as the root since the graph is a tree. For simplicity
    # we always use node 0 for this assignment.
    root = 0

    # Create structure to hold messages
    num_nodes = graph.number_of_nodes()
    messages = [[None] * num_nodes for _ in range(num_nodes)]

    """ YOUR CODE HERE
    Use the algorithm from lecture 4 and perform message passing over the entire
    graph. Recall the message passing protocol, that a node can only send a
    message to a neighboring node only when it has received messages from all
    its other neighbors.
    Since the provided graphical model is a tree, we can use a two-phase 
    approach. First we send messages inward from leaves towards the root.
    After this is done, we can send messages from the root node outward.
    
    Hint: You might find it useful to add auxilliary functions. You may add 
      them as either inner (nested) or external functions.
    """
    n_f = graph.neighbors(root)
    # print(list(n_f))
    for e in n_f:
        collect(graph, root, e, messages)
    n_f = graph.neighbors(root)
    # print(list(n_f))
    for e in n_f:
        distribute(graph, root, e, messages)
    # print(messages[:])
    # print(V)
    for i in V:
        res = computeMarginal(graph, i, messages)
        marginals.append(res)
    # print(marginals)
    return marginals


def map_eliminate(factors, evidence):
    """Obtains the maximum a posteriori configuration for a tree graph
    given optional evidence

    Args:
        factors (List[Factor]): List of factors representing the graphical model
        evidence (Dict): Observed evidence. evidence[k]=v denotes that the
          variable k is assigned to value v.

    Returns:
        max_decoding (Dict): MAP configuration
        log_prob_max: Log probability of MAP configuration. Note that this is
          log p(MAP, e) instead of p(MAP|e), i.e. it is the unnormalized
          representation of the conditional probability.
    """

    max_decoding = {}
    log_prob_max = 0.0

    """ YOUR CODE HERE
    Use the algorithm from lecture 5 and perform message passing over the entire
    graph to obtain the MAP configuration. Again, recall the message passing 
    protocol.
    Your code should be similar to compute_marginals_bp().
    To avoid underflow, first transform the factors in the probabilities
    to **log scale** and perform all operations on log scale instead.
    You may ignore the warning for taking log of zero, that is the desired
    behavior.
    """

    return max_decoding, log_prob_max
