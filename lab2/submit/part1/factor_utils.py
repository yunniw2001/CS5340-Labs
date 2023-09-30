# taken from part 1
import copy
import numpy as np
from factor import Factor, index_to_assignment, assignment_to_index


def factor_product(A, B):
    """
    Computes the factor product of A and B e.g. A = f(x1, x2); B = f(x1, x3); out=f(x1, x2, x3) = f(x1, x2)f(x1, x3)

    Args:
        A: first Factor
        B: second Factor

    Returns:
        Returns the factor product of A and B
    """
    if A.is_empty():
        return B
    if B.is_empty():
        return A
    out = Factor()

    """ YOUR CODE HERE """
    out.var = np.union1d(A.var, B.var)  # find the union, find all x_i

    # Compute mapping between the variable ordering between the two factors
    # and the output to set the cardinality
    out.card = np.zeros(len(out.var), np.int64)
    mapA = np.argmax(out.var[None, :] == A.var[:, None], axis=-1)
    # mapA is an index array, refers to A.var's index in out.var
    mapB = np.argmax(out.var[None, :] == B.var[:, None], axis=-1)
    out.card[mapA] = A.card
    out.card[mapB] = B.card

    out.val = np.zeros(np.prod(out.card))
    assignments = out.get_all_assignments()
    idxA = assignment_to_index(assignments[:, mapA], A.card)
    idxB = assignment_to_index(assignments[:, mapB], B.card)

    out.val = A.val[idxA] * B.val[idxB]
    """ END YOUR CODE HERE """
    return out


def factor_marginalize(factor, var):
    """
    Returns factor after variables in var have been marginalized out.

    Args:
        factor: factor to be marginalized
        var: numpy array of variables to be marginalized over

    Returns:
        marginalized factor
    """
    out = copy.deepcopy(factor)

    """ YOUR CODE HERE
     HINT: Use the code from lab1 """
    out.var = np.setdiff1d(factor.var, var)

    out.card = factor.card[~np.isin(factor.var, var)]

    var = np.array(var)

    var_axis = tuple(np.where(~np.isin(factor.var, var))[0])
    merge_axis = tuple(np.where(np.isin(factor.var, var))[0])

    out.val = np.zeros(np.prod(out.card))

    all_assignments = factor.get_all_assignments()
    seen = {}
    first_ap_row = []
    indices = []

    for i, row in enumerate(map(tuple, all_assignments[:, var_axis])):
        if row not in seen:
            seen[row] = len(seen)
            first_ap_row.append(row)
        indices.append(seen[row])
    # first_ap_row, indices = np.unique(all_assignments[:, var_axis], axis=0, return_inverse=True)

    out.val = np.bincount(indices, weights=factor.val)
    """ END YOUR CODE HERE """
    return out


def factor_evidence(factor, evidence):
    """
    Observes evidence and retains entries containing the observed evidence. Also removes the evidence random variables
    because they are already observed e.g. factor=f(1, 2) and evidence={1: 0} returns f(2) with entries from node1=0
    Args:
        factor: factor to reduce using evidence
        evidence:  dictionary of node:evidence pair where evidence[1] = evidence of node 1.
    Returns:
        Reduced factor that does not contain any variables in the evidence. Return an empty factor if all the
        factor's variables are observed.
    """
    if evidence is None:
        return factor
    out = copy.deepcopy(factor)

    """ YOUR CODE HERE,     HINT: copy from lab2 part 1! """
    cur_all_assignment = out.get_all_assignments()
    observe_vars = np.array(list(evidence.keys()))
    observe_vals = np.array(list(evidence.values()))
    intersection = np.intersect1d(out.var, observe_vars)
    if not len(intersection)>0:
        return out

    observe_vals = [evidence.get(k) for k in intersection]

    observe_vars = intersection
    observe_vars_idx =  np.argmax(out.var[None, :] == observe_vars[:, None], axis=-1)

    related_assignment_mask = np.all(cur_all_assignment[:,observe_vars_idx]==observe_vals,axis=1)

    """ END YOUR CODE HERE """
    out.var = np.array(np.setdiff1d(out.var,intersection))
    out.card = np.array([factor.card[i] for i in range(len(factor.card)) if i not in observe_vars_idx])
    out.val = np.array([factor.val[i] for i in range(len(factor.val)) if related_assignment_mask[i]])
    return out


if __name__ == '__main__':
    main()