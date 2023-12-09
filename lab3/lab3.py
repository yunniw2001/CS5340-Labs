""" CS5340 Lab 3: Hidden Markov Models
See accompanying PDF for instructions.

Name: WU YIHANG
Email: e1216288@u.nus.edu
Student ID: A0285643W
"""
import numpy as np
import scipy.stats
from scipy.special import softmax
from sklearn.cluster import KMeans


def initialize(n_states, x):
    """Initializes starting value for initial state distribution pi
    and state transition matrix A.

    A and pi are initialized with random starting values which satisfies the
    summation and non-negativity constraints.
    """
    seed = 5340
    np.random.seed(seed)

    pi = np.random.random(n_states)
    A = np.random.random([n_states, n_states])

    # We use softmax to satisify the summation constraints. Since the random
    # values are small and similar in magnitude, the resulting values are close
    # to a uniform distribution with small jitter
    pi = softmax(pi)
    A = softmax(A, axis=-1)

    # Gaussian Observation model parameters
    # We use k-means clustering to initalize the parameters.
    x_cat = np.concatenate(x, axis=0)
    kmeans = KMeans(n_clusters=n_states, random_state=seed).fit(x_cat[:, None])
    mu = kmeans.cluster_centers_[:, 0]
    std = np.array([np.std(x_cat[kmeans.labels_ == l]) for l in range(n_states)])
    phi = {'mu': mu, 'sigma': std}

    return pi, A, phi


"""E-step"""
def e_step(x_list, pi, A, phi):
    """ E-step: Compute posterior distribution of the latent variables,
    p(Z|X, theta_old). Specifically, we compute
      1) gamma(z_n): Marginal posterior distribution, and
      2) xi(z_n-1, z_n): Joint posterior distribution of two successive
         latent states

    Args:
        x_list (List[np.ndarray]): List of sequences of observed measurements
        pi (np.ndarray): Current estimated Initial state distribution (K,)
        A (np.ndarray): Current estimated Transition matrix (K, K)
        phi (Dict[np.ndarray]): Current estimated gaussian parameters

    Returns:
        gamma_list (List[np.ndarray]), xi_list (List[np.ndarray])
    """
    n_states = pi.shape[0]
    xi_list = [np.zeros([len(x) - 1, n_states, n_states]) for x in x_list]

    """ YOUR CODE HERE
    Use the forward-backward procedure on each input sequence to populate 
    "gamma_list" and "xi_list" with the correct values.
    Be sure to use the scaling factor for numerical stability.
    """

    # forward
    alphas = [np.zeros((len(xi), n_states)) for xi in x_list]
    p_x_z_list = np.zeros_like(alphas)
    cn_list = [np.zeros(len(xi)) for xi in x_list]

    for i in range(len(alphas)):
        # for each alpha:
        for k in range(len(alphas[i])):
            # k = 0 initialize
            if k == 0:
                p_x_z_list[i] = scipy.stats.norm.pdf(x_list[i][:, np.newaxis], loc=phi['mu'][np.newaxis, :],
                                                     scale=phi['sigma'][np.newaxis, :])
                # c0 = \sum_{z_0}\tilde{\alpha}(z_0) \tilede\alpha_z1 = p(x0|z0)*p(z0)
                tilde_alpha_zn = pi * p_x_z_list[i][k]
            else:
                tilde_alpha_zn = p_x_z_list[i][k] * (alphas[i][k - 1, :] @ A)
            cn_list[i][k] = np.sum(tilde_alpha_zn)
            # scale: \hat\alpha_zn = \tilde\alpha_zn/cn
            alphas[i][k, :] = tilde_alpha_zn / cn_list[i][k]
    # backward
    betas = [np.ones((len(xi), n_states)) for xi in x_list]

    for i in range(len(betas)):
        for k in range(len(betas[i]) - 1, -1, -1):
            if k == len(betas[i]) - 1:
                continue
            else:
                tilde_beta_zn = np.sum(betas[i][k + 1, :] * p_x_z_list[i][k + 1] * A, axis=1)
                betas[i][k, :] = tilde_beta_zn / cn_list[i][k + 1]

    gamma_list = np.multiply(alphas, betas)

    for i in range(len(xi_list)):
        for n in range(1, len(xi_list[i]) + 1):
            xi_list[i][n - 1] = (alphas[i][n - 1][:, np.newaxis] * p_x_z_list[i][n] * A * betas[i][n]) / cn_list[i][n]

    return gamma_list, xi_list


"""M-step"""


def m_step(x_list, gamma_list, xi_list):
    """M-step of Baum-Welch: Maximises the log complete-data likelihood for
    Gaussian HMMs.
    
    Args:
        x_list (List[np.ndarray]): List of sequences of observed measurements
        gamma_list (List[np.ndarray]): Marginal posterior distribution
        xi_list (List[np.ndarray]): Joint posterior distribution of two
          successive latent states

    Returns:
        pi (np.ndarray): Initial state distribution
        A (np.ndarray): Transition matrix
        phi (Dict[np.ndarray]): Parameters for the Gaussian HMM model, contains
          two fields 'mu', 'sigma' for the mean and standard deviation
          respectively.
    """

    n_states = gamma_list[0].shape[1]
    pi = np.zeros([n_states])
    A = np.zeros([n_states, n_states])
    phi = {'mu': np.zeros(n_states),
           'sigma': np.zeros(n_states)}

    """ YOUR CODE HERE
    Compute the complete-data maximum likelihood estimates for pi, A, phi.
    """
    # update A
    xi_list = np.array(xi_list)
    for j in range(len(A)):
        for k in range(len(A[j])):
            A[j][k] = np.sum(xi_list[:, :, j, k]) / np.sum(xi_list[:, :, j, :])

    # update pi
    gamma_list = np.array(gamma_list)
    gama_z1_j_sum = np.sum(gamma_list[:, 0])
    for k in range(len(pi)):
        pi[k] = np.sum(gamma_list[:, 0, k]) / gama_z1_j_sum

    # update miu sigma
    for k in range(n_states):
        gama_zn_k_sum = np.sum(gamma_list[:, :, k])
        phi['mu'][k] = np.sum(gamma_list[:, :, k] * x_list) / gama_zn_k_sum
        phi['sigma'][k] = np.sqrt(
            np.sum(gamma_list[:, :, k] * (x_list - phi['mu'][k]) * (x_list - phi['mu'][k])) / gama_zn_k_sum)

    return pi, A, phi


"""Putting them together"""


def fit_hmm(x_list, n_states):
    """Fit HMM parameters to observed data using Baum-Welch algorithm

    Args:
        x_list (List[np.ndarray]): List of sequences of observed measurements
        n_states (int): Number of latent states to use for the estimation.

    Returns:
        pi (np.ndarray): Initial state distribution
        A (np.ndarray): Time-independent stochastic transition matrix
        phi (Dict[np.ndarray]): Parameters for the Gaussian HMM model, contains
          two fields 'mu', 'sigma' for the mean and standard deviation
          respectively.

    """

    # We randomly initialize pi and A, and use k-means to initialize phi
    # Please do NOT change the initialization function since that will affect
    # grading
    pi, A, phi = initialize(n_states, x_list)

    """ YOUR CODE HERE
     Populate the values of pi, A, phi with the correct values. 
    """
    threshold = 1e-4

    while True:
        old_mu = np.copy(phi['mu'])
        old_sigma = np.copy(phi['sigma'])

        gamma_list, xi_list = e_step(x_list, pi, A, phi)
        pi, A, phi = m_step(x_list, gamma_list, xi_list)

        if_mu_converged = np.all(np.abs(old_mu - phi['mu']) < threshold)
        if_sigma_converged = np.all(np.abs(old_sigma - phi['sigma']) < threshold)

        if if_mu_converged and if_sigma_converged:
            break

    return pi, A, phi
