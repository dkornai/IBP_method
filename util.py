import numpy as np
from scipy.optimize import linear_sum_assignment

def extract_mean_from_samples(As, Zs, Ys, n=10):
    A_mean = np.round(np.mean(np.array(As[-n:]),axis=0), 2)
    Z_mean = np.round(np.mean(np.array(Zs[-n:]),axis=0), 2)
    Y_mean = np.round(np.mean(np.array(Ys[-n:]),axis=0), 2)

    return A_mean, Z_mean, Y_mean

def compare_distance(reference_matrix, inferred_matrix):
    """
    Compare the distance between rows of the reference and the true matrix.
    use this to create a permutation matrix that reorders the inffered matrix to match the reference matrix,
    and return the permutation matrix
    """
    n, m = reference_matrix.shape
    assert inferred_matrix.shape == (n, m)

    # compute the distance matrix
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = np.linalg.norm(reference_matrix[i] - inferred_matrix[j])

    # find the permutation that minimizes the distance
   
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    
    # create the permutation matrix that corresponds to this reordering
    permutation_matrix = np.zeros((n, n))
    for i in range(n):
        permutation_matrix[i, col_ind[i]] = 1

    return permutation_matrix
