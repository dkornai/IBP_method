import numpy as np
from numpy.random import random
import torch
from scipy.optimize import linear_sum_assignment

def extract_mean_from_samples(As, Zs, Ys, n=10):
    A_mean = np.round(np.mean(np.array(As[-n:]),axis=0), 2)
    Z_mean = np.round(np.mean(np.array(Zs[-n:]),axis=0), 2)
    Y_mean = np.round(np.mean(np.array(Ys[-n:]),axis=0), 2)

    F_mean = np.round(Z_mean @ A_mean,2)
    X_mean = np.round(Z_mean @ Y_mean,2)

    return A_mean, Z_mean, Y_mean, F_mean, X_mean

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

def add_noise_to_obs(X, F, F_noise_var = 0.01, lambd=0.98, epsilon=0.02):
    """
    Add gaussian noise to force data, and randomly flip pixels in the observation data
    
    Parameters: 
    X: torch.tensor, the observation data
    F: torch.tensor, the force data
    F_noise_std: float, the standard deviation of the gaussian noise added to the force data
    lambd: float, the probability of a 1 in the observation data being kept as 1
    epsilon: float, the probability of a 0 in the observation data being flipped to 1

    Returns:
    X_noisy: torch.tensor, the noisy observation data
    F: torch.tensor, the noisy force
    """
    # add noise to the force data
    F += (torch.randn(F.size()) * np.sqrt(F_noise_var))
    
    # add noise to the visual data
    X_noisy = torch.zeros(X.size())
    for i in range(X.size()[0]):
        for j in range(X.size()[1]):
            if X[i, j] == 1:
                if random() < lambd: # flip the pixel off with probability 1-lambd
                    X_noisy[i, j] = 1
                else:
                    X_noisy[i, j] = 0
            else:
                if random() < epsilon: # flip the pixel on with probability epsilon
                    X_noisy[i, j] = 1
                else:
                    X_noisy[i, j] = 0

    return X_noisy, F


def check_basis_elements(A, Y):
    """
    Of the six condtions, check which are basis elements in the inferred A and Y matrices
    """
    # Define the six conditions
    element_base1_force      = np.array([-0.866025404, 0.5])
    element_base1_cue        = np.array([1., 0., 0., 0.,])

    element_base2_force      = np.array([0.866025404, 0.5])
    element_base2_cue        = np.array([0., 0., 1., 0.,])

    element_comp_vm_dm_force = np.array([ 0.,-1.])
    element_comp_vm_dm_cue   = np.array([0., 1., 0., 0.,])

    element_comp_vm_dp_force = np.array([ 0.866025404, -0.5])
    element_comp_vm_dp_cue   = np.array([0., 0., 0., 1.,])

    element_comp_vp_dp_force = np.array([ 0., 1.])
    element_comp_vp_dp_cue   = np.array([1., 0., 1., 0.,])

    element_comp_vp_dm_force = np.array([-0.866025404, -0.5])
    element_comp_vp_dm_cue   = np.array([0., 1., 1., 0.,])

    forces = [element_base1_force, element_base2_force, element_comp_vm_dm_force, element_comp_vm_dp_force, element_comp_vp_dp_force, element_comp_vp_dm_force]
    cues   = [element_base1_cue, element_base2_cue, element_comp_vm_dm_cue, element_comp_vm_dp_cue, element_comp_vp_dp_cue, element_comp_vp_dm_cue]
    names  = ['    base1', '    base2', 'vis-/dyn-', 'vis-/dyn+', 'vis+/dyn+', 'vis+/dyn-']

    # Check if the conditions are basis elements
    for i in range(6):
        reference_force = forces[i]
        reference_cue = cues[i]

        basis = False
        for j in range(len(A)):
            a_row = A[j]
            y_row = Y[j]
        
            if np.allclose(a_row, reference_force, atol=0.1) and np.allclose(y_row, reference_cue, rtol=0.1):
                print(f'Element {names[i]} is basis')
                basis = True
        
        if not basis:
                print(f'Element {names[i]} not')