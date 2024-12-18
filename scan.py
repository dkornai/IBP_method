import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import time

from gibbsibp import UncollapsedGibbsIBP
from util import extract_mean_from_samples, compare_distance, add_noise_to_obs


"""
 1): base element
     -----
    | \   |
    |  o  X
    |     |
     -----

 2): base element
     -----
    |   / |
    X  o  |
    |     |
     -----

3): not visual and not dynamic compositional
     --X--
    |     |
    |  o  |
    |  |  |
     -----

4): not visual but dynamic compositional
     -----
    |     |
    |  o  |
    |   \ |
     --X--

5): visual and dynamic compositional
     -----
    |  |  |
    X  o  X
    |     |
     -----

6): not dynamic but visual compositional
     --X--
    |     |
    X  o  |
    | /   |
     -----
"""

conditions = []
conditions.append({'force':np.array([ 0., 0.]),             'cue':np.array([0., 0., 0., 0.,]), 'name':'null'})
conditions.append({'force':np.array([-0.866025404, 0.5]),   'cue':np.array([1., 0., 0., 0.,]), 'name':'base1'})
conditions.append({'force':np.array([ 0.866025404, 0.5]),   'cue':np.array([0., 0., 1., 0.,]), 'name':'base2'})
conditions.append({'force':np.array([ 0.,-1.]),             'cue':np.array([0., 1., 0., 0.,]), 'name':'Comp. visual-/dynamic-'})
conditions.append({'force':np.array([ 0.866025404, -0.5]),  'cue':np.array([0., 0., 0., 1.,]), 'name':'Comp. visual-/dynamic+'})
conditions.append({'force':np.array([ 0, 1]),               'cue':np.array([1., 0., 1., 0.,]), 'name':'Comp. visual+/dynamic+'})
conditions.append({'force':np.array([-0.866025404, -0.5]),  'cue':np.array([0., 1., 1., 0.,]), 'name':'Comp. visual+/dynamic-'})

# # temporary division by 4, to make the model converge
# for condition in conditions:
#     condition['force'] = condition['force']/4

true_A = []
true_Y = []
for i, condition in enumerate(conditions):
    if i > 0:
        true_A.append(condition['force'])
        true_Y.append(condition['cue'])

def generate_input_data(condition_seq):
    """
    Generate input data for the model, based on sequences of conditions experienced by the subjects.
    """
    X_dataset = []
    F_dataset = []

    for condition in condition_seq:
        X_dataset.append(conditions[condition]['cue'])
        F_dataset.append(conditions[condition]['force'])

    X_dataset = torch.tensor(X_dataset, dtype=torch.float32)
    F_dataset = torch.tensor(F_dataset, dtype=torch.float32)

    return X_dataset, F_dataset

dataset = pd.read_excel('training_data.xlsx')
dataset.head()

seq_1 = dataset['cp_pres_1'].to_list()
seq_1 = [int(x) for x in seq_1]
seq_1 = [x for x in seq_1 if x != 0]
print(len(seq_1))


times = []

## Add noise to the observation data
SIGMA_A = 1.0
alpha_list   = [0.01, 0.02, 0.05, 0.1]
sigma_n_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
epsilon_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
lambd_list   = [1-x for x in epsilon_list]

i = 1

for alpha in alpha_list:
    for sigma_n in sigma_n_list:
        for epsilon, lambd in zip(epsilon_list, lambd_list):
                print(f"\nStarting analysis {i}/144")
                print(f"sigma_n: {sigma_n}, epsilon: {epsilon}, lambd: {lambd}")
                
                X_dataset, F_dataset = generate_input_data(seq_1)
                # add noise to the observation data
                X_dataset, F_dataset = add_noise_to_obs(
                    X_dataset, F_dataset, F_noise_std=sigma_n, lambd=lambd, epsilon=epsilon)
                # initialize the model
                inf = UncollapsedGibbsIBP(
                    alpha=alpha, K=1, max_K=6, sigma_a=SIGMA_A, sigma_n=sigma_n, epsilon=epsilon, lambd=lambd, phi=0.25)
                # run the model
                start_time = time.time()
                
                As, Zs, Ys = inf.gibbs(F_dataset, X_dataset, iters = 300)
                
                end_time = time.time()
                
                # measure the time
                total_time = np.round(end_time - start_time,1)
                times.append(total_time)
                avg_time = np.round(np.mean(times), 1)
                print(f"Current runtime: {total_time}, average runtime: {avg_time}s,  projected time remaining: {np.round(avg_time*(144-i)/3600)} hours")

                # save the results
                os.chdir('paramscan')
                pickle.dump(As, open(f'As_{i}.pkl', 'wb'))
                pickle.dump(Zs, open(f'Zs_{i}.pkl', 'wb'))
                pickle.dump(Ys, open(f'Ys_{i}.pkl', 'wb'))
                os.chdir('..')

                i += 1
