'''
Project Submission.

This project is an implementation of Orthogonal Matching Pursuit (OMB)
for sparse signal recovery.

Class: ECE 269A, Linear Algebra
Instructor: Dr. Piya Pal
Term: Fall 2020


Write by Eric D. Weise (ericdweise@gmail.com)
'''

import matplotlib.pyplot as plt 
import numpy as np


signal_choices = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9,10]


def check_lists_equal(list_1, list_2):
    """ Check if both the lists are of same length and if yes then compare
    sorted versions of both the list to check if both of them are equal
    i.e. contain similar elements with same frequency. """
    if len(list_1) != len(list_2):
        return False
    return sorted(list_1) == sorted(list_2)


def generate_random_matrix(m, n, mean=0, var=1):
    '''Create a random matrix, A. 
    Each entry of A is taken randomly from the standard normal 
    distribution: N(mean, var).
    Args:
        m: number of rows in A
        n: number of cols in A
        mean: mean of the elements of A. [default: 0]
        var: variance of the elements in A. [default 1]
    Returns:
        A: m by n ndarray
    '''
    # Generate Random Matrix
    A = np.random.normal(mean, var, (m,n))

    return A


def generate_random_pure_signal(dictionary, s):
    '''Generate sparse signal vector.
    Generate a random vector that is a linear combination of s atoms
    in the dictionary, with coefficients in {-10, ... ,-1}U{1, ... 10}
    Args:
        dictionary: nparray whose columns are a dictionary of atoms
        s: number of non-zero entries in x.
    Yields:
        ndarray that is an exact solution to Ax=b
    '''
    n = dictionary.shape[1]
    v = np.zeros(n)
    indices = np.random.choice(range(n), s, replace=False)

    for i in indices:
        v[i] = np.random.choice(signal_choices)

    return np.dot(dictionary, v), indices


def generate_noise(M, variance):
	return np.random.normal(0, variance, M)


def plot_esr(part, N, m_coords, s_coords, esr_val):
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    ax.plot_surface(m_coords, s_coords, esr_val)
    ax.set_ylabel('Sparsity, s')
    ax.set_xlabel('Signal Length, M')
    ax.set_zlabel('Probability (percent)')
    ax.set_title(f'Probability of Exact Support Recovery\nDictionary length: N={N}')
    ax.view_init(azim=45)
    plt.savefig(f'./plots/part-{part}-N-{N}-esr.png')
    plt.close()


def plot_error(part, N, m_coords, s_coords, err_val):
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    ax.plot_surface(m_coords, s_coords, err_val)
    ax.set_ylabel('Sparsity, s')
    ax.set_xlabel('Signal Length, M')
    ax.set_zlabel('Normalized\nError')
    ax.set_title(f'Normalized Error\nDictionary length: N={N}')
    ax.view_init(azim=45)
    plt.savefig(f'./plots/part-{part}-N-{N}-error.png')
    plt.close()
