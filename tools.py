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

    # Normalize columns
    for j in range(A.shape[1]):  # iterate over columns
        col_norm = np.linalg.norm(A[:,j])
        for i in range(A.shape[0]):  # iterate over rows
            A[i,j] /= col_norm

    return A


def generate_pure_signal(N, s):
    '''Generate sparse signal vector.
    Generate a random vector that is a linear combination of s atoms
    in the dictionary, with coefficients in {-10, ... ,-1}U{1, ... 10}
    Args:
        N: length of signal to be generated.
        s: number of non-zero entries in x.
    Returns:
        ndarray
    '''
    v = np.zeros(N)
    index_set = np.random.choice(range(N), s, replace=False)

    for i in index_set:
        v[i] = np.random.choice(signal_choices)

    return v, index_set


def add_noise(vector, variance):
    noise = np.random.normal(0, variance, vector.shape[0])
    noise_norm = np.linalg.norm(noise)

    return vector + noise, noise_norm


def normalized_error(v1, v2):
    return np.linalg.norm(v1 - v2) / np.linalg.norm(v1) 


def save_data(M, S, ERR, ESR, part, N, sigma=None):
    # Save Numpy Files
    if sigma is None:
        np.save(f'./data/{part}_N-{N}_m-values', M)
        np.save(f'./data/{part}_N-{N}_s-values', S)
        np.save(f'./data/{part}_N-{N}_esr-values', ESR)
        np.save(f'./data/{part}_N-{N}_err-values', ERR)
    else:
        np.save(f'./data/{part}_N-{N}_noise-{sigma}_m-values', M)
        np.save(f'./data/{part}_N-{N}_noise-{sigma}_s-values', S)
        np.save(f'./data/{part}_N-{N}_noise-{sigma}_esr-values', ESR)
        np.save(f'./data/{part}_N-{N}_noise-{sigma}_err-values', ERR)

    ## ERROR PLOT
    fig = plt.figure()
    ax = plt.axes(projection ='3d')

    ax.set_zlabel('Normalized\nError')
    ax.set_ylabel('Signal Sparsity, s')
    ax.set_xlabel('Number of Measurements, M')
    ax.view_init(azim=-45)

    if sigma is None:
        ax.set_title(f'N = {N}')
        err_filename = f'./plots/{part}-error_N-{N}.png'
        esr_filename = f'./plots/{part}-recov_N-{N}.png'
    else:
        ax.set_title(f'N = {N}\nNoise: {sigma}')
        err_filename = f'./plots/{part}-error_N-{N}_noise-{sigma}.png'
        esr_filename = f'./plots/{part}-recov_N-{N}_noise-{sigma}.png'
    ax.plot_surface(M, S, ERR)
    plt.savefig(err_filename)

    plt.close()

    # ESR PLOT
    fig = plt.figure()
    ax = plt.axes(projection ='3d')

    ax.set_zlabel('Probability (percent)')
    ax.set_ylabel('Signal Sparsity, s')
    ax.set_xlabel('Number of Measurements, M')
    ax.view_init(azim=45)

    if sigma is None:
        ax.set_title(f'N = {N}')
        err_filename = f'./plots/{part}-error_N-{N}.png'
        esr_filename = f'./plots/{part}-recov_N-{N}.png'
    else:
        ax.set_title(f'N = {N}\nNoise: {sigma}')
        err_filename = f'./plots/{part}-error_N-{N}_noise-{sigma}.png'
        esr_filename = f'./plots/{part}-recov_N-{N}_noise-{sigma}.png'
    ax.plot_surface(M, S, ESR)
    plt.savefig(esr_filename)

    plt.close()


def plot_from_data():

    # part c
    stub = 'data/part-c-N-100-'
    helper('c', 100, f'{stub}m-values.npy', f'{stub}s-values.npy', f'{stub}ERR-values.npy', f'{stub}ESR-values.npy')

    stub = 'data/part-c-N-20-'
    helper('c', 20, f'{stub}m-values.npy', f'{stub}s-values.npy', f'{stub}ERR-values.npy', f'{stub}ESR-values.npy')

    stub = 'data/part-c-N-50-'
    helper('c', 50, f'{stub}m-values.npy', f'{stub}s-values.npy', f'{stub}ERR-values.npy', f'{stub}ESR-values.npy')

    # part d1
    stub = 'data/part-d1-N-100-sigma-0.05-'
    helper('d1', '100-sigma-0.05', f'{stub}m-values.npy', f'{stub}s-values.npy', f'{stub}ERR-values.npy', f'{stub}ESR-values.npy')
    
    stub = 'data/part-d1-N-100-sigma-5-'
    helper('d1', '100-sigma-5', f'{stub}m-values.npy', f'{stub}s-values.npy', f'{stub}ERR-values.npy', f'{stub}ESR-values.npy')
    
    stub = 'data/part-d1-N-20-sigma-0.05-'
    helper('d1', '20-sigma-0.05', f'{stub}m-values.npy', f'{stub}s-values.npy', f'{stub}ERR-values.npy', f'{stub}ESR-values.npy')
    
    stub = 'data/part-d1-N-20-sigma-5-'
    helper('d1', '20-sigma-5', f'{stub}m-values.npy', f'{stub}s-values.npy', f'{stub}ERR-values.npy', f'{stub}ESR-values.npy')
    
    stub = 'data/part-d1-N-50-sigma-0.05-'
    helper('d1', '50-sigma-0.05', f'{stub}m-values.npy', f'{stub}s-values.npy', f'{stub}ERR-values.npy', f'{stub}ESR-values.npy')
    
    stub = 'data/part-d1-N-50-sigma-5-'
    helper('d1', '50-sigma-5', f'{stub}m-values.npy', f'{stub}s-values.npy', f'{stub}ERR-values.npy', f'{stub}ESR-values.npy')
    
    # part d2
    stub = 'data/part-d2-N-100-sigma-0.05-'
    helper('d2', '100-sigma-0.05', f'{stub}m-values.npy', f'{stub}s-values.npy', f'{stub}ERR-values.npy', f'{stub}ESR-values.npy')
    
    stub = 'data/part-d2-N-100-sigma-5-'
    helper('d2', '100-sigma-5', f'{stub}m-values.npy', f'{stub}s-values.npy', f'{stub}ERR-values.npy', f'{stub}ESR-values.npy')
    
    stub = 'data/part-d2-N-20-sigma-0.05-'
    helper('d2', '20-sigma-0.05', f'{stub}m-values.npy', f'{stub}s-values.npy', f'{stub}ERR-values.npy', f'{stub}ESR-values.npy')
    
    stub = 'data/part-d2-N-20-sigma-5-'
    helper('d2', '20-sigma-5', f'{stub}m-values.npy', f'{stub}s-values.npy', f'{stub}ERR-values.npy', f'{stub}ESR-values.npy')
    
    stub = 'data/part-d2-N-50-sigma-0.05-'
    helper('d2', '50-sigma-0.05', f'{stub}m-values.npy', f'{stub}s-values.npy', f'{stub}ERR-values.npy', f'{stub}ESR-values.npy')
    
    stub = 'data/part-d2-N-50-sigma-5-'
    helper('d2', '50-sigma-5', f'{stub}m-values.npy', f'{stub}s-values.npy', f'{stub}ERR-values.npy', f'{stub}ESR-values.npy')
