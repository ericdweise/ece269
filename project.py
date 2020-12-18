'''
Mini Project Submission.

This project is an implementation of Orthogonal Matching Pursuit (OMB)
for sparse signal recovery.

Note, this implementation assumes that 

Class: ECE 269A, Linear Algebra
Instructor: Dr. Piya Pal
Term: Fall 2020


Write by Eric D. Weise (ericdweise@gmail.com)
'''



import matplotlib.pyplot as plt 
import numpy as np
import random

from mpl_toolkits import mplot3d 

from omp import OmpSolver
from omp import print_vec


# CONSTANTS
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


def part_c_worker(N, m_values, s_values):

    # Set up plotting variables
    m_coords = np.outer(np.array(m_values), np.ones(len(s_values)))
    m_coords = m_coords.astype('int16')
    s_coords = np.outer(np.ones(len(m_values)), np.array(s_values))
    s_coords = s_coords.astype('int16')

    esr_val = np.zeros(s_coords.shape)
    err_val = np.zeros(s_coords.shape)

    # Iterate over height of matrix
    for i in range(m_coords.shape[0]):
        M = m_coords[i,0]

        print(f'Created {M}x{N} matrix')
        A = generate_random_matrix(M,N)
        omp_solver = OmpSolver(A)

        for j in range(s_coords.shape[1]):
            s = s_coords[0,j]

            print(f'    s={s}')
            # 2000 monte carlo experiments
            esr_count = 0
            error_tot = 0
            for exp_num in range(2000):
                signal, index_set = generate_random_pure_signal(A,s)
                recov_signal, support_set, error = omp_solver.solve(signal)

                if check_lists_equal(index_set, support_set):
                    esr_count += 1

                error_tot += error

            esr_val[i,j] = float(esr_count) / 20    # in percent
            err_val[i,j] = error_tot / 2000

    # Plot Noiseless Phase Transition
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    ax.plot_surface(m_coords, s_coords, esr_val)
    ax.set_ylabel('Sparsity, s')
    ax.set_xlabel('Signal Length, M')
    ax.set_zlabel('Probability (percent)')
    ax.set_title(f'Noiseless Phase Transition\nDictionary length: N={N}')
    ax.view_init(azim=45)
    # plt.show()
    plt.savefig(f'noiseless-phase-transition-plot-N-{N}.png')

    # Plot Error values
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    ax.plot_surface(m_coords, s_coords, err_val)
    ax.set_ylabel('Sparsity, s')
    ax.set_xlabel('Signal Length, M')
    ax.set_zlabel('Normalized Error')
    ax.set_title(f'Normalized Error\nDictionary length: N={N}')
    ax.view_init(azim=45)
    plt.savefig(f'error-phase-transition-plot-N-{N}.png')

    # save data in csvs
    with open(f'part-c-data-N-{N}.csv','w') as fp:
        fout.write('M,S,ESR,Err\n')
        for i in range(err_val.shape[0]):
            for j in range(err_val.shape[1]):
                fout.write(f'{m_coords[i,j]},{s_coords[i,j]},{esr_val[i,j]},{err_val[i,j]}\n')


def part_c():
    '''
    Perform the noiseless experiments as listed in part C
    '''

    # EXACT SUPPORT RECOVERY
    N = 20
    Ms = range(1,11)
    Ss = range(1,21)
    part_c_worker(N,Ms, Ss)

    N = 50
    Ms = range(1,25,2)
    Ss = range(1,51,3)
    part_c_worker(N,Ms, Ss)

    N = 100
    Ms = range(1,50,3)
    Ss = range(1,101,4)
    part_c_worker(N,Ms, Ss)


def pard_d_1():
    pass


def part_d_2():
    pass


def part_d_3():
    pass


def test():
    m_rows = 5
    n_cols = 50
    A = generate_random_matrix(m_rows, n_cols)
    print(f'\nshape of A: {A.shape}')

    osolver = OmpSolver(A)
    print('\n===== OMP SOLVER ====')
    print(f'Shape of dictionary: {osolver.dictionary.shape}')
    print(f'Coherence coefficient: {osolver.mu}')
    print(f'Maximum sparsity: {osolver.max_sparsity}')

    print('\n****** pure vector with sparsity=3 *****')
    b, i = generate_random_pure_signal(A, 3)
    print('\n===== measured signal =====')
    print_vec(b)

    print('\n===== reconstructed signal =====')
    b_reconstructed = osolver.solve(b)
    print_vec(b_reconstructed)
    print(f'Support set: {osolver.support_set}')


def main():
    part_c()


if __name__ == '__main__':
    main()
