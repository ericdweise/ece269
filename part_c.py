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

from omp import OmpSolver
from tools import signal_choices
from tools import plot_error
from tools import plot_esr
from tools import check_lists_equal
from tools import generate_random_matrix
from tools import generate_random_pure_signal
from tools import generate_noise
from omp import print_vec



def build_c_plots(N, m_values, s_values):

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

    # save data
    np.save(f'./data/part-c-N-{N}-m-values', m_coords)
    np.save(f'./data/part-c-N-{N}-s-values', s_coords)
    np.save(f'./data/part-c-N-{N}-esr-values', esr_val)
    np.save(f'./data/part-c-N-{N}-err-values', err_val)

    # Plot Noiseless Phase Transition
    plot_esr(f'c', N, m_coords, s_coords, esr_val)

    # Plot Error values
    plot_error(f'c', N, m_coords, s_coords, err_val)


def run_part_c():
    '''Perform the noiseless experiments detailed in part c
    '''
    N = 20
    Ms = range(1,11)
    Ss = range(1,21)
    build_c_plots(N, Ms, Ss)

    N = 50
    Ms = range(1,17,3)
    Ss = [1,2,3,4,5,10,15,20,25,35,50]
    build_c_plots(N, Ms, Ss)

    N = 100
    Ms = range(1,22,3)
    Ss = [1,2,3,4,5,10,15,20,30,40,60,80,100]
    build_c_plots(N, Ms, Ss)
