'''
Project Submission.

This project is an implementation of Orthogonal Matching Pursuit (OMB)
for sparse signal recovery.

Class: ECE 269A, Linear Algebra
Instructor: Dr. Piya Pal
Term: Fall 2020


Write by Eric D. Weise (ericdweise@gmail.com)
'''



import numpy as np

from tools import omp
from tools import signal_choices
from tools import save_data
from tools import check_lists_equal
from tools import generate_random_matrix
from tools import generate_pure_signal
from tools import add_noise
from tools import normalized_error



def build_d2_plots(N, m_values, s_values, sigma):

    # Set up plotting variables
    M = np.outer(np.array(m_values), np.ones(len(s_values)))
    M = M.astype('int16')
    S = np.outer(np.ones(len(m_values)), np.array(s_values))
    S = S.astype('int16')

    ESR = np.zeros(S.shape)
    ERR = np.zeros(S.shape)

    # Iterate over height of matrix
    for i in range(M.shape[0]):
        m = M[i,0]

        print(f'  Dictionary size: {m}x{N}')
        A = generate_random_matrix(m,N)

        for j in range(S.shape[1]):
            s = S[0,j]

            # 2000 monte carlo experiments
            esr_count = 0
            error_tot = 0
            for exp_num in range(2000):
                signal, index_set = generate_pure_signal(N, s)

                y = np.dot(A, signal)
                y_noisy, noise_norm = add_noise(y, sigma)

                recov_signal, support_set = omp(A, y_noisy,
                        error_bound=noise_norm)

                if check_lists_equal(index_set, support_set):
                    esr_count += 1

                error_tot += normalized_error(signal, recov_signal)

            ESR[i,j] = float(esr_count) / 20    # in percent
            ERR[i,j] = error_tot / 2000

    # save data
    save_data(M, S, ERR, ESR, 'd2', N, sigma)


def run_part_d2():
    '''Perform the noisy experiments detailed in part d 1
    '''
    N = 20
    Ms = range(1,11)
    Ss = range(1,21)
    sigma = 0.05
    build_d2_plots(N, Ms, Ss, sigma)
    sigma = 1
    build_d2_plots(N, Ms, Ss, sigma)

    N = 50
    Ms = range(1,17,3)
    Ss = [1,2,3,4,5,10,15,20,25,35,50]
    sigma = 0.05
    build_d2_plots(N, Ms, Ss, sigma)
    sigma = 1
    build_d2_plots(N, Ms, Ss, sigma)

    N = 100
    Ms = range(1,22,3)
    Ss = [1,2,3,4,5,10,15,20,30,40,60,80,100]
    sigma = 0.05
    build_d2_plots(N, Ms, Ss, sigma)
    sigma = 1
    build_d2_plots(N, Ms, Ss, sigma)


if __name__ == '__main__':
    run_part_d2()
