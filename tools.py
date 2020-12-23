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

from math import log


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
    noise = np.random.normal(0, variance, vector.shape)
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


def omp(A, y, error_bound=0.01, stop_after=float('inf')):
    '''Find the OMP solution with index for the given signal.

    Args:
        signal: The "measured" signal
        error_bound: OPTIONAL: If set OMP will stop when 
                ||Ax-b|| < error_bound
        stop_after_n: OPTIONAL: An integer number of steps 
                after which OMP will stop

    Returns:
        x: a 1xN sparse vector
        support set: a list of the indices used
    '''
    M = A.shape[0]
    N = A.shape[1]
    residual = np.copy(y)
    support_set = []
    x_hat = np.zeros(N)

    # MAIN LOOP
    count = 0
    while count < stop_after and np.linalg.norm(residual) > error_bound:
        count += 1

        # Find index of atom with largest correlation to residual
        correlations = np.dot(np.transpose(A), residual)
        index = np.argmax(abs(correlations))
        support_set.append(index)

        # Create matrix with only support set atoms 
        Asub = A[:, support_set]

        # least squares to find reconstructed signal, x-hat
        #    x_hat = (Al^T * Al)^(-1) * Al^T * y
        # where: Al is the tall submatrix containing only the atoms selected so far
        x_hat[support_set] = np.linalg.inv(np.dot(Asub.T, Asub)).dot(Asub.T).dot(y)

        # Update residual: r(k) = y - A(k) * x_hat
        residual = y - np.dot(A, x_hat)

    return x_hat, support_set


def mean_squared_error(img1, img2):
    '''Calculate the Mean Signal to Noise Ratio of two images.
    Images must have the same shape'''
    assert(img1.shape[0] == img2.shape[0])
    assert(img1.shape[1] == img2.shape[1])

    mse = 0.

    for i in range(img1.shape[0]):
        for j in range(img2.shape[1]):
            mse += (img1[i,j] - img2[i,j])**2

    mse /= img1.shape[0] 
    mse /= img1.shape[1]

    return mse


def peak_snr(img1, img2):
    '''Calculate the Mean Signal to Noise Ratio of two images.
    Images must have the same shape'''
    assert(img1.shape[0] == img2.shape[0])
    assert(img1.shape[1] == img2.shape[1])

    mse = mean_squared_error(img1, img2)

    return 20 * log(255) - 10 * log(mse)
