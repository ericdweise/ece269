import numpy as np

from math import floor
from tools import generate_random_matrix


def print_vec(v):
    s = np.array_repr(v).replace('\n','')
    print(s)


def compute_correlation(v1, v2):
    return abs(np.dot(v1, v2))

def expand_sparse_vector(vector, support_set, N):
    expanded = np.zeros(N)

    for i in range(len(support_set)):
        expanded[support_set[i]] = vector[i]

    return expanded


class OmpSolver(object):
    '''Calculate the Orthogonal Matching Pursuit solution
    '''

    def __init__(self, M, N, mean=0, var=1):
        '''
        Args:
            M: The number of measurements to make, or that were made
            N: Size of input signals
            stop_at_n: The number of steps after which to terminate OMP.
        '''
        self.dictionary = generate_random_matrix(M,N)


    def calulate_coherence_parameter(self):
        '''Calculate the coherence parameter of the dictionary. 
        This is set in mu. Also calculates the maximum allowed
        sparsity of the solution, set as max_sparsity.
        '''
        self.mu = 0

        for i in range(self.dictionary.shape[1]):
            for j in range(i+1, self.dictionary.shape[1]):
                new = abs(np.dot(self.dictionary[:,i], self.dictionary[:,j]))
                if new > self.mu:
                    self.mu = new

        # Calculate the maximum sparsity of the solution
        self.max_sparsity = floor(1 / 2 / self.mu - 0.5)


    def compress(self, signal):
        return np.dot(self.dictionary, signal)


    def decompress(self, compressed_signal, error_bound=10**(-3), stop_after_n=None):
        '''Find the OMP solution with index for the given signal.

        Args:
            signal: The "measured" signal
            error_bound: OPTIONAL: If set OMP will stop when 
                    ||Ax-b|| < error_bound
            terminate_after_n: OPTIONAL: An integer number of steps 
                    after which OMP will stop
        '''
        assert(len(compressed_signal.shape) == 1)
        assert(compressed_signal.shape[0] == self.dictionary.shape[0]) 

        ## INITIALIZE
        residual = np.copy(compressed_signal)
        support_set = []

        if stop_after_n is None:
            stop_after_n = float('inf')

        # MAIN LOOP
        for loop_count in range(self.dictionary.shape[0]):

            # Find index of atom with largest correlation to residual
            max_correlation = 0
            max_correlation_index = 0

            for j in range(self.dictionary.shape[1]):

                # skip atoms already in support set
                if j in support_set:
                    continue

                correlation = compute_correlation(self.dictionary[:,j], residual)

                if correlation > max_correlation:
                    max_correlation = correlation
                    max_correlation_index = j

            # Save maximum correlation index to the support set
            support_set.append(max_correlation_index)

            # Create matrix with only support set atoms 
            sub_matrix = self.dictionary[:, support_set]

            # least squares to find reconstructed signal, x-hat
            # I implement this step by step, but the basic maths are:
            #    x_hat = (A^T * A)^(-1) * A^T * b
            # where:
            #    A is the tall submatrix containing only the atoms selected so far
            #    b is the compressed_signal
            x_hat = np.dot(np.transpose(sub_matrix), sub_matrix)

            if x_hat.shape[0] > 1:
                x_hat = np.linalg.inv(x_hat)
            else:
                x_hat = 1/x_hat

            x_hat = np.dot(x_hat, np.transpose(sub_matrix))
            x_hat = np.dot(x_hat, compressed_signal)

            # Find the solution, b-hat, using the subset of the dictionary
            #  with the highest correlations and the reconstructed signal.
            compressed_estimate = np.dot(sub_matrix, x_hat)

            # Update residual: r(k) = y - A(k) * x_hat
            residual = compressed_signal - compressed_estimate

            # Check stop conditions
            if loop_count + 1 >= stop_after_n:
                break 

            elif np.linalg.norm(compressed_estimate - compressed_signal) < error_bound:
                break

        return expand_sparse_vector(x_hat, support_set, self.dictionary.shape[1]), support_set
