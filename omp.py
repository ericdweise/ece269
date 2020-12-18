import numpy as np

from math import floor


def print_vec(v):
    s = np.array_repr(v).replace('\n','')
    print(s)


def compute_correlation(v1, v2):
    return abs(np.dot(v1, v2))


class OmpSolver(object):
    '''Calculate the Orthogonal Matching Pursuit solution
    '''

    mu = 0
    max_sparsity = 0
    dictionary = None


    def __init__(self, dictionary):
        '''Args:
            dictionary: The set of atoms. An m by n ndarray with n >> m.
            signal: The measured signal. An m by 1 ndarray.
            noise: Optional, noise. An m by 1 ndarray.
            error_bound: Bound on the error. Computation will stop when the 2-norm of the 
                         difference between the reconstructed signal and the input signal
                         is less than the error_bound. Must be a positive number.
                         Default: 10**(-3)
        '''
        self.set_dictionary(dictionary)


    def set_dictionary(self, A):
        assert(A.shape[0] < A.shape[1])
        
        self.dictionary = np.copy(A)

        # Normalize columns of A
        for j in range(self.dictionary.shape[1]):  # iterate over columns
            col_norm = np.linalg.norm(self.dictionary[:,j])
            for i in range(self.dictionary.shape[0]):  # iterate over rows
                self.dictionary[i,j] /= col_norm

        # Calculate coherence parameter
        self.calulate_coherence_parameter()


    def set_signal(self, signal):
        assert(b.shape[0] == self.dictionary.shape[0])
        self._signal = signal


    def get_max_sparsity(self):
        return self.max_sparsity


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


    def solve(self, signal, error_bound=10**(-3)):
        '''Find the OMP solution with index for the given signal.

        Args:
            signal: The "measured" signal
        '''
        ## INITIALIZE
        residual = np.copy(signal)
        support_set = []  # Indices of atoms with maximum correlation to residual
        signal_estimate = np.zeros(self.dictionary.shape[0])

        # MAIN LOOP
        for k in range(self.dictionary.shape[1]):
            max_correlation = 0
            max_correlation_index = 0

            # Find index of atom with largest correlation to residual
            for j in range(self.dictionary.shape[1]):
                correlation = compute_correlation(self.dictionary[:,j], residual)

                if correlation > max_correlation:
                    max_correlation = correlation
                    max_correlation_index = j

            # Save index to the support set
            support_set.append(max_correlation_index)

            # Create matrix with only selected atoms 
            self._sub_matrix = self.dictionary[:, support_set]

            # least squares to find reconstructed solution, x-hat
            # I implement this step by step, but the basic maths are:
            #    (A^T * A)^(-1) * A^T * b
            # where:
            #    A is the tall submatrix containing only the atoms selected so far
            #    b is the signal
            x_hat = np.dot(np.transpose(self._sub_matrix), self._sub_matrix)

            if x_hat.shape[0] > 1:
                x_hat = np.linalg.inv(x_hat)
            else:
                x_hat = 1/x_hat

            x_hat = np.dot(x_hat, np.transpose(self._sub_matrix))
            x_hat = np.dot(x_hat, signal)

            # Find the solution, b-hat, using the subset of the dictionary
            #  with the highest correlations and the reconstructed signal.
            signal_estimate = np.dot(self._sub_matrix, x_hat)

            # Calculate Normalized Error
            error = np.linalg.norm(signal - signal_estimate) / np.linalg.norm(signal)

            # update residual
            residual = signal - signal_estimate

            # print_vec(signal_estimate)

            # Check if solution is close enough to stop
            if  error < error_bound:
                break

        return signal_estimate, support_set, error
