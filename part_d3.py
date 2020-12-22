'''
Project Submission.

This project is an implementation of Orthogonal Matching Pursuit (OMB)
for sparse signal recovery.

Class: ECE 269A, Linear Algebra
Instructor: Dr. Piya Pal
Term: Fall 2020


Write by Eric D. Weise (ericdweise@gmail.com)
'''



from PIL import Image
from math import floor
from scipy.fftpack import dct
from scipy.fftpack import idct
from math import floor

import numpy as np
import pywt

from omp import OmpSolver




def to_fourier(arr):
    '''
    Convert 8x8 array in image space to 1x8 array in dct space
    '''
    assert(arr.shape == (8,8))
    coeffs = dct(arr)
    coeffs = np.reshape(coeffs, (64,))
    coeffs = coeffs / 16
    return coeffs


def inverse_fourier(coeffs):
    '''
    Convert 1x64 array in dct basis to 8x8 array in image space.
    '''
    assert(coeffs.shape == (64,))
    coeffs = np.reshape(coeffs, (8,8))
    return idct(coeffs)


def img_to_dct(img):
    '''convert image into dct space.
    
    The output is an array that has 64 rows and m*n/64 columnss'''
    dct_img = np.zeros((64, int(img.shape[0]*img.shape[1]/64)))
    J = 0
    for i in range(0, img.shape[0]-7, 8):
        for j in range(0, img.shape[1]-7, 8):
            row = to_fourier(img[np.ix_(range(i,i+8),range(j,j+8))])
            dct_img[:, J] = row
            J += 1

    return dct_img


def dct_to_img(img_dct, shape):
    '''Convert a 64xN matrix into a matrix with dimensions in tuple shape
    ''' 
    assert(img_dct.shape[0]*img_dct.shape[1] == shape[0]*shape[1])

    img = np.zeros(shape)

    J = 0
    for top in range(0, shape[0]-7, 8):
        for left in range(0, shape[1]-7, 8):
            im_8x8 = inverse_fourier(img_dct[:,J])

            for i in range(8):
                for j in range(8):
                    img[top + i, left + j] = im_8x8[i,j]

            J += 1

    return img



def load_image(filepath):

    # Open image file
    img = Image.open(filepath)

    numpydata = np.asarray(img)

    # Convert RGB to greyscale
    if len(numpydata.shape) == 3:
        numpydata = numpydata[:,:,0]

    # convert image file to 1D array
    imshape = numpydata.shape
    img_array = np.reshape(numpydata, imshape[0]*imshape[1])

    return img_array, imshape


def save_image(array, shape, path):
    array2d = np.reshape(array, shape)
    image = Image.fromarray(array2d)
    image = image.convert('RGB')
    image.save(path)


def omb_image(img_array, M):
    N = img_array.shape[0]
    omp_solver = OmpSolver(M, N)
    print('            dictionary made')

    compressed_image = np.dot(omp_solver.dictionary, img_array)
    print('            sparse image made')

    reconstructed_image, index_set = omp_solver.decompress(compressed_image)
    print('            reconstructed image made')

    return reconstructed_image


def helper(filepath):
    print(f'Reconstucting image {filepath}')

    img = Image.open(filepath)
    arr = np.asarray(img)
    if len(arr.shape) == 3:
        arr = arr[:,:,0]

    img_arr = img_to_dct(arr) # 64 rows

    N = img_arr.shape[1]
    M = 100
    print(f'  matrix: {M}x{N}')

    osolver = OmpSolver(64, img_arr.shape[1])
    recov_dct = np.zeros(img_arr.shape)

    for k in range(img_arr.shape[0]):
        y = osolver.compress(img_arr[k,:])
        x_recov, _ = osolver.decompress(y)
        recov_dct[k,:] = x_recov

    recov_array = dct_to_img(recov_dct, img_arr.shape)
    image2 = Image.fromarray(recov_array)
    image2 = image2.convert('RGB')
    image2.save(filepath.replace('.png', f'-recovered_{M}_{N}.png'))


def helper2(filepath):
    print(f'Reconstucting image {filepath}')
    img = Image.open(filepath)
    img_arr = np.asarray(img)
    if len(img_arr.shape) == 3:
        img_arr = img_arr[:,:,0]

    N = img_arr.shape[0]

    maxM = img_arr.shape[1]
    minM = floor(0.75*maxM)
    print(f'  minimum M: {minM}')

    for M in range(maxM-10, minM, -10):
        print(f'    M: {M}')
        img_recov = np.zeros(img_arr.shape)
        omp_solver = OmpSolver(M, N)

        # Iterate over columns
        for k in range(img_arr.shape[1]):
            x = img_arr[:,k]
            y = omp_solver.compress(x)

            x_recov, _ = omp_solver.decompress(y)

            img_recov[:,k] = x_recov

        img2 = Image.fromarray(img_recov)
        image2 = img2.convert('RGB')
        image2.save(image2, filepath.replace('.png', f'-recovered-{M}.png'))


def test():

    # Test 
    x1 = np.ones((8,8))
    w = to_fourier(x1)
    y = inverse_fourier(w)
    print(x1)
    print(y)
    assert(np.array_equal(x1, y))

    print('-'*50)

    x2 = np.zeros((8,8))
    for i in range(8):
        for j in range(8):
            x2[i,j] = i+j
    w = to_fourier(x2)
    y = inverse_fourier(w)
    print(x2)
    print(y)
    assert(np.allclose(x2, y))

    print('-'*50)

    # Test converting images to and from fourier space
    A = np.zeros((16,16))

    for I in range(16):
        for J in range(16):
            if I > 7:
                i = I-8
            else:
                i = I
            if J > 7:
                 j = J-8
            else:
                j = J
            if (I<8) & (J>8):
                A[I,J] = x2[i,j]
            else:
                A[I,J] = x1[i,j]
    Adct = img_to_dct(A)
    Arecov = dct_to_img(Adct, A.shape)
    print(A)
    print(Arecov)
    assert(np.allclose(A, Arecov))


def run_part_d3():
    # helper2('images/arch.png')
    # helper2('images/elephant.png')
    # helper2('images/koala.png')
    helper('images/spiral.png')

if __name__ == '__main__':
    # run_part_d3()
    test()
