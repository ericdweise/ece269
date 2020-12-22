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
from scipy.fftpack import dct
from scipy.fftpack import idct

import numpy as np
import matplotlib.pyplot as plt 

from tools import add_noise
from tools import omp
from tools import generate_random_matrix
from tools import mean_squared_error
from tools import peak_snr



zzOrd = {  0: (0,0),   1: (0,1),   5: (0,2),   6: (0,3),  14: (0,4),  15: (0,5),  27: (0,6),  28: (0,7),
           2: (1,0),   4: (1,1),   7: (1,2),  13: (1,3),  16: (1,4),  26: (1,5),  29: (1,6),  42: (1,7),
           3: (2,0),   8: (2,1),  12: (2,2),  17: (2,3),  25: (2,4),  30: (2,5),  41: (2,6),  43: (2,7),
           9: (3,0),  11: (3,1),  18: (3,2),  24: (3,3),  31: (3,4),  40: (3,5),  44: (3,6),  53: (3,7),
          10: (4,0),  19: (4,1),  23: (4,2),  32: (4,3),  39: (4,4),  45: (4,5),  52: (4,6),  54: (4,7),
          20: (5,0),  22: (5,1),  33: (5,2),  38: (5,3),  46: (5,4),  51: (5,5),  55: (5,6),  60: (5,7),
          21: (6,0),  34: (6,1),  37: (6,2),  47: (6,3),  50: (6,4),  56: (6,5),  59: (6,6),  61: (6,7),
          35: (7,0),  36: (7,1),  48: (7,2),  49: (7,3),  57: (7,4),  58: (7,5),  62: (7,6),  63: (7,7) }


def to_fourier(arr):
    '''
    Convert 8x8 array in image space to 1x64 array in dct space

    The ZigZag ordering of DCT coefficients is used to preference 
    low frequency signals
    '''
    assert(arr.shape == (8,8))
    coeffs = dct(arr)
    coeffs = coeffs / 16

    # Apply zig-zag ordering
    coeffs2 = np.zeros(64)
    for k in range(64):
        coeffs2[k] = coeffs[zzOrd[k][0], zzOrd[k][1]]

    return coeffs2


def inverse_fourier(coeffs):
    '''
    Convert 1x64 array in dct basis to 8x8 array in image space.
    '''
    assert(coeffs.shape == (64,))
    coeffs2 = np.zeros((8,8))

    for k in range(64):
        coeffs2[zzOrd[k][0], zzOrd[k][1]] = coeffs[k]
    return idct(coeffs2)


def img_to_dct(img):
    '''convert image into dct space.
    
    The output is an array that has 64 rows and m*n/64 columns
    '''
    dct_arr = np.zeros((64, int(img.shape[0]*img.shape[1]/64)))
    J = 0
    for i in range(0, img.shape[0]-7, 8):
        for j in range(0, img.shape[1]-7, 8):
            row = to_fourier(img[np.ix_(range(i,i+8),range(j,j+8))])
            dct_arr[:, J] = row
            J += 1

    return dct_arr


def dct_to_img(img_dct, shape):
    '''Convert a 64xN matrix into a matrix with dimensions in tuple shape
    ''' 
    # assert(img_dct.shape[0]*img_dct.shape[1] == shape[0]*shape[1])

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
    arr = np.asarray(img)

    # Convert RGB to greyscale
    if len(arr.shape) == 3:
        arr = arr[:,:,0]

    return arr


def save_image(array, path):
    image = Image.fromarray(array)
    image = image.convert('RGB')
    image.save(path)


def run_image(filepath, noise=None):
    print(f'Reconstucting image {filepath}')

    image_name = filepath.replace('images/', '')
    image_name = image_name.replace('.png', '')
    if noise is not None:
        image_name = image_name + '_noisy'

    image_array = load_image(filepath)

    dct_array = img_to_dct(image_array)

    N = 64
    Ms = range(2,64,4)

    mse = []
    psnr = []

    for M in Ms:
        A = generate_random_matrix(M,N)
        dct_recov = np.zeros(dct_array.shape)

        for i in range(dct_array.shape[1]):
            x = dct_array[:,i]
            y = np.dot(A, x)

            if noise is not None:
                y, _ = add_noise(y, noise)

            x_recovered, _ = omp(A, y, M)

            dct_recov[:,i] = x_recovered

        image_recovered = dct_to_img(dct_recov, image_array.shape)

        save_image(image_recovered, f'images/{image_name}' + f'-recovered_{M}.png')

        # Convert from float64 to uit8
        image_recovered = image_recovered / image_recovered.max()
        image_recovered = image_recovered * 255
        image_recovered = image_recovered.astype('uint8')

        mse.append(mean_squared_error(image_array, image_recovered))
        psnr.append(peak_snr(image_array, image_recovered))

    fig = plt.figure()

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title('Mean Squared Error')
    ax1.set_ylabel('MSE')
    ax1.set_xlabel('Number of Measurements, M')
    ax1.plot(Ms, mse, '-m')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title('Peak Signal to Noise Ratio')
    ax2.set_ylabel('PSNR (db)')
    ax2.set_xlabel('Number of Measurements, M')
    ax2.plot(Ms, psnr, '-c')

    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()

    plt.savefig(f'./plots/d3-{image_name}.png')

    plt.close()


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
    run_image('images/arch.png')
    run_image('images/elephant.png')
    run_image('images/koala.png')
    run_image('images/spiral.png')

    # And with noise:
    # Adding a noise with variance of 10 (out of 255) is about 2.5%
    run_image('images/arch.png', 10)
    run_image('images/elephant.png', 10)
    run_image('images/koala.png', 10)
    run_image('images/spiral.png', 10)


if __name__ == '__main__':
    test()
    run_part_d3()
