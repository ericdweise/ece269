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


def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


def to_dct(arr):
    '''
    Convert 8x8 array in image space to 1x64 array in dct space

    The ZigZag ordering of DCT coefficients is used to preference 
    low frequency signals
    '''
    assert(arr.shape == (8,8))
    coeffs = dct2(arr)

    # Apply zig-zag ordering
    coeffs2 = np.zeros(64)
    for k in range(64):
        coeffs2[k] = coeffs[zzOrd[k][0], zzOrd[k][1]]

    return coeffs2


def from_dct(coeffs):
    '''
    Convert 1x64 array in dct basis to 8x8 array in image space.
    '''
    assert(coeffs.shape == (64,))
    coeffs2 = np.zeros((8,8))

    for k in range(64):
        coeffs2[zzOrd[k][0], zzOrd[k][1]] = coeffs[k]
    return idct2(coeffs2)


def showstuff():
	for i in range(8):
		for j in range(8):
			a = np.zeros((8,8))
			a[i,j] = 1
			print(a)
			print(idct2(a))
			f = input('pause')

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


def make_sparse(x, s):
    y = np.sort(x)
    z = np.array([x[i] if x[i] >= y[-s] else 0 for i in range(x.shape[0])])
    return z


def run_image(filepath, noise=None):
    print(f'Reconstucting image {filepath}')

    label = filepath.replace('images/', '')
    label = label.replace('.png', '')

    original = load_image(filepath)

    if noise is not None:
        label = label + f'_noise-{noise}'
        image, _ = add_noise(original, noise)
        save_image(image, 'images/' + label + '.png')
    else:
        image = original

    N = 64
    M = 30
    Ss = [1,2,3,4,5,6,8,10,12]

    A = generate_random_matrix(M,N)

    psnr = []

    for s in Ss:
        img_recov = np.zeros(image.shape)

        for I in range(0, image.shape[0]-7, 8):
            for J in range(0, image.shape[1]-7, 8):
                subimage = image[I:I+8, J:J+8]

                x = to_dct(subimage)
                xs = make_sparse(x, s)
                y = np.dot(A, xs)

                x_recovered, _ = omp(A, y, error_bound=0.001)

                img_recov[I:I+8, J:J+8] = from_dct(x_recovered)

        save_image(img_recov, f'images/{label}' + f'-recovered_{s:02d}.png')

        # Convert from float64 to uit8
        img_recov = img_recov / img_recov.max()
        img_recov = img_recov * 255
        img_recov = img_recov.astype('uint8')

        psnr.append(peak_snr(original, img_recov))

    return Ss, psnr


def test():

    # Test transformations to/from Fourier space
    x1 = np.ones((8,8))
    w = to_dct(x1)
    y = from_dct(w)
    print(x1)
    print(y)
    assert(np.allclose(x1, y))

    print('-'*50)

    x2 = np.zeros((8,8))
    for i in range(8):
        for j in range(8):
            x2[i,j] = i+j
    w = to_dct(x2)
    y = from_dct(w)
    print(x2)
    print(y)
    assert(np.allclose(x2, y))

    print('-'*50)


def run_part_d3():

    # NO NOISE
    fig = plt.figure()
    ax = plt.axes()

    ax.set_title('Image Recovery PSNR\nNo Noise')
    ax.set_ylabel('PSNR (dB)')
    ax.set_xlabel('Signal Sparsity, s')

    s, psnr = run_image('images/arch.png')
    ax.plot(s, psnr, '-m', label='arch')
    s, psnr = run_image('images/elephant.png')
    ax.plot(s, psnr, '-c', label='elephant')
    s, psnr = run_image('images/koala.png')
    ax.plot(s, psnr, '-b', label='koala')
    s, psnr = run_image('images/spiral.png')
    ax.plot(s, psnr, '-g', label='spiral')

    ax.legend()
    plt.savefig(f'./plots/d3-no-noise.png')
    plt.close()


    # And with noise variance 50
    fig = plt.figure()
    ax = plt.axes()

    ax.set_title('Image Recovery PSNR\nNoise Variance: 50')
    ax.set_ylabel('PSNR (dB)')
    ax.set_xlabel('Signal Sparsity, s')

    s, psnr = run_image('images/arch.png', 50)
    ax.plot(s, psnr, '-m', label='arch')
    s, psnr = run_image('images/elephant.png', 50)
    ax.plot(s, psnr, '-c', label='elephant')
    s, psnr = run_image('images/koala.png', 50)
    ax.plot(s, psnr, '-b', label='koala')
    s, psnr = run_image('images/spiral.png', 50)
    ax.plot(s, psnr, '-g', label='spiral')

    plt.savefig(f'./plots/d3-noise-50.png')
    plt.close()


    # And with noise variance 10
    fig = plt.figure()
    ax = plt.axes()

    ax.set_title('Image Recovery PSNR\nNoise Variance: 10')
    ax.set_ylabel('PSNR (dB)')
    ax.set_xlabel('Signal Sparsity, s')

    s, psnr = run_image('images/arch.png', 10)
    ax.plot(s, psnr, '-m', label='arch')
    s, psnr = run_image('images/elephant.png', 10)
    ax.plot(s, psnr, '-c', label='elephant')
    s, psnr = run_image('images/koala.png', 10)
    ax.plot(s, psnr, '-b', label='koala')
    s, psnr = run_image('images/spiral.png', 10)
    ax.plot(s, psnr, '-g', label='spiral')

    plt.savefig(f'./plots/d3-noise-10.png')
    plt.close()


if __name__ == '__main__':
    test()
    run_part_d3()
