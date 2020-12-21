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
import numpy as np
import pywt

from omp import OmpSolver



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


def helper(in_path):
    print(f'Reconstucting image {in_path}')
    image_array, image_shape = load_image(in_path)

    N = image_array.shape[0]

    # for M in range(1, int(N/10), 5):
    for M in [196,]:
        print(f'   Dictionary size: {M}x{N}')
        out_path = in_path.replace('.png', f'-reconstructed-M-{M}.png')
        reconstructed_image = omb_image(image_array, M)
        save_image(reconstructed_image, image_shape, out_path)


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
        ima2 = img2.convert('RGB')
        ima2.save(filepath.replace('.png', f'-recovered-{M}.png'))


def test():
    im_shape = (3,4)
    orig = np.ones(im_shape)
    orig_arr = np.reshape(orig, (12,))
    im_array = np.reshape(orig_arr, im_shape[0]*im_shape[1])
    im_recon = omb_image(im_array, 8)
    im_recon_2d = np.reshape(im_recon, im_shape)
    print(orig)
    print(im_recon_2d)


def run_part_d3():
    helper2('images/arch.png')
    helper2('images/elephant.png')
    helper2('images/koala.png')
    helper2('images/spiral.png')

if __name__ == '__main__':
    run_part_d3()
