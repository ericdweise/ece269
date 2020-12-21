'''
Project Submission.

This project is an implementation of Orthogonal Matching Pursuit (OMB)
for sparse signal recovery.

Class: ECE 269A, Linear Algebra
Instructor: Dr. Piya Pal
Term: Fall 2020


Write by Eric D. Weise (ericdweise@gmail.com)
'''



import os

from part_c import run_part_c
from part_d1 import run_part_d1
from part_d2 import run_part_d2
# from part_d3 import run_part_d3



def main():

    if not os.path.isdir('./plots'):
        os.mkdir('./plots')

    if not os.path.isdir('./data'):
        os.mkdir('./data')

    if not os.path.isdir('./images'):
        os.mkdir('./images')

    # print('*** Part C ***')
    # run_part_c()

    # print('*** Part D 1 ***')
    # run_part_d1()

    print('*** Part D 2 ***')
    run_part_d2()

    # print('*** Part D 3 ***')
    # run_part_d3()


if __name__ == '__main__':
    main()
