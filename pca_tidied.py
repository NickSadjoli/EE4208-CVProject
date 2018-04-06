import numpy as np
import scipy.linalg as la
import os
import sys
import cv2
import pandas as pd

from utils import *

DB_FOLDER = "./database/"
AVE_FACE = "./average_face.jpg"
EIGVEC = "eigenvectors_scipy_normalized.csv"
EIGVEC_SCIPY_NORM = "./eigenvectors_scipy_normalized.csv"
normalize = False
reduced = False
mode = "None"
lib = "numpy"
scale = 1

def parse_args():
    global external
    global update
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--normalize", action='store_true', help="Choice to normalize the eigenvector or not")
    parser.add_argument("-s", "--scale", type=int, help="choose the scaling of eigenvector/eigenface value")
    parser.add_argument("-r", "--reduced", action='store_true', help="Choice to use the reduct function or not")
    #parser.add_argument("-ex", "--excam", type=str, choices=["True","False"], help = "Usage of internal or external camera. 'True' for external, and 'False' for internal")
    args = parser.parse_args()

    if args.normalize:
        normalize = True

    if args.scale:
        scale = args.scale

    if args.reduced:
        reduced = True   


def pca_reduct(table, eigenvect):
    eigenvector = pd.read_csv(eigenvect,header=None).as_matrix()

def main(database):
    image_table = database_to_table(database)
    print "Dimension of image table: ", np.shape(image_table), "\n", image_table
    np.savetxt("./image_table.csv", image_table, fmt='%i', delimiter=',')
    if reduced and normalize:
    	pca_reduct(image_table, EIGVEC)
    else:
    	pca_normal(image_table)





if __name__ == "__main__":
    parse_args()
    main(DB_FOLDER)

