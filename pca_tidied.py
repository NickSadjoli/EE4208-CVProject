'''
Authors: Nicholas Sadjoli and Zhen Yu

Description: Library script containing variations of the database training used for 
             PCA reduction of image database in the CV system
'''


import numpy as np
import scipy.linalg as la
import os
import sys
import cv2
import pandas as pd
import argparse

from utils import *

DB_FOLDER = "./database/"
AVE_FACE = "./average_face.jpg"
EIGVEC = "eigenvectors.csv"
EIGVEC_NORM = "eigenvectors_norm.csv"
EIGVEC_SCIPY = "eigenvectors_scipy.csv"
EIGVEC_SCIPY_NORM = "./eigenvectors_scipy_normalized.csv"
normalize = False
reduced = False
lib = "numpy"
scale = 1



def parse_args():
    global external
    global update
    global lib
    global scale
    global normalize
    global reduced

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--normalize", action='store_true', help="Choice to normalize the eigenvector or not")
    parser.add_argument("-s", "--scale", type=int, help="choose the scaling of eigenvector/eigenface value")
    parser.add_argument("-r", "--reduced", action='store_true', help="Choice to use the reduct function or not")
    parser.add_argument("-l", "--library", choices=["numpy", "scipy"], help="Which library to use")
    #parser.add_argument("-ex", "--excam", type=str, choices=["True","False"], help = "Usage of internal or external camera. 'True' for external, and 'False' for internal")
    args = parser.parse_args()

    if args.normalize:
        normalize = True

    if args.scale:
        scale = args.scale

    if args.reduced:
        reduced = True

    if args.library:
        lib = args.library




def get_eigenvector_numpy(mean_deducted, norm=False):
    global scale

    print "Using numpy"
    cols = np.shape(mean_deducted)[1]
    cov = np.dot(mean_deducted.T , mean_deducted)/(np.shape(mean_deducted)[0] - 1)

    eigvalue, eigenvectors = np.linalg.eigh(cov, UPLO='L')

    sorted_index = np.argsort(eigvalue)
    eigenvectors_sorted = np.copy(eigenvectors[:, sorted_index])[:, cols-200-1:cols-1]
    eigenvectors_fin = np.copy(eigenvectors_sorted) * scale
    eigenvectors_normalized = np.copy(eigenvectors_sorted)
    for i in range(0, cols):
        eigenvectors_normalized[i] = eigenvectors_normalized[i]/np.linalg.norm(eigenvectors_normalized[i]) * scale

    np.savetxt(EIGVEC, eigenvectors_fin, delimiter=',')
    np.savetxt(EIGVEC_NORM, eigenvectors_normalized,delimiter=',')
    for i in range(199,150,-1):
        cv2.imwrite("pca_eigenface/pca_eigenface_"+str(i)+".jpg", np.reshape(eigenvectors_fin[:,i], (100,100)) )
        cv2.imwrite("pca_eigenface/pca_eigenface_norm_"+str(i)+".jpg",np.reshape(eigenvectors_normalized[:,i],(100,100)) )
    
    if norm:
        return eigenvectors_normalized
    return eigenvectors_fin




def get_eigenvector_scipy(mean_deducted, norm=False):
    global scale

    print "Using scipy"
    cols = np.shape(mean_deducted)[1]
    cov = np.dot(mean_deducted.T , mean_deducted)/(np.shape(mean_deducted)[0] - 1)

    _, eigenvectors = la.eigh(cov, eigvals=(cols-200,cols-1))

    eigenvectors_sorted = np.copy(eigenvectors)
    eigenvectors_normalized = np.copy(eigenvectors_sorted)
    for i in range(0, cols):
            eigenvectors_normalized[i] = eigenvectors_normalized[i]/np.linalg.norm(eigenvectors_normalized[i]) * scale

    np.savetxt(EIGVEC_SCIPY, eigenvectors_sorted, delimiter=',')
    np.savetxt(EIGVEC_SCIPY_NORM, eigenvectors_normalized,delimiter=',')
    for i in range(199,150,-1):
        cv2.imwrite("pca_eigenface_scipy/pca_eigenface_scipy"+str(i)+".jpg", np.reshape(eigenvectors_sorted[:,i], (100,100)) )
        cv2.imwrite("pca_eigenface_scipy/pca_eigenface_scipy_norm_"+str(i)+".jpg",np.reshape(eigenvectors_normalized[:,i],(100,100)) )
    
    if norm:
        return eigenvectors_normalized
    return eigenvectors_sorted




def pca_normal(table):
    '''
    Reduce input face matrix without any outer inputs. Also calculates average face and eigenvector.

    Input: table - input face matrix to be reduced

    Output: reduced version of the input face matrix
    '''
    global normalize
    global lib

    col_size = np.shape(table)[1]
    faces = np.shape(table)[0]

    average = [None] * col_size

    #calculate average values of faces, and the matrix deducted from it
    mean_deducted = np.zeros(np.shape(table))
    for i in range(0, col_size):
        average[i] = np.mean(table[:,i])
        mean_deducted[:,i] = table[:,i] - average[i]

    #write out resulting average face values
    ave = np.array(np.reshape(average, (100,100)))
    cv2.imwrite("average_face.jpg", ave)

    #calculate the eigenvectors for the mean deducted matrix
    eigenvect = None
    if lib == "numpy":
        eigenvect = get_eigenvector_numpy(mean_deducted, normalize)
    elif lib == "scipy":
        eigenvect = get_eigenvector_scipy(mean_deducted, normalize)
    
    #calculate the reduced face matrix and return
    mean_deducted = np.array(mean_deducted)
    reduced = np.zeros((np.shape(mean_deducted)[0], np.shape(eigenvect)[1]))
    for i in range(0,faces):
        reduced[i] = np.dot(mean_deducted[i], eigenvect)

    return reduced



def pca_reduct(table, average_face, eigenvect):
    '''
    Reduce input face matrix via usage of an existing average face and eigenvectors.

    Input: table - input face matrix to be reduced
           average_face - average of faces in database so far
           eigenvect - path to eigenvector matrix used to reduce the input matrix

    Output: reduced version of the input face matrix
    '''

    #prepare the eigenvector from path given
    eigenvector = pd.read_csv(eigenvect,header=None).as_matrix()
    col_size = np.shape(table)[1]
    faces = np.shape(table)[0]

    #calculate the matrix deducted from the average values of faces 
    mean_deducted = np.zeros(np.shape(table))
    for i in range(0, col_size):
        mean_deducted[:,i] = table[:,i] - average_face[i]

    #calculate the reduced face matrix and return
    mean_deducted = np.array(mean_deducted)
    reduced = np.zeros((np.shape(mean_deducted)[0], np.shape(eigenvector)[1]))
    for i in range(0,faces):
        reduced[i] = np.dot(mean_deducted[i], eigenvector)

    return reduced



def pca_live(table, average_face, eigenvectors):
    col_size = np.shape(table)[1]
    num_of_face = np.shape(table)[0]
    #var_sum = [None] * col_size
    var_sum = np.zeros(np.shape(table))

    for i in range(0,col_size):
        #var_sum[:,i] = vector of the (fi-fmean) value for each ith pixel in a face
        var_sum[:,i] = table[:,i] - average_face[i]
    
    var_array = np.array(var_sum)
    #reduced = np.zeroes((,col_size))
    reduced = np.dot(var_array, eigenvectors)
    #print np.shape(reduced), reduced

    return reduced



def main(database):
    image_table = database_to_table(database)
    print "Dimension of image table: ", np.shape(image_table), "\n", image_table
    np.savetxt("./image_table.csv", image_table, fmt='%i', delimiter=',')
    result = None
    if reduced:
        print "Reducing"
        #sys.exit(0)
        average_face = cv2.imread(AVE_FACE,0).flatten()
        if lib == "numpy":
            if normalize:
                result = pca_reduct(image_table, average_face, EIGVEC_NORM)
                np.savetxt("./pca_eigenface/reduced_final_norm_i.csv", result, delimiter=',', fmt='%.2f')
            else:
                result = pca_reduct(image_table, average_face, EIGVEC)
                np.savetxt("./pca_eigenface/reduced_final_no_norm_i.csv", result, delimiter=',', fmt='%.2f')
        elif lib == "scipy":
            if normalize:
                result = pca_reduct(image_table,average_face, EIGVEC_SCIPY_NORM)
                np.savetxt("./pca_eigenface_scipy/reduced_final_norm_i.csv", result, delimiter=',', fmt='%.2f')
            else:
                result = pca_reduct(image_table, average_face, EIGVEC_SCIPY)
                np.savetxt("./pca_eigenface_scipy/reduced_final_no_norm_i.csv", result, delimiter=',', fmt='%.2f')


    else:
        print "Normal pca"
        #sys.exit(0)
    	result = pca_normal(image_table)
        if normalize:
            if lib == "numpy":
                np.savetxt("./pca_eigenface/reduced_final_norm.csv", result, delimiter=',', fmt='%.2f')
            elif lib == "scipy":
                np.savetxt("./pca_eigenface_scipy/reduced_final_norm.csv", result, delimiter=',', fmt='%.2f')
        else:
            if lib == "numpy":
                np.savetxt("./pca_eigenface/reduced_final_no_norm.csv", result, delimiter=',', fmt='%.2f')
            elif lib == "scipy":
                np.savetxt("./pca_eigenface_scipy/reduced_final_no_norm.csv", result, delimiter=',', fmt='%.2f')        





if __name__ == "__main__":
    parse_args()
    print reduced, normalize, lib, scale
    #sys.exit(0)
    main(DB_FOLDER)

