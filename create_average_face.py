'''
Script to create average face from database
'''

import numpy as np
import scipy.linalg as la
import os
import sys
import cv2
import csv

from utils import *

DB_FOLDER = "./database/"

table = database_to_table(DB_FOLDER)

col_size = np.shape(table)[1]

average_face = [None] * col_size
#var_sum = [None] * col_size
print "size of average face: " + str(np.shape(average_face))

#Get the mean vector and variance matrix from the created table
for i in range(0,col_size):
    #mean value for the i-th pixel of a face
    average_face[i] = np.mean(table[:,i])

    #var_sum[i] = vector of the (fi-fmean) value for each ith pixel in a face
    #var_sum[i] = table[:,i] - average_face[i]

#print np.reshape(average_face, (100,100))
np.savetxt("average_face.csv", average_face, delimiter = ',', fmt='%.2f')
ave = np.array(np.reshape(average_face, (100,100)))
print np.shape(ave)
cv2.imwrite("average_face.jpg", ave)


