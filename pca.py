import numpy as np
import scipy.linalg as la
import os
import sys
import cv2
import pandas as pd

from utils import *


DB_FOLDER = "./database/"
AVE_FACE = "./average_face.jpg"
state = "norm"
mode = "None"
lib = "numpy"
scale = 1


def create_covariance(var_sum, num_of_faces):

    '''
    #if using the longer method (finding each element of covariance matrix one by one)
    cov = np.zeros((pix_per_face, pix_per_face))
    print np.dot(var_sum[0], var_sum[0].T)
  
    #covar[i,j] = sum( (fi - fmean)*(fj - fmean)^T ) / (n-1)
    for i in range(0, pix_per_face):
        for j in range(i, pix_per_face):
            cov[i][j] = np.dot(var_sum[i], var_sum[j].T)/(num_of_faces - 1)
            cov[j][i] = cov[i][j]
        #print "current row " + str(i)
    '''

    #much faster method, via matrix multiplication of var_sum with its own transpose.
    var_matrix = np.array(var_sum)

    #AAT covariance matrix
    cov = np.dot(var_matrix, var_matrix.T)/(num_of_faces - 1)

    return cov



def pca_other(table, average_face):
	
    col_size = np.shape(table)[1]
    var_sum = [None] * col_size
    '''
    average_face = [None] * col_size
    #var_sum = [None] * col_size
    print "size of average face: " + str(np.shape(average_face))
	
    #Get the mean vector and variance matrix from the created table
    '''

    for i in range(0,col_size):
        #mean value for the i-th pixel of a face
        average_face[i] = np.mean(table[:,i])

        #var_sum[i] = vector of the (fi-fmean) value for each ith pixel in a face
        var_sum[i] = table[:,i] - average_face[i]

    #print np.reshape(average_face, (100,100))
    #ave = np.array(np.reshape(average_face, (100,100)))
    #cv2.imwrite("average_face.jpg", ave)

    #custom implementation of covariance calculation
    cov_matrix = create_covariance(var_sum, np.shape(table)[0])

    #if using numpy's built in covariance calculation
    #cov_matrix = np.cov(table.T)

    print cov_matrix, np.shape(cov_matrix)
    #sys.exit(0)
   
    
    var_matrix = np.array(var_sum)
    L = np.dot(var_matrix.T, var_matrix)

    #find eigenvector of L
    L_eig_val, L_eig_vect = np.linalg.eigh(L, UPLO='L')


    eigenface = np.dot(var_sum, L_eig_vect)
     
    for i in range(0, col_size):
        eigenface[i] = eigenface[i]/np.linalg.norm(eigenface[i]) * 255

    #eigenface = eigenface_not_norm/np.linalg.norm(eigenface_not_norm)) * 255
    #eigenface = eigenface_not_norm 

    #print eig_val, feature_table, np.shape(eig_val), np.shape(feature_table)
    print L_eig_val, np.shape(L_eig_val)
    print "******"
    print L_eig_vect, np.shape(L_eig_vect)
    print "******"
    print eigenface, np.shape(eigenface)


    reduced = np.absolute(eigenface)
    for i in range(99,50,-1):
        cv2.imwrite("pca_test_ATA"+str(i)+".jpg", np.reshape(reduced[:,i], (100,100)) )
    #cv2.imwrite("pca_test_ATA_i.jpg", np.reshape(reduced[:,99],(100,100)))
    np.savetxt("./reduced_pca_ATA_i.csv", reduced, delimiter=',')


def pca(table):
    global state
    global lib
    global scale

    reduced = None

    col_size = np.shape(table)[1]
    #var_sum = [None] * col_size
    num_of_face = np.shape(table)[0]
    var_sum = np.zeros(np.shape(table))
    
    average_face = [None] * col_size
    #var_sum = [None] * col_size
    print "size of average face: " + str(np.shape(average_face))

    #Get the mean vector and variance matrix from the created table
    
    print np.shape(table[:,0]), np.shape(average_face[0]), np.shape(var_sum)

    #sys.exit(0)

    for i in range(0,col_size):
        #mean value for the i-th pixel of a face
        average_face[i] = np.mean(table[:,i])

        #var_sum[:,i] = vector of the (fi-fmean) value for each ith pixel in a face
        var_sum[:,i] = table[:,i] - average_face[i]

    #print np.reshape(average_face, (100,100))
    ave = np.array(np.reshape(average_face, (100,100)))
    cv2.imwrite("average_face.jpg", ave)

    #custom implementation of covariance calculation
    cov_matrix = create_covariance(var_sum.T, np.shape(table)[0])

    #if using numpy's built in covariance calculation
    #cov_matrix = np.cov(table.T)

    print cov_matrix, np.shape(cov_matrix), np.shape(var_sum) #var_sum
    #sys.exit(0)
    
    
    #calculate eigenvalue and eigenvectors, veerrry slooowwww

    if (lib == "numpy"):
        print "Using numpy"
        #using numpy eigh, finds ALL possible eigenvalues and eigenvectors
        eig_val, feature_table = np.linalg.eigh(cov_matrix, UPLO='L')
        sort_index = np.argsort(eig_val)
        feature_sorted = np.copy(feature_table[:, sort_index])[:, col_size-200-1:col_size-1] * scale
        #feature_sorted = feature_sorted[::-1]

        feature_normalized = np.copy(feature_sorted) / scale
        for i in range(0, col_size):
            feature_normalized[i] = feature_normalized[i]/np.linalg.norm(feature_normalized[i]) * scale

        print np.shape(feature_sorted)
        np.savetxt("./eigenvectors.csv", feature_sorted, delimiter=',')
        np.savetxt("./eigenvectors_normalized.csv",feature_normalized,delimiter=',')
        for i in range(199,150,-1):
            cv2.imwrite("pca_eigenface/pca_eigenface_"+str(i)+".jpg", np.reshape(feature_sorted[:,i], (100,100)) )
            cv2.imwrite("pca_eigenface/pca_eigenface_norm_"+str(i)+".jpg",np.reshape(feature_normalized[:,i],(100,100)) )

        
        init_var = np.array(var_sum)
        
        #reduced = np.dot(init_var, feature_sorted)
        if state == "norm":
            reduced = np.zeros((np.shape(init_var)[0], np.shape(feature_normalized)[1]))
            for i in range(0,num_of_face):
            	reduced[i] = np.dot(init_var[i], feature_normalized)
        
        elif state == "no_norm":
            reduced = np.zeros((np.shape(init_var)[0], np.shape(feature_sorted)[1]))
            for i in range(0,num_of_face):
            	reduced[i] = np.dot(init_var[i], feature_sorted)
        print np.shape(reduced), reduced

    else:
        print "Using scipy"
        #only find first 200 largest eigenvectors (grouped together to become one big feature table) and eigenvalues
        #eig_val, feature_table = la.eigh(cov_matrix, eigvals=(0,199))#(col_size-200,col_size-1))
        
        eig_val, feature_table = la.eigh(cov_matrix, eigvals=(col_size-200,col_size-1))
        feature_sorted = np.copy(feature_table)
        feature_sorted = feature_sorted * scale

        feature_normalized = np.copy(feature_table)
        for i in range(0, col_size):
            feature_normalized[i] = feature_normalized[i]/np.linalg.norm(feature_normalized[i]) * scale
        feature_normalized = np.absolute(feature_normalized)
        #feature_sorted = np.copy(feature_table) * 255

        print np.shape(feature_sorted)
        np.savetxt("./eigenvectors_scipy.csv", feature_sorted, delimiter=",")
        np.savetxt("./eigenvectors_scipy_normalized.csv", feature_normalized, delimiter=",")
        for i in range(199,150,-1):
            cv2.imwrite("pca_eigenface_scipy/pca_eigenface_scipy_"+str(i)+".jpg", np.reshape(feature_sorted[:,i], (100,100)) )
            cv2.imwrite("pca_eigenface_scipy/pca_eigenface_scipy_normalized_"+str(i)+".jpg", np.reshape(feature_normalized[:,i], (100,100)) )
        
        init_var = np.array(var_sum)
        
        #reduced = np.dot(init_var, feature_sorted)
        #reduced = np.dot(init_var, feature_normalized)
        if state == "norm":
            reduced = np.zeros((np.shape(init_var)[0], np.shape(feature_normalized)[1]))
            for i in range(0,num_of_face):
                reduced[i] = np.dot(init_var[i], feature_normalized)
        elif state == "no_norm":
            reduced = np.zeros((np.shape(init_var)[0], np.shape(feature_sorted)[1]))
            for i in range(0,num_of_face):
                reduced[i] = np.dot(init_var[i], feature_sorted)
        print np.shape(reduced), reduced
        

    '''
    for i in range(99,50,-1):
    	cv2.imwrite("pca_eigenface_scipy/pca_reduced_"+str(i)+".jpg", np.reshape(reduced[i],(20,10)))

    #for i in range(0,50,1):
    #	cv2.imwrite("pca_eigenface_scipy/pca_reduced_"+str(i)+".jpg", np.reshape(reduced[i],(20,10)))
    np.savetxt("./pca_eigenface_scipy/reduced_pca_i.csv", reduced, delimiter=',')
    '''



    '''
    reduced = [None] * 200
    new_average = [None] * col_size
    new_var_sum = [None] * col_size
    #reduce by dotting each column in var_sum with obtained eigenvectors
    for i in range(0,col_size):
    	new_average[i] = np.mean(feature_table[i])
    	new_var_sum = feature_table[i] - new_average[i]
        reduced[i] = np.dot(var_sum[i])
    '''
    
    '''
    #eigenfaces with L (ATA) calculation
    #reduced = feature_table * 255
    reduced = np.absolute(eigenface)
    for i in range(99,50,-1):
        cv2.imwrite("pca_test_ATA"+str(i)+".jpg", np.reshape(reduced[:,i], (100,100)) )
    #cv2.imwrite("pca_test_ATA_i.jpg", np.reshape(reduced[:,99],(100,100)))
    np.savetxt("./reduced_pca_ATA_i.csv", reduced, delimiter=',')
    '''

    return reduced



def norm_cust(matrix):
    result = 0
    for i in range(0,len(matrix)):
        result += (matrix[:,i])**2
    #print np.shape(matrix1), np.shape(matrix2)

    return math.sqrt(result)

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

def reduct(table, average_face):
    col_size = np.shape(table)[1]
    #var_sum = [None] * col_size
    var_sum = np.zeros(np.shape(table))
    '''
    average_face = [None] * col_size
    #var_sum = [None] * col_size
    print "size of average face: " + str(np.shape(average_face))

    #Get the mean vector and variance matrix from the created table
    '''
    print np.shape(table[:,0]), np.shape(average_face[0]), np.shape(var_sum)

    #sys.exit(0)

    for i in range(0,col_size):
        #mean value for the i-th pixel of a face
        #average_face[i] = np.mean(table[:,i])

        #var_sum[:,i] = vector of the (fi-fmean) value for each ith pixel in a face
        var_sum[:,i] = table[:,i] - average_face[i]
    init_var = np.array(var_sum)
    eigenvectors = pd.read_csv("./eigenvectors_scipy.csv",header=None).as_matrix()
    #print np.shape(init_var)
    #sys.exit(0)
    #reduced = np.dot(init_var, eigenvectors)
    reduced = np.zeros((np.shape(init_var)[0], np.shape(eigenvectors)[1]))
    num_of_face = np.shape(reduced)[0]
    for i in range(0,num_of_face):
    	reduced[i] = np.dot(init_var[i], eigenvectors)
    return reduced

'''
def database_to_table(database):
    people = None
    try:
        people = [person for person in os.listdir(database)]
    except:
        print("Ensure database is not empty right now!")
        sys.exit(1)
    
    print("Currently listed people in database: ")
    for p in people:
        print("==> " + p)
    
    images_table = []
    needed_faces = 100
    num_of_faces = 0
    #print "Init table" + str(np.shape(images_table)) + str(len(images_table))

    for count, p in enumerate(people):
        
        if count == needed_people:
            break
        

        if num_of_faces == needed_faces:
            break 
        current_path = database + p
        print ("current path " + current_path + ". Counter = " + str(count))
        for image in os.listdir(current_path):
            #print image
            image_data = cv2.imread(current_path + '/' + image, 0)
            save_data = image_data.flatten()
            save_data = np.reshape(save_data, (1, len(save_data) ) )

            
            if count == 0:
                print "original", image_data, np.shape(image_data)
                print "******"
                print "flattened", save_data, np.shape(save_data)
                print ""
            

            #images_table.append(cv2.imread(current_path + '/' + image, 0))
            if len(images_table) < 1:
                images_table = save_data
            else:
                # axis=0 indicates appending along the row direction (i.e. subsequent lists added as new rows)
                images_table = np.append(images_table, save_data, axis=0) 

                #alternative way to apepend/concatenate
                #images_table = np.concatenate((images_table,save_data), axis=0)
            num_of_faces += 1

    return images_table
'''
def main(database):
    global state
    global mode
    global lib
    image_table = database_to_table(database)

    print "Final dimension of table " + str(np.shape(image_table))
    print image_table
    np.savetxt("./image_table.csv", image_table, fmt='%i',  delimiter=',')
    '''
    average_face = cv2.imread(AVE_FACE).flatten()
    average_face = np.reshape(average_face, (len(average_face),1) )
    '''
    #average_face = pd.read_csv(AVE_FACE, header=None).as_matrix()
    #average_face = cv2.imread(AVE_FACE,0).flatten()
    #print np.shape(average_face), average_face[0]
    #sys.exit(0)
    #reduced = pca(image_table, average_face)
    if mode == "reduct":
    	average_face = cv2.imread(AVE_FACE,0).flatten()
    	reduced = reduct(image_table, average_face)
    elif mode == "None":
    	reduced = pca(image_table)
    #reduced = reduct(image_table, average_face)
    if state == "no_norm":
    	if mode == "None":
    		if lib == "numpy":
    			np.savetxt("./pca_eigenface/reduced_final_no_norm.csv", reduced, delimiter=',', fmt='%.2f')
    		elif lib == "scipy":
    			np.savetxt("./pca_eigenface_scipy/reduced_final_no_norm.csv", reduced, delimiter=',', fmt='%.2f')
    	elif mode == "reduct":
    		if lib == "numpy":
    			np.savetxt("./pca_eigenface/reduced_final_no_norm_i.csv", reduced, delimiter=',', fmt='%.2f')
    		elif lib == "scipy":
    			np.savetxt("./pca_eigenface_scipy/reduced_final_no_norm_i.csv", reduced, delimiter=',', fmt='%.2f')
    elif state == "norm":
    	if mode == "None":
	    	if lib == "numpy":
	    		np.savetxt("./pca_eigenface/reduced_final_norm.csv", reduced, delimiter=',', fmt='%.2f')
	    	elif lib == "scipy":
	    		np.savetxt("./pca_eigenface_scipy/reduced_final_norm.csv", reduced, delimiter=',', fmt='%.2f')
    	if mode == "reduct":
    		if lib == "numpy":
    			np.savetxt("./pca_eigenface/reduced_final_norm_i.csv", reduced, delimiter=',', fmt='%.2f')
    		elif lib == "scipy":
    			np.savetxt("./pca_eigenface_scipy/reduced_final_norm_i.csv", reduced, delimiter=',', fmt='%.2f')
    


if __name__ == '__main__':
    state = sys.argv[1]
    mode = sys.argv[2]
    lib = sys.argv[3]
    scale = int(sys.argv[4])
    main(DB_FOLDER)