import numpy as np
import cv2
import sys
import os
import shutil
import argparse
import threading
import time
import pandas as pd
import math

from pca import pca_live

update = False
external = False
update_counter = 0
counter_flag = True

RED_PCA = "./pca_eigenface_scipy/reduced_final_no_norm_i.csv"
AV_FACE = "./average_face.jpg"
EIGENVECTOR = "./eigenvectors_scipy.csv"
PEOPLE_INDEX = "./person_index.csv"


#function to parse incoming arguments from user
def parse_arguments():
    global external
    global update
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="Action that you want to do with the program. Options are to 'update' database or to 'recognize' faces")
    parser.add_argument("-c", "--camera", type=int, choices=[0,1], help="Use external or internal webcam. '0' is for internal, and '1' is for external")
    #parser.add_argument("-ex", "--excam", type=str, choices=["True","False"], help = "Usage of internal or external camera. 'True' for external, and 'False' for internal")
    args = parser.parse_args()
    if (args.mode == "update") or (args.mode == "recognize"):
        if args.mode == "update":
            update = True
        elif args.mode == "recognize":
            update = False
    else:
        print("Invalid mode inputted. Please only use program to either 'update' database or 'recognize' faces.")
        sys.exit(1)


    #version that uses integer options
    if args.camera == 1:
        external = True
    else:
        external = False     
    

#main loop for the database updating function
def update_loop(dir_path, photo_count, name):
    global external
    if external:
        cam = cv2.VideoCapture(1)
    else:
        cam = cv2.VideoCapture(0)
    raw_input("Taking "+ str(photo_count) + " pictures. Press ENTER when ready")
    os.mkdir(dir_path)
    count = 0
    '''
    print("made path")
    sys.exit(0)
    '''
    timer = 0
    while(count < photo_count):

        #read a frame from the camera in grayscale
        _,frame = cam.read()
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        #prepare face detector/cascade
        face_cascade = cv2.CascadeClassifier('Haar_Cascades/haarcascade_frontalface_default.xml')

        #returns coordinate, width and height of each detected faces in frame (x,y,w,h)
        faces_coordinate = face_cascade.detectMultiScale(frame, 1.3, 5)

        face_rect = []

        cv2.namedWindow('Feed', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Inputted Face', cv2.WINDOW_NORMAL)

        #collect actual faces from the frame
        for (x,y,w,h) in faces_coordinate:

            #append actual faces from a frame to a list, via coordinate and width/height adjustment
            face_rect.append(frame[y : y + h, x : x + w])

            #draw rectangles around each detected faces
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        #if already at interval,
        if ((timer % 100) == 10) and len(face_rect):

            norm_size = (100,100)
            final_faces = []

            #resize all faces to be 100 x 100
            for face in face_rect:
                face_normalized = None
                if face.shape < norm_size:
                    face_normalized = cv2.resize(face, norm_size, interpolation=cv2.INTER_AREA)
                else:
                    face_normalized = cv2.resize(face, norm_size, interpolation=cv2.INTER_CUBIC)
                final_faces.append(face_normalized)

            #write the face(s) into a jpg file
            #print final_faces
            print np.shape(final_faces[0])
            cv2.imwrite(dir_path + '/' + name + '_' + str(count+1)+'.jpg', final_faces[0])
            cv2.imshow('Inputted Face', final_faces[0])
            count +=1 

        cv2.imshow('Feed', frame)
        cv2.waitKey(50)

        timer += 10

'''
def counter_timer():
    global update_counter
    global counter_flag

    while(counter_flag):
        update_counter = time.millis()
'''


#function to update the database folder
def update_database(folder):
    global external
    face_cascade = cv2.CascadeClassifier('Haar_Cascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('Haar_Cascades/haarcascade_eye.xml')

    photo_count = None

    while(type(photo_count) != int):
        photo_count = int(input("Please indicate how many pictures would like to be taken for database => "))

    name = raw_input("Please enter your name: ").lower()

    count = 0

    #create path with folder + name
    dir_path = folder+name 

    #if path does not exist yet
    if not os.path.exists(dir_path):
        update_loop(dir_path, photo_count, name)


    #if path already exists, then no need to take a picture of user anymore
    else:
        choice = ["Yes", "Y", "y", "No", "N", "n"]
        ans = None
        while ans not in choice:
            ans = raw_input("You are already registered in the database! Do you want to overwrite and re-do face input? => ")
        if ans in ["Yes", "Y", "y"]:
            #remove the name directory, and then do the update_loop
            shutil.rmtree(dir_path)
            update_loop(dir_path, photo_count, name)
        elif ans in ["No", "N", "n"]:
            print("Exiting...")
            sys.exit(0)
        sys.exit(0)


#NCC process for recognizing an incoming face 
def ncc(face, database_pca, average_face, eigenvectors, pp_index):
    face_flat = face.flatten()
    face_flat = np.reshape(face_flat, (1,len(face_flat)) )
    #print np.shape(face_flattened)
    #sys.exit(0)
    reduced_face = pca_live(face_flat, average_face, eigenvectors)
    #print np.shape(reduced_face), np.shape(database_pca)[0]
    #sys.exit(0)
    num_of_face = np.shape(database_pca)[0]
    minimum = None
    face_id = 0
    for i in range(0, num_of_face):
        norm_diff = np.linalg.norm(reduced_face - database_pca[i])
        #norm_diff = norm_cust(reduced_face, database_pca[i])
        #print norm_diff, i, " ** "
        #print reduced_face
        #print database_pca[i]
        print " "
        if minimum is None:
            minimum = norm_diff
        elif norm_diff < minimum:
            minimum = norm_diff
            face_id = i
    #print face_id
    #sys.exit(0)
    print face_id, ""
    return pp_index[face_id][0]+","+str(face_id)
    #sys.exit(0)

def norm_cust(matrix1, matrix2):
    result = 0
    for i in range(0,len(matrix1)):
        result += (matrix1[:,i] - matrix2[i])**2
    #print np.shape(matrix1), np.shape(matrix2)

    return math.sqrt(result)

    

#function for recognizing faces
def recog_mode(database):

    try:
        people = [person for person in os.listdir(database)]
    except:
        print("Ensure database is not empty right now!")
        sys.exit(1)

    print("Currently listed people in database: ")
    people.sort()
    for p in people:
        print("==> " + p)

    pca_database = pd.read_csv(RED_PCA, header=None).as_matrix()
    #average_face = pd.read_csv("./average_face.csv", header=None).as_matrix()
    average_face = cv2.imread(AV_FACE, 0).flatten()
    eigenvectors = pd.read_csv(EIGENVECTOR, header=None).as_matrix()
    people_index = pd.read_csv(PEOPLE_INDEX, header=None).as_matrix().tolist()
    print "pca_database_size: ", np.shape(pca_database), ". Ave_face size: ", np.shape(average_face), np.shape(eigenvectors)

    face_cascade = cv2.CascadeClassifier('Haar_Cascades/haarcascade_frontalface_default.xml')
    font = cv2.FONT_HERSHEY_SIMPLEX
    #eye_cascade = cv2.CascadeClassifier('Haar_Cascades/haarcascade_eye.xml')

    #determine whether to stream from external or internal webcam
    print external
    if external:
        vid = cv2.VideoCapture(1)
    else :
        vid = cv2.VideoCapture(0)

    

    while(True):

        #capture frame by frame
        ret, frame = vid.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces_coor = face_cascade.detectMultiScale(gray, 1.3, 5)

        face_rect = []

        names = []

        cv2.namedWindow('Feed', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Inputted Face', cv2.WINDOW_NORMAL)

        norm_size = (100,100)

        #collect actual faces from the frame
        for (x,y,w,h) in faces_coor:

            #append actual faces from a frame to a list, via coordinate and width/height adjustment
            face = gray[y : y + h, x : x + w]
            #face_rect.append(frame[y : y + h, x : x + w])
            face_cut = None
            if face.shape < norm_size:
                face_cut = cv2.resize(face, norm_size, interpolation=cv2.INTER_AREA)
            else:
                face_cut = cv2.resize(face, norm_size, interpolation=cv2.INTER_CUBIC)
            #face_cut = cv2.resize(face, norm_size, interpolation=cv2.INTER_AREA)
            face_rect.append(face_cut)
            #print np.shape(face_cut), np.shape(face_cut[0,0]), np.shape(face)
            #sys.exit(0)
            name = str(ncc(face_cut, pca_database, average_face, eigenvectors, people_index))

            #draw rectangles around each detected faces
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(frame, name, (x,y), font, 0.8, (255,255,255), 2,cv2.LINE_AA)
            cv2.imshow('Feed', face_cut)

        cv2.imshow('Inputted Face', frame)
        #cv2.imshow('Feed', face_rect[0])
        k = cv2.waitKey(1)
        if (k & 0xFF == ord('q')) or (k & 0xFF == 27):
            cv2.destroyAllWindows()
            break

    #release everything when done
    vid.release()



if __name__ == '__main__':
    parse_arguments()
    DB_FOLDER = "./database/"
    if update:
        if not os.path.exists(DB_FOLDER):
            os.makedirs(DB_FOLDER)
        update_database(DB_FOLDER)
    else:
        recog_mode(DB_FOLDER)