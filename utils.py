import sys
import cv2
import numpy as np
import os

def database_to_table(database):
    people = None
    try:
        people = [person for person in os.listdir(database)]
    except:
        print("Ensure database is not empty right now!")
        sys.exit(1)
    
    print("Currently listed people in database: ")

    people.sort()
    for p in people:
        print("==> " + p)
    #sys.exit(0)
    
    images_table = []
    #needed_faces = 15
    num_of_faces = 0
    person_index = [None] * needed_faces
    #print "Init table" + str(np.shape(images_table)) + str(len(images_table))

    for count, p in enumerate(people):
        '''
        if count == needed_people:
            break
        '''
        '''
        if num_of_faces == needed_faces:
            print "current counter", count
            break 
        '''
        current_path = database + p
        print ("current path " + current_path + ". Counter = " + str(count))
        for image in os.listdir(current_path):
            person_index[num_of_faces] = p
            #print image
            image_data = cv2.imread(current_path + '/' + image, 0)
            #print np.shape(image_data)
            save_data = image_data.flatten()
            #print len(save_data)
            save_data = np.reshape(save_data, (1, len(save_data) ) )
            save_data = save_data[:10000]

            '''
            if count == 0:
                print "original", image_data, np.shape(image_data)
                print "******"
                print "flattened", save_data, np.shape(save_data)
                print ""
            '''

            #images_table.append(cv2.imread(current_path + '/' + image, 0))
            if len(images_table) < 1:
                images_table = save_data
            else:
                # axis=0 indicates appending along the row direction (i.e. subsequent lists added as new rows)
                images_table = np.append(images_table, save_data, axis=0) 

                #alternative way to apepend/concatenate
                #images_table = np.concatenate((images_table,save_data), axis=0)
            #num_of_faces += 1
    np.savetxt("person_index.csv", person_index, delimiter=',', fmt='%s')
    return images_table
