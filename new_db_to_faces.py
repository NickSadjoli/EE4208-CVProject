import cv2
import numpy as np
import sys
import os

NEW_DB = "./New_Face_DB_i/Set1_Greyscale/"

TEMP_DB = "./Translated_new/"

people = None
print os.listdir(NEW_DB)
#sys.exit(0)
try:
    people = [person for person in os.listdir(NEW_DB)]
except:
    print("Ensure database is not empty right now!")
    sys.exit(1)

print("Currently listed people in database: ")

people.sort()
for p in people:
    print("==> " + p)
#sys.exit(0)

images_table = []
needed_faces = 100
num_of_faces = 0
person_index = [None] * needed_faces
#print "Init table" + str(np.shape(images_table)) + str(len(images_table))

for count, p in enumerate(people):
    current_path = NEW_DB + p
    print ("current path " + current_path + ". Counter = " + str(count))
    for image in os.listdir(current_path):
        person_index[num_of_faces] = p
        #print image
        image_data = cv2.imread(current_path + '/' + image, 0)

        cv2.imwrite(TEMP_DB + image.split(".")[0]+".jpg", image_data)

