import numpy as np
import cv2
import sys

if len(sys.argv) != 2:
	print("Please specify using external or internal webcam!")
else:
	external = sys.argv[1]

face_cascade = cv2.CascadeClassifier('Haar_Cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('Haar_Cascades/haarcascade_eye.xml')

'''
img = cv2.imread('Sample_Images/sample_1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
'''

#detecting face from video stream
if(external == "external"):
	vid = cv2.VideoCapture(1)
elif (external == "internal"):
	vid = cv2.VideoCapture(0)

while(True):

    #capture frame by frame
    ret, frame = vid.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
    	print faces
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        '''
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
    	    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    	'''
    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    if (k & 0xFF == ord('q')) or (k & 0xFF == 27):
    	cv2.destroyAllWindows()
    	break

#release everything when done
vid.release()


'''
#display resulting images with face detected

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
cv2.imshow('img',img)
'''
'''
cv2.waitKey(0)
cv2.destroyAllWindows()
'''