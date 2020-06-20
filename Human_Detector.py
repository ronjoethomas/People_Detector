'''This program is the human face detector. The purpose is to identify human faces in a camera view.
    If the amount of faces is the frame at a time is over a parameter defined in 'people_limit', there
    will be an on screen warning indicating there are too many people. '''

import cv2
import numpy as np
import os
import time

dir_path = os.path.dirname(os.path.realpath(__file__))

print(dir_path+ '/cascade_models/haarcascade_frontalcatface_extended.xml')

#set up the classifiers for front and profile face
face_front_cascade = cv2.CascadeClassifier(dir_path + '/cascade_models/haarcascade_frontalface_alt.xml')
face_profile_cascade = cv2.CascadeClassifier(dir_path + '/cascade_models/haarcascade_profileface.xml')

video = cv2.VideoCapture(0)

#define the people limit here
people_limit = 1

#constantly read the video
while True:

    check, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces_front = face_front_cascade.detectMultiScale(gray,1.3,5)
    faces_profile = face_profile_cascade.detectMultiScale(gray,1.3,5)
    
    e=0
    #identifying the front faces
    for x,y,w,h in faces_front:
        cv2.rectangle(frame, (x ,y), (x+w,y+h), (12,255,22),4 )
        cv2.putText(frame,"Human #"+str(e+1),(x,y-int(y/15)),cv2.FONT_HERSHEY_COMPLEX,0.5,(140,140,0),1)
    
    #identifying the profile faces
    e=0
    for x,y,w,h in faces_profile:
        cv2.rectangle(frame, (x,y),(x+w,y+h), (12,255,22),4)
        cv2.putText(frame,'Human #'+str(e+len(faces_front)+1),(x,y-int(y/15)),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,140),1)
    

    #create warning if the amount of faces is greater than a certain number
    if(len(faces_front)+len(faces_profile)>people_limit):
        x,y = int(frame.shape[1]/14),int(frame.shape[0]/14)
        cv2.putText(frame,"WARNING: TOO MANY PEOPLE",(x,y),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,0,140),1)

    #display the frames on screen to shocase the video
    cv2.imshow('The Live Video',frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

#release the camera being used and destroy the window for the video
cv2.release()
cv2.destroyAllWindows()
