#mouth cascade
import cv2
import numpy as np
import os

mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

if mouth_cascade.empty():
  raise IOError('Unable to load the mouth cascade classifier xml file')

path = "test"
images = [os.path.join(path,f) for f in os.listdir(path)]
for image in images:
    image =cv2.imread(image)
    #frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mouth_rects = mouth_cascade.detectMultiScale(gray, 1.3, 25)
    
    for (x,y,w,h) in mouth_rects:
        y= int(y - 0.15*h)
        w = int(w + 0.5*w)
        cv2.rectangle(gray, (x,y), (x+w,y+h), (0,255,0), 3)
        break

    cv2.imshow('Mouth Detector', gray)
    cv2.waitKey(0)