import cv2
import os

import SaveFeatures as sf
reload(sf)

import FeatureCalculation as fc
reload(fc)

class FaceRecognition:
    
    def __init__(self, path):
        self.InitCascadeFiles()
        self.path = path
    
    def InitCascadeFiles(self):
        print 'In InitCascadeFiles'
        self.headShoulderCascade = cv2.CascadeClassifier("HS.xml")
        self.mouthCascade        = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")
        self.frontalFaceCascade  = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.eyeCascade          = cv2.CascadeClassifier("haarcascade_eye.xml")
        self.noseCascade         = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")

    def FaceDetection(self, All_Features_Dict):
            images = [os.path.join(self.path,f) for f in os.listdir(self.path)]
            for imageName in images:
                feature_dict   = {}
                left_eye_list  = []
                right_eye_list = []
                nose_list      = []
                mouth_list     = []
                
                image = cv2.imread(imageName)
                print len(image)
                resized = cv2.resize(image,None,fx = 0.5,fy = 0.5,interpolation = cv2.INTER_AREA) #resize the image
                print "---------------------------------------"
                gray  = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                print len(gray)
                face = self.frontalFaceCascade.detectMultiScale(gray, 1.3, 4, cv2.cv.CV_HAAR_SCALE_IMAGE, (30,30))
                        
                for (x,y,w,h) in face:
                    
                    cv2.rectangle(gray, (x,y), (x+w,y+h), (255,0,0),1)
                    
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color =gray[y:y+h, x:x+w]

                    eyes = self.eyeCascade.detectMultiScale(roi_gray)
                    nose = self.noseCascade.detectMultiScale(roi_gray)
                    mouth = self.mouthCascade.detectMultiScale(gray, 1.3, 25)

                    if ((len(eyes) == 2)  and
                        (len(nose)  == 1) and
                        (len(mouth) == 1)):
                        for (ex,ey,ew,eh) in eyes:
                            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                        
                        for (nx,ny,nw,nh) in nose:
                            #print "nose :", nx,ny,nw,nh
                            cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(255,255,0),2)
                        
                        for (mx, my, mw, mh) in mouth:
                            my = int(my - 0.15 * mh)
                            mw = int(mw + 0.5 * mw)
                            cv2.rectangle(gray, (mx,my),(mx+mw,my+mh), (0, 255, 0), 3)
    
                        left_eye_midpoint_x = eyes[0][0] + eyes[0][2]/2
                        left_eye_midpoint_y = eyes[0][1] + eyes[0][3]/2
                        right_eye_midpoint_x = eyes[1][0] + eyes[1][2]/2
                        right_eye_midpoint_y = eyes[1][1] + eyes[1][3]/2
                        nose_midpoint_x = nose[0][0] + nose[0][2]/2
                        nose_midpoint_y = nose[0][1] + nose[0][3]/2
                        mouth_midpoint_x = mouth[0][0] + mouth[0][2]/2
                        mouth_midpoint_y = mouth[0][1] + mouth[0][3]/2

                        left_eye_list.append(left_eye_midpoint_x)
                        left_eye_list.append(left_eye_midpoint_y)
                        right_eye_list.append(right_eye_midpoint_x)
                        right_eye_list.append(right_eye_midpoint_y)
                        nose_list.append(nose_midpoint_x)
                        nose_list.append(nose_midpoint_y)
                        mouth_list.append(mouth_midpoint_x)
                        mouth_list.append(mouth_midpoint_y)
    
                        feature_dict["left_eye" ] = left_eye_list
                        feature_dict["right_eye"] = right_eye_list
                        feature_dict["nose"     ] = nose_list
                        feature_dict["mouth"    ] = mouth_list
    
                        imageNamePart = imageName.split('\\')
                        imageName = imageNamePart[len(imageNamePart)-1]
                        All_Features_Dict[imageName] = feature_dict
    
                        #cv2.imshow('Faces Found', gray)
                        #cv2.waitKey(0)
                        #cv2.destroyAllWindows()