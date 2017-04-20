#Python Project 
import cv2
import os
import FeatureCalculation as fc
import shutil

class TestingModule:
    testPath = ''
    
    def  __init__(self, path):
        self.InitCascadeFiles()
        TestingModule.testPath = path
    
    def InitCascadeFiles(self):
        self.headShoulderCascade = cv2.CascadeClassifier("HS.xml")
        self.mouthCascade        = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")
        self.frontalFaceCascade  = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.eyeCascade          = cv2.CascadeClassifier("haarcascade_eye.xml")
        self.noseCascade         = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")


    def Testing(self):
        imageName = raw_input("Enter the name of the image to test : ")
        imageName = TestingModule.testPath + "\\" + imageName + ".jpg"
        
        feature_dict   = {}
        left_eye_list  = []
        right_eye_list = []
        nose_list      = []
        mouth_list     = []
            
        image   = cv2.imread(imageName)

        resized = cv2.resize(image,None,fx = 0.5,fy = 0.5,interpolation = cv2.INTER_AREA) #resize the image
        gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        face = self.frontalFaceCascade.detectMultiScale(gray, 1.3, 4, cv2.cv.CV_HAAR_SCALE_IMAGE, (30,30))

        for (x,y,w,h) in face:
            cv2.rectangle(gray, (x,y), (x+w,y+h), (255,0,0),1)
                
            roi_gray = gray[y:y+h, x:x+w]
            roi_color =gray[y:y+h, x:x+w]
                
            eyes  = self.eyeCascade.detectMultiScale(roi_gray)
            nose  = self.noseCascade.detectMultiScale(roi_gray)
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

                    left_eye_midpoint_x = eyes[0][0] + eyes[0][2] / 2
                    left_eye_midpoint_y = eyes[0][1] + eyes[0][3] / 2
                    right_eye_midpoint_x = eyes[1][0] + eyes[1][2] / 2
                    right_eye_midpoint_y = eyes[1][1] + eyes[1][3] / 2
                    nose_midpoint_x = nose[0][0] + nose[0][2] / 2
                    nose_midpoint_y = nose[0][1] + nose[0][3] / 2
                    mouth_midpoint_x = mouth[0][0] + mouth[0][2] / 2
                    mouth_midpoint_y = mouth[0][1] + mouth[0][3] / 2

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
                    
                    imageName_org = imageName
                    imageNamePart = imageName.split('\\')
                    imageName = imageNamePart[len(imageNamePart)-1]
                    
                    ImageDictionary = {}
                    ImageDistances = {}
                    ImageDictionary[imageName] = feature_dict
                    
                    featureCal = fc.Feature_Calculation()
                    featureCal.calculate_features(ImageDictionary, ImageDistances)

                    self.compareFeatures(ImageDistances,gray,imageName_org)


    def compareFeatures(self, ImageDistances,gray,imageName_org):
        #for key, value in ImageDistances.items():
         #   ImageDistances_List = [key, value]
        Image_Distances_str = " "
        for i in ImageDistances:
            Image_Distances_str = i +"\t"+ str(ImageDistances[i])
        Image_Distances_str.replace("{'}","",10)
        Image_Distances_str = Image_Distances_str.split("\t")
        Image_Distances_str[1] = Image_Distances_str[1].replace("{","",10)
        print Image_Distances_str[1]

        with open('featuresFile.txt') as f:
            present = False
            file_line_list = f.readline()
