import numpy as np
import cv2
import os
import SaveFeatures as SF
import math

All_Features_Dict = {}
Feature_Stats = {}

class FaceRecognition:
    #All_Features_Dict = {}
    def __init__(self, path):
        print 'In Facerecognition Constructor'
        self.InitCascadeFiles()
        self.path = path
        #self.feature_id = feature_id
        self.sf = SF.SaveFeatures()
    
    def InitCascadeFiles(self):
        print 'In InitCascadeFiles'
        self.headShoulderCascade = cv2.CascadeClassifier("HS.xml")
        self.mouthCascade        = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")
        self.frontalFaceCascade  = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.eyeCascade          = cv2.CascadeClassifier("haarcascade_eye.xml")
        self.noseCascade         = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")

    def FaceDetection(self):
        print "In Face Detecttion"
        image_count=0
        images = [os.path.join(self.path,f) for f in os.listdir(self.path)]
        for image in images:
            image_count = image_count + 1
            feature_dict = {}
            left_eye_list = []
            right_eye_list = []
            nose_list = []
            mouth_list = []
            image = cv2.imread(image) #Read Image
            #r = 275.0 / image.shape[1]
            #dim = (275, int(image.shape[0]*r))
            #print 'dimension = ', dim, 'value of r = ', r
            resized = cv2.resize(image,None,fx = 0.5,fy = 0.5,interpolation = cv2.INTER_AREA) #resize the image
            gray  = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) #BGR to Gray Scale Conversion
            face = self.frontalFaceCascade.detectMultiScale(gray, 1.3, 4, cv2.cv.CV_HAAR_SCALE_IMAGE, (30,30))
            # Used to find faces and eyes.
            #cv2.CascadeClassifier.detectMultiScale(image,scaleFactor,minNeighbors,flags,minSize,maxSize)
                               
            print "FACE :", face
            print "Found {0} faces!".format(len(face))
                       
            for (x,y,w,h) in face:
                #print "faces :", x,y,w,h
                cv2.rectangle(gray, (x,y), (x+w,y+h), (255,0,0),1)
                #cv2.rectangle(image,(vertex of rect, vert of rect opposite to first vertex), (Thickness of lines that makeup the rectangle
                # ), (lineType(R,G,B)), shift)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color =gray[y:y+h, x:x+w]
                
                eyes = self.eyeCascade.detectMultiScale(roi_gray)
                print "Found {0} eyes!".format(len(eyes))
                for (ex,ey,ew,eh) in eyes:
                    print "eyes :", ex,ey,ew,eh
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                left_eye_list.append(eyes[0][0])
                left_eye_list.append(eyes[0][1])
                right_eye_list.append(eyes[1][0])
                right_eye_list.append(eyes[1][1])
        
                nose = self.noseCascade.detectMultiScale(roi_gray)
                print "Found {0} nose!".format(len(nose)) 
                for (nx,ny,nw,nh) in nose:
                    print "nose :", nx,ny,nw,nh
                    cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(255,255,0),2)
                    nose_list.append(nx)
                    nose_list.append(ny)

                mouth = self.mouthCascade.detectMultiScale(gray, 1.3, 25)
                print "Found {0} mouth".format(len(mouth))
                for (mx, my, mw, mh) in mouth:
                    my = int(my - 0.15 * mh)
                    mw = int(mw + 0.5 * mw)
                    print "mouth :", mx,my,mw,mh
                    print "---------------------------------------------------------"
                    cv2.rectangle(gray, (mx,my),(mx+mw,my+mh), (0, 255, 0), 3)
                    mouth_list.append(mx)
                    mouth_list.append(my)
                    break

            feature_dict["left_eye"] = left_eye_list
            feature_dict["right_eye"] = right_eye_list
            feature_dict["nose"] = nose_list
            feature_dict["mouth"] = mouth_list

            All_Features_Dict[image_count] = feature_dict

            cv2.imshow('Faces Found', gray)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return All_Features_Dict

class Feature_Calculation:
    def calculate_features(self,All_Features_Dict):
        image_count = 1
        for image in All_Features_Dict:
            feature_stats_per_image = {}
            eyes_distance = math.sqrt(
                math.pow(All_Features_Dict[image]['right_eye'][0]-All_Features_Dict[image]['left_eye'][0],2)
                + math.pow(All_Features_Dict[image]['right_eye'][1]-All_Features_Dict[image]['left_eye'][1],2))

            left_eye_nose_distance = math.sqrt(
                math.pow(All_Features_Dict[image]['nose'][0] - All_Features_Dict[image]['left_eye'][0],
                         2) + math.pow(
                    All_Features_Dict[image]['nose'][1] - All_Features_Dict[image]['left_eye'][1], 2))

            right_eye_nose_distance = math.sqrt(
                math.pow(All_Features_Dict[image]['nose'][0] - All_Features_Dict[image]['right_eye'][0],
                         2) + math.pow(
                    All_Features_Dict[image]['nose'][1] - All_Features_Dict[image]['right_eye'][1], 2))

            mouth_nose_distance = math.sqrt(
                math.pow(All_Features_Dict[image]['nose'][0] - All_Features_Dict[image]['mouth'][0],
                         2) + math.pow(
                    All_Features_Dict[image]['nose'][1] - All_Features_Dict[image]['mouth'][1], 2))

            feature_stats_per_image['eyes_dist'] = float("{0:.2f}".format(eyes_distance))
            feature_stats_per_image['left_eye_nose'] = float("{0:.2f}".format(left_eye_nose_distance))
            feature_stats_per_image['right_eye_nose'] = float("{0:.2f}".format(right_eye_nose_distance))
            feature_stats_per_image['mouth_nose'] = float("{0:.2f}".format(mouth_nose_distance))
            Feature_Stats[image_count] = feature_stats_per_image
            image_count = image_count + 1
        return Feature_Stats


def main():
    print 'In Main'
    facereco = FaceRecognition("faces")
    All_Features_Dict = facereco.FaceDetection()
    feature_cal = Feature_Calculation()
    Feature_Stats = feature_cal.calculate_features(All_Features_Dict)
    print Feature_Stats
    
if __name__ == "__main__":
    main()