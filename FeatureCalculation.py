import math

class Feature_Calculation:
   
    def calculate_features(self, All_Features_Dict, Feature_Stats):           
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

            left_eye_mouth_distance = math.sqrt(
                math.pow(All_Features_Dict[image]['mouth'][0] - All_Features_Dict[image]['left_eye'][0],
                         2) + math.pow(
                    All_Features_Dict[image]['mouth'][1] - All_Features_Dict[image]['left_eye'][1], 2))

            right_eye_mouth_distance = math.sqrt(
                math.pow(All_Features_Dict[image]['mouth'][0] - All_Features_Dict[image]['right_eye'][0],
                         2) + math.pow(
                    All_Features_Dict[image]['mouth'][1] - All_Features_Dict[image]['right_eye'][1], 2))

            feature_stats_per_image['eyes_dist'] = float("{0:.2f}".format(eyes_distance))
            feature_stats_per_image['left_eye_nose'] = float("{0:.2f}".format(left_eye_nose_distance))
            feature_stats_per_image['right_eye_nose'] = float("{0:.2f}".format(right_eye_nose_distance))
            feature_stats_per_image['mouth_nose'] = float("{0:.2f}".format(mouth_nose_distance))
            feature_stats_per_image['left_eye_mouth'] = float("{0:.2f}".format(left_eye_mouth_distance))
            feature_stats_per_image['right_eye_mouth'] = float("{0:.2f}".format(right_eye_mouth_distance))
            
            Feature_Stats[image] = feature_stats_per_image