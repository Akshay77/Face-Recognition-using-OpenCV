#Store the data retrieved from the features.
class SaveFeatures:  
    Feature_Stats = {}
    
    def WriteToFile(self, Feature_Stats):        
        #Error Handle
        with open("featuresFile.txt","w") as fToWrite:
            for imageName in Feature_Stats:
                line = imageName + "\t" + str(Feature_Stats[imageName]) + "\n"
                fToWrite.write(line)