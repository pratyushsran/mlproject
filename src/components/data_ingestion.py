import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

#DataIngestion: A class to handle the ingestion of data. It reads data from a CSV file, splits it into training and testing sets, and saves them to specified paths.

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts',"train.csv")  ##isse ham iss class ko bata rhe ki respective data kaaha save karna hai
    test_data_path:str=os.path.join('artifacts',"test.csv")
    raw_data_path:str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_Data_Ingestion(self):
        logging.info("Entered into data ingestion method or component")
        try:
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the dataset as a frame')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)  #Saves the DataFrame df to the file path specified in self.ingestion_config.raw_data_path. The index=False argument prevents Pandas from writing row indices to the file,

            logging.info("Train Test Split Innitiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            #Saves the training AND test  DataFrame train_set to the file path specified in 
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True) 
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion of Data is Completed")

            # Returns the paths of the training and testing data
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        except Exception as  e:
            raise CustomException(e,sys)
        

# if __name__=="__main__":: This line ensures that the following block of code only runs if this script is executed as the main program. It will not run if the script is imported as a module in another script.
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_Data_Ingestion()#Calls the initiate_Data_Ingestion method on the obj instance, performing the data ingestion process and storing the returned training and testing data paths in the variables train_data and test_data.

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))


