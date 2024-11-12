import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig


# if we are defining only variables inside a class, we can use decorator @dataclass to make it simpler
# it wouldnt require a separate constructor to be defined
@dataclass
class DataIngestionConfig:
    raw_data_path: str=os.path.join('artifacts','raw.csv')
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered Data Ingestion Method')
        try:
            df=pd.read_csv('notebook/data/stud.csv')
            logging.info('Data read from csv file')

            # exist_ok=True: keeps adding to the directory
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) 
           
            '''
            index=False: This excludes the DataFrame's index from the CSV output, saving only the data in the columns. This is 
            often used when you don’t need the row indices in the CSV file, which makes it cleaner and focuses only on the data.
            index=True (default): This includes the DataFrame’s index as the first column in the CSV file. This can be useful if 
            the index holds meaningful information or if you want to preserve it as part of the data.
            '''
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info('Initiating Train Test Split')
            train_set, test_set=train_test_split(df,test_size=0.2,random_state=12)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)
            

