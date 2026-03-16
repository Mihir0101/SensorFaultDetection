import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


### DATA INGESTION CONFIG

@dataclass
class DataIngestionConfig :

    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')
    raw_data_path = os.path.join('artifacts','raw.csv')


### DATA INGESTION CLASS

class DataIngestion :
    def __init__(self) :

        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self) :
        logging.info('DATA INGESTION STARTSSS!')

        try :

            wafer = pd.read_csv(r"C:\Users\Admin\Desktop\WaferFaultDetectionnn\notebooks\data\wafer_23012020_041211 (1).csv")
            logging.info('WE GOT DATA INTO WAFER!')

            wafer.drop('Unnamed: 0',axis=1,inplace=True)

            # os.makedir(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok = True)
            wafer.to_csv(self.ingestion_config.raw_data_path,index=False)

            logging.info('HEADING TO SPLIT')

            train_data, test_data = train_test_split(wafer,test_size=0.20)
            train_data.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('DATA INGESTION COMPLITED :)')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e :
            logging.info('ERROR OCCURED WHILE DATA INGESTION :(')