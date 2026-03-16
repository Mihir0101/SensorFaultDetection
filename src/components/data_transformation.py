from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import sys, os
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object



## DATA TRANSFORMATION CONFIG
class DataTransformationConfig :
    preprocessor_obj_path = os.path.join('artifacts','preprocessor.pkl')




## DATA TRANSFORMATION CLASS
class DataTransformation :
    def __init__(self) :
        self.data_transformation_config = DataTransformationConfig()


    
    def get_data_transformer(self,data_path = './artifacts/raw.csv') :
        
        logging.info('PROCESS OF BUILDING PREPROCESSOR BEGINS!')

        try :

            data = pd.read_csv(data_path)
            cols = data.select_dtypes(exclude='object').columns
            logging.info('EXCLUDING THE TARGET COLUMN')
            num_cols = list(cols)
            num_cols.remove('Good/Bad')

            logging.info("INITIALIZING PIPELINE")

            logging.info("STARTING WITH NUMERICAL PIPELINE")
            num_pipeline = Pipeline(
                steps = [('Imputer',SimpleImputer(strategy='median')),
                         ('Scaler',StandardScaler())]
            )

            logging.info("GETTING PREPROCESSOR READY")
            preprocessor = ColumnTransformer([('NumericalPipeline',num_pipeline,num_cols)])

            return preprocessor
            logging.info("PIPELINE COMPLETED")

        except Exception as e :

            logging.info("GOT AN ERROR WHILE TRANSFORMING DATA :(")
            raise CustomException(e,sys)



    def initiate_data_transformer(self, train_data_path, test_data_path) :
        
        try :

            training_data = pd.read_csv(train_data_path)
            testing_data = pd.read_csv(test_data_path)

            logging.info('READ THE TRAINING AND TESTING DATA')
            logging.info(f'TrainingData : \n{training_data.head().to_string()}')
            logging.info(f'TestingData : \n {testing_data.head().to_string()}')

            preprocessor_obj = self.get_data_transformer()
            logging.info('GOT THE PREPROCESSOR!')

            target_column = 'Good/Bad'
            

            logging.info('WE ARE GOING TO GET INDEPENDENT AND DEPENDENT FEATURES!')
            x_train = training_data.drop(target_column,axis=1)
            y_train = training_data[target_column]

            x_test = testing_data.drop(target_column,axis=1)
            y_test = testing_data[target_column]


            logging.info('TRANSFORMATION BEGIN!!')
            x_train_transformed = preprocessor_obj.fit_transform(x_train)
            x_test_transformed = preprocessor_obj.transform(x_test)
            logging.info('DONE WITH TRANSFORMATION!')

            trining_arr = np.c_[x_train_transformed, np.array(y_train)]
            testing_arr = np.c_[x_test_transformed, np.array(y_test)]

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_path,
                obj = preprocessor_obj
            )
            logging.info('PRE-PROCESSOR GOT SAVED IN PICKLE FILE!!!')

            return(
                trining_arr,
                testing_arr,
                self.data_transformation_config.preprocessor_obj_path
            )
        
        except Exception as e :
            logging.info('EXCEPTION OCCURED WHILE DATA TRANSFORMATION :(')

            raise CustomException(e,sys)            