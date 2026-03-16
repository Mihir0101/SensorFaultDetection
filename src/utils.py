import os
import sys
import pickle
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import accuracy_score

def save_object(file_path, obj) :

    try :
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok= True)

        with open(file_path, 'wb') as file_obj :
            pickle.dump(obj, file_obj)

    except Exception as e :
        raise CustomException(e, sys)
    


def evaluate_model_performance(x_train,y_train,x_test,y_test,models) :

    try :
        report = {}

        for name, model in models.items() :

            model.fit(x_train,y_train)

            y_predicted = model.predict(x_test)
            accuracy = accuracy_score(y_test,y_predicted)

            report[name] = accuracy

        return report
    
    except Exception as e :
        logging.info('WE GOT SOME TROUBLE WHILE MODEL TRAINING!!')
        raise CustomException(e, sys)
    


def load_object(file_path) :
    
    try :
        with open(file_path,'rb') as file_obj :
            return pickle.load(file_obj)
    
    except Exception as e :
        logging.info("SOMETHING BAD HAPPEND WHILE LOADING OBJECTS :(")
        raise CustomException(e, sys)