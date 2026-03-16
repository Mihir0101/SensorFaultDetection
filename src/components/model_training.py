import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from src.utils import save_object
from src.exception import CustomException
from src.logger import logging
import sys
import os
from dataclasses import dataclass

from src.utils import evaluate_model_performance


@dataclass
class ModelTrainingConfig :
    trained_model_file_path = os.path.join('artifacts','model.pkl')



class ModelTrainer :
    def __init__(self) :
        self.model_trainer_config = ModelTrainingConfig()


    def model_training(self,training_arr, testing_arr) :

        try :
            logging.info('GETTING INDEPENDENT AND DEPENDENT FEATURES...')

            x_train, y_train, x_test, y_test = (training_arr[:,:-1],
             training_arr[:,-1],
             testing_arr[:,:-1],
             testing_arr[:,-1]
            )

            models = {
                'RandomForest' : RandomForestClassifier(n_estimators=100),
                'SVC' : SVC(kernel='rbf', C=1.0, gamma='scale'),
                'DecisionTree' : DecisionTreeClassifier(criterion='entropy', max_depth=5, class_weight='balanced', random_state=33)
            }
            logging.info('SUCCESSFULLY GOT THE MODELSSS!')

            
            report:dict = evaluate_model_performance(x_train,y_train,x_test,y_test,models)

            print(report)
            logging.info(f'MODEL REPORT : {report}')

            best_score = max(sorted(report.values()))
            best_model_name = list(report.keys())[list(report.values()).index(best_score)]
            best_model = models[best_model_name]


            logging.info('WE GOT OUR BEST PERFORMING MODEL WITH ITS PERFORMANCE SCORE :)')

            print(f'BEST MODEL : {best_model_name}, SCORE : {best_score}')
            logging.info(f'BEST MODEL : {best_model_name}, SCORE : {best_score}')

            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=best_model)

        except Exception as e :
            logging.info('UNUNNNNN...SOMETHING WENT WRONG WHILE MODEL TRAINNG :(')
            raise CustomException(e, sys)