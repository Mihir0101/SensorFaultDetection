import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd



class PredictionPipeline :

    def __init__(self) :
        pass



    def prediction(self, features) :

        try :

            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model_path = os.path.join('artifacts','model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            transformed_data = preprocessor.transform(features)

            if  len(transformed_data.shape) == 1 :
                transformed_data = transformed_data.reshape(1,-1)
            logging.info('RESHAPED OUR DATA FOR FLUID PREDICTIONSSS')

            predicted_values = model.predict(transformed_data)

            return predicted_values
        
        except Exception as e :
            logging.info("SOMETHING WENT WRONG WHILE PREDICTING!!!")
            raise CustomException(e, sys)
        


    def getting_features(self, file):
        try:
            logging.info('# 1. Read the uploaded CSV')
            df = pd.read_csv(file)
            
            logging.info("# 2. Load the saved preprocessor")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            preprocessor = load_object(preprocessor_path)
            
            logging.info("""# 3. Get the exact 590 column names the preprocessor was trained on
            # Accessing the feature names from the ColumnTransformer""")
            expected_features = preprocessor.get_feature_names_out()
            
            logging.info("""
            # Cleaning the names: Scikit-learn adds 'NumericalPipeline__' prefix 
            # to names in a ColumnTransformer. We need the raw names.""")
            clean_expected_features = [col.split('__')[-1] for col in expected_features]

            logging.info("""# 4. Check if we can find these 590 features in the uploaded file
            # If the user uploaded the 'raw' style file (592 columns), this filters it.
            # If they uploaded only 461, it will tell you which ones are missing.""")
            try:
                df_ready = df[clean_expected_features]
            except KeyError as e:
                logging.info("# If the columns don't match, we force them by index (IF the order is the same)")
                logging.info("Column names mismatch. Forcing column alignment by position.")
                if df.shape[1] >= len(clean_expected_features):
                    df_ready = df.iloc[:, :len(clean_expected_features)]
                    df_ready.columns = clean_expected_features
                else:
                    raise Exception(f"File only has {df.shape[1]} columns. Model requires {len(clean_expected_features)}.")

            return df_ready

        except Exception as e:
            raise CustomException(e, sys)
        