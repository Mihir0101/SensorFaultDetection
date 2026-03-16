from flask import Flask, request, render_template
from src.pipelines.prediction_pipeline import PredictionPipeline
from src.logger import logging
from src.exception import CustomException
import sys


application = Flask(__name__)
app = application


@app.route('/') 
def index() :
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_datapoint() :

    logging.info('HOMEPAGE GOT GIT BY USER :)')
    
    if 'file' not in request.files :
        logging.info("FILE HAVEN'T UPLOADED YET! ")
        return 'NO FILE UPLOADED', 400
    
    file = request.files['file']

    if file.filename == " " :
        logging.info("FILE HAVEN'T SELECTED YET!!")
        return 'NO FILE SELECTED', 400
    

    try :

        prediction_pipeline = PredictionPipeline()
        df_data = prediction_pipeline.getting_features(file)
        logging.info('WE GOT OUR DATA :)')


        predictions = prediction_pipeline.prediction(df_data)
        logging.info('DONE WITH THE PREDICTIONS!!!')

        status = "WE ARE GOOD" if predictions[0] == 1 else "WE AIN'T NO GOOD"
        logging.info("EVERYTHING IS GOOOOOD :))")

        return render_template('results.html', final_result=status)
    
    except Exception as e :
        logging.info('SOMETHING WENT WRONG WHILE PREDICTING THROUGH FRONTEND :(( ')
        error = CustomException(e, sys)
        return str(error), 500


if __name__ == '__main__' :
    app.run(host='0.0.0.0', debug= True)