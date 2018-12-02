import pickle
import json
import numpy
from sklearn.externals import joblib
from azureml.core.model import Model
import azureml.train.automl

model_name = "model.pkl"

def init():
    global model
    model_path = Model.get_model_path(model_name = model_name)
    # deserialize the model file back into a sklearn model
    model = joblib.load(model_path)

def run(rawdata):
    try:
        data = json.loads(rawdata)['data']
        data = numpy.array(data)
        result = model.predict(data)
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
    return json.dumps({"result":result.tolist()})
