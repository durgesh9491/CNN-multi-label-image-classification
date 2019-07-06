# USAGE
# FLASK_APP=server.py FLASK_ENV=development DEBUG=True flask run

from flask import Flask, request, jsonify
from classify import classifyObject
from train import trainCNN

import pdb
import numpy as np
import keras
import tensorflow as tf

app = Flask(__name__)
graph = tf.get_default_graph()

# we need to redefine our metric function in order to use it when loading the model
def redefine_metrics(y_true, y_pred):
  ret = tf.metrics.auc(y_true, y_pred)[1]
  keras.backend.get_session().run(tf.local_variables_initializer())
  return ret

classification_instance = classifyObject(
  model = "car.model",
  mlb = "mlb.pickle",
  func = redefine_metrics,
  image_dims = (128, 128, 3),
  prob_cutoff = 0.75)

training_instance = trainCNN(
  dataset = "dataset",
  model = "car.model",
  mlb = "mlb.pickle",
  learning_rate = 0.05,
  epochs = 75,
  batch_size = 32,
  image_dims = (128, 128, 3),
  test_size = 0.20,
  seed = 42)

@app.route('/predict')
def predict():
  data = {"success": False}
  if 'img' in request.args:
    data["response"] = classification_instance.process_prediction(request.args['img'], graph)
    data["success"] = True
  return jsonify(data)

@app.route('/train', methods=["POST"])
def train():
  data = {"success": True}
  data["response"] = training_instance.process_training(graph)
  return jsonify(data)

if __name__ == '__main__':
  app.run(host='0.0.0.0', port = 5000, debug = True)
