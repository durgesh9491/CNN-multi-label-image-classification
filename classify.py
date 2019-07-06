# USAGE
# python classify.py

import numpy as np
import imutils
import cv2
import os
import pickle
import pdb

from imutils import paths
from keras.models import load_model
from keras.preprocessing.image import img_to_array

class classifyObject:

  @staticmethod
  def init_configs():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

  def set_params(self, **options):
    if 'image_dims' in options: self.image_dims = options['image_dims']
    if 'prob_cutoff' in options: self.prob_cutoff = options['prob_cutoff']

  def __init__(self, model, mlb, func, **options):
    self.init_configs()
    self.model = load_model(model, custom_objects = {'auc': func})
    self.mlb = pickle.loads(open(mlb, "rb").read())
    self.set_params(**options)

  #predict the image label with maximum probability
  def process_prediction(self, imagePath, graph):
    with graph.as_default():
      try:
        image = cv2.resize(cv2.imread(imagePath), (self.image_dims[1], self.image_dims[0]))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis = 0)
        proba = self.model.predict(image)[0]
        if max(proba) >= self.prob_cutoff:
          label = self.mlb.classes_[np.argsort(proba)[::-1][:1][0]]
          return label
        else:
          return "System is unable to recognize the given image!"

      except Exception as e:
        return str(e)



