# USAGE
# multi-label classification problem
# python train.py

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from pyimagesearch.smallervggnet import SmallerVGGNet
from imutils import paths

import numpy as np
import random
import pickle
import cv2
import os
import pdb

class trainCNN:

  @staticmethod
  def init_configs():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

  def __init__(self, dataset, model, mlb, **options):
    self.dataset = dataset
    self.model = model
    self.mlb = mlb
    self.set_params(**options)

  def set_params(self, **options):
    if 'learning_rate' in options: self.learning_rate = options['learning_rate']
    if 'epochs' in options: self.epochs = options['epochs']
    if 'batch_size' in options: self.batch_size = options['batch_size']
    if 'image_dims' in options: self.image_dims = options['image_dims']
    if 'test_size' in options: self.test_size = options['test_size']
    if 'seed' in options: self.seed = options['seed']

  def shuffle_dataset(self):
    imagePaths = sorted(list(paths.list_images(self.dataset)))
    random.seed(self.seed)
    random.shuffle(imagePaths)
    return imagePaths

  def process_training(self, graph):
    with graph.as_default():
      try:
        imagePaths = self.shuffle_dataset()
        data, labels = [], []
        for imagePath in imagePaths:
          image = cv2.resize(cv2.imread(imagePath), (self.image_dims[1], self.image_dims[0]))
          image = img_to_array(image)
          data.append(image)
          labels.append(imagePath.split(os.path.sep)[-2].split())

        data = np.array(data, dtype = "float") / 255.0
        labels = np.array(labels)

        mlb = MultiLabelBinarizer()
        labels = mlb.fit_transform(labels)
        (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = self.test_size, random_state = self.seed)

        aug = ImageDataGenerator(
          rotation_range = 25,
          width_shift_range = 0.1,
          height_shift_range = 0.1,
          shear_range = 0.2,
          zoom_range = 0.2,
          horizontal_flip = True,
          vertical_flip = True,
          fill_mode = "nearest")

        model = SmallerVGGNet.build(
          width = self.image_dims[1],
          height = self.image_dims[0],
          depth = self.image_dims[2],
          classes = len(mlb.classes_),
          finalAct = "sigmoid")

        model.compile(
          loss = "binary_crossentropy",
          optimizer = Adam(lr = self.learning_rate, decay = self.learning_rate / self.epochs),
          metrics = ["accuracy"])

        model.fit_generator(
          aug.flow(trainX, trainY, batch_size = self.batch_size),
          validation_data = (testX, testY),
          steps_per_epoch = len(trainX) // self.batch_size,
          epochs = self.epochs,
          verbose = 1)

        model.save(self.model)
        f = open(self.mlb, "wb")
        f.write(pickle.dumps(mlb))
        f.close()
        return "training completed successfully!"

      except Exception  as e:
        return str(e)
