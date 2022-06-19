print('Version 1.1')

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inception_v3_preprocessor

from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model

from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

import cv2
import pandas as pd
import numpy as np
import keras
from tqdm import tqdm
# from tqdm.notebook import tqdm_notebook

# enable gpu on keras
import tensorflow as tf
# from tensorflow.python.keras import backend as K
# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
# K.set_session(sess)

config = tf.compat.v1.ConfigProto(log_device_placement=True)
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

class image_binary_classify_keras(object):
    def __init__(self):
        self.model = None
        self.train_generator = None
        self.val_generator = None
        self.datagen = ImageDataGenerator(rescale=1./255, 
                               rotation_range=30, 
                               # zoom_range = 0.3, 
                               width_shift_range=0.2,
                               height_shift_range=0.2, 
                               horizontal_flip = 'true')
        self.onehot = None

    def init_model(self, checkpoint_path=None, binary_class=True, num_classes=None):
        '''
            checkpoint_path: checkpoint model
        '''
        if checkpoint_path: # load checkpoint model
            self.model = keras.models.load_model(checkpoint_path)
            print('===== Load model successful from', checkpoint_path)

        elif binary_class: # init model with imagenet weight
            print('===== Init new model for binary classes.')
            # Get the InceptionV3 model so we can do transfer learning
            base_model = InceptionV3(weights = 'imagenet', include_top = False, input_shape=(299, 299, 3))

            # Add a global spatial average pooling layer
            x = base_model.output
            x = GlobalAveragePooling2D()(x)

            # Add a fully-connected layer and a logistic layer 
            x = Dense(512, activation='relu')(x)
            predictions = Dense(2, activation='sigmoid')(x)

            # first: train only the top layers i.e. freeze all convolutional InceptionV3 layers
            for layer in base_model.layers:
                layer.trainable = False

            # The model we will train
            self.model = Model(inputs = base_model.input, outputs = predictions)

            # Compile with Adam
            self.model.compile(Adam(lr=.0001), loss='binary_crossentropy', metrics=['accuracy'])

        else:
            print(f'===== Init new model for {num_classes} classes.')
            # Get the InceptionV3 model so we can do transfer learning
            base_model = InceptionV3(weights = 'imagenet', include_top = False, input_shape=(299, 299, 3))

            # Add a global spatial average pooling layer
            x = base_model.output
            x = GlobalAveragePooling2D()(x)

            # Add a fully-connected layer and a logistic layer 
            x = Dense(512, activation='relu')(x)
            predictions = Dense(num_classes, activation='softmax')(x)

            # first: train only the top layers i.e. freeze all convolutional InceptionV3 layers
            for layer in base_model.layers:
                layer.trainable = False

            # The model we will train
            self.model = Model(inputs = base_model.input, outputs = predictions)
                
            # Compile with Adam
            self.model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    def generate_train_valid(self, img_path_arr, label_arr):
        '''
            img_path_arr: pandas series (path of image)
            label_arr: pandas series 
        '''
        self.onehot = pd.get_dummies(label_arr, sparse = True)
        
        # Load image to array
        print(f'===== Load images...')
        train_data = np.array([img_to_array(load_img(img, target_size=(299, 299))) for img in tqdm(img_path_arr.values.tolist())]).astype('float32')

        # Split
        x_train, x_validation, y_train, y_validation = train_test_split(train_data, label_arr, test_size=0.1, stratify=np.array(label_arr), random_state=2022)
        y_train = pd.get_dummies(y_train.reset_index(drop=True)).values
        y_validation = pd.get_dummies(y_validation.reset_index(drop=True)).values

        self.train_generator = self.datagen.flow(x_train, y_train, shuffle=False, batch_size=10, seed=2022)
        self.val_generator = self.datagen.flow(x_validation, y_validation, shuffle=False, batch_size=10, seed=2022)

    def train_model(self, num_epochs=10):
        self.model.fit_generator(self.train_generator,
                                steps_per_epoch = 256,
                                validation_data = self.val_generator,
                                validation_steps = 32,
                                epochs = num_epochs,
                                verbose = 1)

    def save_model(self, model_path):
        self.model.save(model_path)
        print(f'===== Saved model.')