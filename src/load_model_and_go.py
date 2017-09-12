import matplotlib.pyplot as plt
plt.style.use('ggplot')

from PIL import Image, ImageOps
import numpy as np

#import os
import cPickle as pickle
from src import picture_stuff as pix

from keras.models import load_model

'''
This file:
 1) Loads a pickled batch of brick pictures
 2) Loads a trained Keras-TensorFlow model basd on InceptionV3
 3) Takes 1 new picture
 4) Runs the pictures through the model to make predictions
 5) Displays pictures from the original batch matching predictions
 '''

def load_examples(filepath,use_gray=True):
    '''Take path input information and load pickled
       photos and labels.
       INPUTS:  path: string of relative path to pickled models
                use_gray: boolean of whether to convert to grayscale
       OUTPUTS: X: numpy array of sample pictures
                y_list: list of integer lego IDs as labels to pictures
    '''
    with open(filepath + "X_examples.pkl") as f_un:
        X = pickle.load(f_un)

    if use_gray:
        X = pix.convert_to_gray(X)

    with open(filepath + "y_examples_list.pkl") as f_un:
        y_list = pickle.load(f_un)

    return X,y_list

def load_keras_model(filepath, use_gray=True):
    '''Take path input information and load keras model
       INPUTS:  path: string of relative path to models
                use_gray: boolean of whether to convert to grayscale
       OUTPUTS: model: trained Keras-TensorFlow model
    '''
    if use_gray:
        #fname = filepath + 'keras_inception_all_partially_trained_gray.h5'
        #fname = filepath + 'keras_inception_all_trained_gray.h5'
        fname = filepath + 'keras_example_inception_trained-gray.h5'
    else:
        fname = filepath + 'keras_example_inception_trained-copy2.h5'

    model = load_model(fname)
    return model

def load_label_dictionary(filepath, use_gray=True):
    '''Take path input information and load dictionary data for
       relating the Keras-TensorFlow model to brick IDs
       INPUTS:  path: string of relative path to models
                use_gray: boolean of whether to convert to grayscale
       OUTPUTS: label_dic: dictionary of label data
    '''
    if use_gray:
        fname = filepath + "label_dic-gray.pkl"
    else:
        fname = filepath + "label_dic.pkl"
    with open(fname) as f_un:
        label_dic = pickle.load(f_un)
    return label_dic


if __name__ == '__main__':
    use_gray = True
    X,y_list = load_examples("data/",use_gray)
    filepath = '../saved_models/'
    model = load_keras_model(filepath, use_gray)
    label_dic = load_label_dictionary(filepath, use_gray)
