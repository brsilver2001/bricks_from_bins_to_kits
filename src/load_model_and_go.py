import matplotlib.pyplot as plt
plt.style.use('ggplot')

from PIL import Image, ImageOps
import numpy as np

#import os
import cPickle as pickle
import picture_stuff as pix

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
        fname = filepath + 'keras_inception_all_re-trained-grey.h5'
    else:
        fname = filepath + 'keras_inception_all_re-trained.h5'

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
        fname = filepath + "label_dic_all-gray.pkl"
    else:
        fname = filepath + "label_dic_all.pkl"
    with open(fname) as f_un:
        label_dic = pickle.load(f_un)
    return label_dic

def plot_top_8(one_pic_X,pic_label,X,idx_preds,preds,weights):
    '''Just a plot routine to show the predicted results
       Currently hard-coded to display 2 rows, top 8 results
       but could be made more flexible later
       INPUTS: one_pic_X, numpy array of test picture
               pic_label, label for picture
               X, numpy arrary of all example pictures
               idx_preds, list of indices for predicted pictures in X
               preds, list of predicted labels
               weights, list of prediction weights
        OUTPUTS: fig, ax, matplotlib figure and axis objects
    '''
    fig, ax = plt.subplots(2,6,figsize=(14,5))

    ax1 = plt.subplot2grid((2, 6), (0, 0), colspan=2,rowspan=2)
    ax1.imshow(Image.fromarray(one_pic_X[:,:,::-1]))

    ax1.set_title("new brick pic: " + str(pic_label))
    ax1.grid(False)
    ax1.axis('off')

    for idx1 in range(2):
        for idx2 in range(4):
            ax2 = plt.subplot2grid((2, 6), (idx1,idx2+2))
            ax2.imshow(Image.fromarray(X[idx_preds[idx1*4+idx2]]))
            ax2.set_title("{} @ {:.1f}%".format(
                preds[idx1*4+idx2],100*weights[idx1*4+idx2]))
            ax2.grid(False)
            ax2.axis('off')
    plt.show();

    # If you like the figure, save it!
    picfilename = ("../saved_brick_predictions/" + str(pic_label) + "_temp.png")
    with open(picfilename, 'wb') as whatever:
        fig.savefig(whatever)

    return fig, ax

if __name__ == '__main__':
    print "Please wait a moment while loading some data and models"
    use_gray = True
    X,y_list = load_examples("../data/",use_gray)
    filepath = '../saved_models/'

    print "Data loaded, loading models"
    model = load_keras_model(filepath, use_gray)
    label_dic = load_label_dictionary(filepath, use_gray)

    image_dims = 299
    border_fraction = .3
    picture_index_lookup = pix.picture_index_function(y_list)
    print "models loaded, ready to take pictures"

    camera = pix.initialize_camera()

    # Input a file name = brick shape: e.g. 3021
    pic_label = raw_input('Type label (integer as file name):')

    extension, filename = pix.increment_filename(pic_label,extension=1)

    one_pic_X = pix.keep_shooting_until_acceptable(camera,filename)
    del(camera)

    print "checkpoint: one_pic_X size",one_pic_X.shape, "data type", one_pic_X.dtype
    predict_gen = model.predict_on_batch(np.expand_dims(one_pic_X,axis=0))

    preds, weights = pix.make_one_prediction_list(
        predict_gen,label_dic,n_match=10)

    idx_preds = [picture_index_lookup[pred] for pred in preds]

    plot_top_8(one_pic_X,pic_label,X,idx_preds,preds,weights);
