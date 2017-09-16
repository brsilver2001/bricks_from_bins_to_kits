from flask import Flask, render_template, request, jsonify

# NOTE: REVIEW IMPORTS -- SOME DON'T HAVE TO BE HERE

import matplotlib.pyplot as plt
plt.style.use('ggplot')

from PIL import Image, ImageOps
import numpy as np
import cv2
import base64
from io import BytesIO
import StringIO


#import os
import cPickle as pickle
import src.picture_stuff as pix
import src.load_model_and_go as ld

from keras.models import load_model

app = Flask(__name__)


# NOTE: to graph in flask, see
# https://gist.github.com/wilsaj/862153/119c6fc8ba2b0f3ffcd285a6852acb028660395b
# ALSO, see this:
# https://pythonspot.com/en/flask-web-app-with-python/

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/2_take_pictures', methods=['GET'])
def take_pictures():
    print "executing template: 2_take_pictures"
    return render_template('2_take_pictures.html')

@app.route('/3_login', methods=['GET'])
def login():
    return render_template('3_login.html')

@app.route('/4_new_user', methods=['GET'])
def new_user():
    return render_template('4_new_user.html')

@app.route('/5_make_brick_list', methods=['GET'])
def make_brick_list():
    return render_template('5_make_brick_list.html')

@app.route('/cam1', methods=['GET'])
def cam1():
    # At the moment, the camera.html file is also on
    # http://html.camera.test.s3-website-us-west-2.amazonaws.com/
    return render_template('cam1.html')



@app.route('/classify', methods=['POST'])
def classify():
    print "executing template: classify"
    user_data = request.json
    pic_label = user_data['JSON_input_label']

    #one_pic_X will be the picture from <img id="photo">
    raw_data = user_data['JSON_pic']

    intermediate = raw_data.split(',')[1]

    base64_X = base64.decodestring(intermediate.encode('utf-8'))
    print "checkpoint: user data input = ",pic_label

    #with open('test.png','wb') as f:
    #    f.write(base64_X)

    tempBuff = StringIO.StringIO()
    tempBuff.write(base64_X)
    tempBuff.seek(0) #need to jump back to the beginning before handing it off to PIL
    X_image = Image.open(tempBuff)
    print "X_image type ???", type(X_image)
    #X_image.save('test2.png')

    one_pic_X = np.array(X_image)

    print "checkpoint: one_pic_X shape:",one_pic_X.shape
    one_pic_X = pix.crop_and_scale(one_pic_X,use_gray=True,
                               image_dims=299,border_fraction=0.0)
    print "checkpoint: one_pic_X shape:",one_pic_X.shape

    Image.fromarray(one_pic_X).save('test_image.png')

    #  Currently needs to fix sizing before going into model_output
    #  See https://github.com/nodeca/pica

    extension, filename = pix.increment_filename(pic_label,extension=1)
    print "checkpoint: file name is ",filename

    this_model_input = np.expand_dims(one_pic_X,axis=0)
    #print "this_model_input size",this_model_input.shape, "data type", this_model_input.dtype

    predict_gen = model.predict_on_batch(this_model_input)

    preds, weights = pix.make_one_prediction_list(
        predict_gen,label_dic,n_match=10)

    idx_preds = [picture_index_lookup[pred] for pred in preds]

    fig, ax = ld.plot_top_8(one_pic_X,pic_label,X,idx_preds,preds,weights);


    picfilename = "../saved_brick_predictions/" + pic_label + "_temp.png"

    model_output = "label = {}, predicted ID = {}, weight = {:.1f}%".format(
                        pic_label,preds[0], weights[0]*100)


    teststring = "testing"

    print "ok to here: model_output =", model_output
    return jsonify({'root_1': model_output, 'pic_x': picfilename})




if __name__ == '__main__':

    # first thing to do on start-up is load the model stuff, then run the app
    print "Please wait a moment while loading some data and models"
    use_gray = True
    X,y_list = ld.load_examples("../data/",use_gray)
    filepath = '../saved_models/'

    print "Data loaded, loading models"
    model = ld.load_keras_model(filepath, use_gray)
    label_dic = ld.load_label_dictionary(filepath, use_gray)

    image_dims = 299
    border_fraction = .3
    picture_index_lookup = pix.picture_index_function(y_list)
    print "models loaded, ready to take pictures"


    '''
    one_pic_X = X[0]
    print "Model test during setup"
    this_model_input = np.expand_dims(one_pic_X,axis=0)
    print "Test OK: this_model_input size",this_model_input.shape, "data type", this_model_input.dtype
    predict_gen = model.predict_on_batch(this_model_input)
    '''

    app.run(host='0.0.0.0', threaded=True)
