from flask import Flask, render_template, request, jsonify

import matplotlib.pyplot as plt
plt.style.use('ggplot')

from PIL import Image, ImageOps
import numpy as np

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
    '''
    # THE FOLLOWING 5 LINES OF CODE ARE UNCHANGED FROM load_model_and_go.py
    camera = pix.initialize_camera()
    # Input a file name = brick shape: e.g. 3021
    pic_label = raw_input('Type label (integer as file name):')
    extension, filename = pix.increment_filename(pic_label,extension=1)
    one_pic_X = pix.keep_shooting_until_acceptable(camera,filename)
    del(camera)
    '''
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

@app.route('/camera', methods=['GET'])
def camera():
    # At the moment, the camera.html file is also on
    # http://html.camera.test.s3-website-us-west-2.amazonaws.com/
    return render_template('camera.html')



@app.route('/solve', methods=['POST'])
def solve():
    user_data = request.json
    JSON_input_values = user_data['JSON_INPUT_KEY']
    model_output = _predictor_model(JSON_input_values)
    return jsonify({'root_1': model_output})


def _predictor_model(model_input):
    model_output = model_input+1.0
    return model_output


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

    app.run(host='0.0.0.0', threaded=True)
