from flask import Flask, render_template, request, jsonify

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from PIL import Image, ImageOps
import numpy as np
import cv2
import base64
from io import BytesIO
import StringIO

import cPickle as pickle
import src.picture_stuff as pix
import load_and_go as ld
import uuid


from keras.models import load_model

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/about', methods=['GET'])
def about():
    '''Placeholder - stub only
    '''
    return render_template('about.html')

@app.route('/2_take_pictures', methods=['GET'])
def take_pictures():
    '''
    The core functionaly for classifying photos starts on this page and
    the calls to
    '''
    return render_template('2_take_pictures.html')

@app.route('/3_login', methods=['GET'])
def login():
    '''Placeholder - stub only
    '''
    return render_template('3_login.html')

@app.route('/4_new_user', methods=['GET'])
def new_user():
    '''Placeholder - stub only
    '''
    return render_template('4_new_user.html')

@app.route('/5_make_brick_list', methods=['GET'])
def make_brick_list():
    '''Placeholder - stub only
       This page will eventually call the functions to compare a list
       of brick ID numbes to full kits in order to determine which kits
       they are likely from.
    '''
    return render_template('5_make_brick_list.html')

@app.route('/cam1', methods=['GET'])
def cam1():
    '''At the moment, the camera.html file is also on
       http://html.camera.test.s3-website-us-west-2.amazonaws.com/
       And I'm not actually using it for anything.
    '''
    return render_template('cam1.html')


@app.route('/classify', methods=['POST'])
def classify():
    '''This page provides the calls to all the photo-to-ID functionality.
       Most calculation is in
            src.picture_stuff as pix
            src.load_model_and_go as ld
       User types in a label for the list, which is passed back through
       json.  Photo is processed and displayed through matplotlib.
    '''
    # See static/app.js for source of JSON_input_label and JSON_pic
    # They come from the get_input_label() function called by a click on
    # button#classify
    pix.clear_old_pictures()
    user_data = request.json

    #one_pic_X will be the picture from <img id="photo">
    raw_data = user_data['JSON_pic']

    # There is probably a better way to do this, but in order to convert
    # the picture to the numpy array needed by the Keras/TensorFlow model,
    # I had to convert it through several intermdiate steps.
    intermediate = raw_data.split(',')[1]
    base64_X = base64.decodestring(intermediate.encode('utf-8'))
    tempBuff = StringIO.StringIO()
    tempBuff.write(base64_X)
    tempBuff.seek(0) #need to jump back to the beginning before handing it off to PIL
    X_image = Image.open(tempBuff)
    # Finally, after raw_date --> base64 --> StringIO --> Image --> numpy
    one_pic_X = np.array(X_image)

    # Crop and scale the picture to a 299 x 299 image.
    one_pic_X = pix.crop_and_scale(one_pic_X,use_gray=True,
                               image_dims=299,border_fraction=0.0)

    Image.fromarray(one_pic_X).save('test_image.png')

    pic_label = uuid.uuid4()
    extension, filename = pix.increment_filename(pic_label,extension=1)
    print "checkpoint: file name is ",filename

    # This just re-formats the image since the Keras/TensorFlow model
    # wants an RGB input of shape  N x 299 x 299 x 3.
    this_model_input = np.expand_dims(one_pic_X,axis=0)

    predict_gen = model.predict_on_batch(this_model_input)

    preds, weights = pix.make_one_prediction_list(
        predict_gen,label_dic,n_match=10)

    idx_preds = [picture_index_lookup[pred] for pred in preds]


    fig, ax = ld.plot_top_8(one_pic_X,pic_label,X,idx_preds,preds,weights);

    picfilename = ("static/images/saved_brick_predictions/" + str(pic_label) + ".png")
    #picfilename = "../saved_brick_predictions/" + pic_label + "_temp.png"

    model_output = "predicted ID = {}, weight = {:.1f}%".format(
                        preds[0], weights[0]*100)

    print "ok to here: model_output =", model_output
    return jsonify({'model_output': model_output, 'pic_x': picfilename})




if __name__ == '__main__':

    # first thing to do on start-up is load the model stuff, then run the app
    print "Please wait a moment while loading some data and models"
    use_gray = True
    X,y_list = ld.load_examples("data/",use_gray)
    filepath = 'saved_models/'

    print "Data loaded, loading models"
    model = ld.load_keras_model(filepath, use_gray)
    label_dic = ld.load_label_dictionary(filepath, use_gray)

    image_dims = 299
    border_fraction = .3
    picture_index_lookup = pix.picture_index_function(y_list)
    print "models loaded, ready to take pictures"

    print "Model test during setup"
    one_pic_X = X[0]
    this_model_input = np.expand_dims(one_pic_X,axis=0)
    print "Test OK: this_model_input size",this_model_input.shape, "data type", this_model_input.dtype
    predict_gen = model.predict_on_batch(this_model_input)

    # Set-up done -- fire up the Flask app!
    app.run(host='0.0.0.0', threaded=True)
