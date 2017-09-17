# bricks_from_bins_to_kits

Teaching a convolutional neural network through Keras &amp; TensorFlow to identify
pictures from a mixed bin of Lego bricks and create an inventory of ID numbers
to associate them with their original kits.


### Background
There are 2 kinds of kids who play with Legos.  One kind builds a kit, and
wants to leave the bricks together forever.  The other kind builds something,
rebuilds it into something else, and at the end of the day, all the bricks go
into a bin, never to be separated again.  Until now.

There are a number of places on the web where one can get pictures of Lego
bricks if you know their ID number.  For example:
  * https://shop.lego.com/en-US/Pick-a-Brick
  * https://rebrickable.com/
  * https://brickset.com/

There are also a number of places where you can get the lego ID numbers if you
know the name and number of a kit.  However,until now, getting the reverse
information was possible only by consulting a subject matter expert, or doing
a long series of Google searches.  Once this all works, that will no longer be
the case.  

### Step 0: Set up and install software
See the /documents folder.

### Step 1: Take a bunch of pictures
As a proof of concept, I have taken over 1600 photos of 56 different lego
bricks, under different lighting conditions, in different positions, and with
different backgrounds.  Over the years, Lego has produced over 10,000 different
brick shapes.  As a future exploration, I am looking at whether one can save
images from something like the Lego Digital Designer to generate training
data for the classifier, or whether I can scrape data from elsewhere on the web,
of whether I just have to slog through and take another 100,000 photos.

### Step 2: Train a convolutional neural network
To manage the training of the convolutional neural network, I am using the
Keras front-end with TensorFlow back-end.  Initially, all training and testing
was run and saved through Jupyter Notebooks.

After several iterations, I concluded that training a model from scratch was
not going to be feasible with the resources I had available.  I therefore
turned to a partially pre-trained model.

As a base model, I am using Inception V3 with additional layers attached to
the exit layer.  Each incarnation of training was done on a MacBook Pro with
16 GB RAM and 2.5 GHz Intel Core i7 processor, and ran for around 4-8 hours.

As it turns out, Inception V3 likes colors.  It frequently would mis-classify
a blue brick as another blue brick, for example.  After re-processing all the
photos in grayscale, classification by shape alone was much better.

With the new grayscale model,  approximately 5 hours of training was required
to reach over 99% correct classification on the training data set (80-20 split)
and 85% correct classification on the test set.  The model was then re-trained
on the whole set for use with new pictures.

### Step 3: Pipeline for new pictures

There are plenty of ways to take photos to train on.  I set up a notebooks
which crops and scales to the 299 x 299 size that Inception V3 prefers, and
saves all photos under a name and number based on the brick ID.  Thus the first
3 shots for brick 3003 are 3003-001.jpeg, 3003-002.jpeg, and 3003-003.jpeg.
The notebook below check whether pictures already exist under that brick ID,
and increments the index as needed.
        Photo_lego_pipeline_repeats.ipynb

### Step 4: Construction of usable applications

Several incarnations of the wrapper for the model exist:

1. In Jupyter notebooks, there is a well-annotated version which stands alone:
        load_model_and_GO.ipynb
2. In Jupyter notebooks, there is a version which calls most of its functions
   from the Python .py files:
        src/load_model_and_GO_from_py_files.ipynb
3. You can run the whole set from the Python .py files:
        src/load_model_and_go.py
4. Work in progress -- I'm working on a web app that moves the functionality
   of the camera to a remote client, uses their webcam or phone camera,
   and passes the picture back to the classifier through Javascript, with
   the server side handled through Flask.
        app/app.py
