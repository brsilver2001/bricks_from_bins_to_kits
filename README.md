# bricks_from_bins_to_kits

Teaching a convolutional neural network through Keras &amp; TensorFlow to identify pictures from a mixed bin of Lego bricks and create an inventory of ID numbers to associate them with their original kits.


### Background
There are 2 kinds of kids who play with Legos.  One kind builds a kit, and
wants to leave the bricks together forever.  The other kind builds something,
rebuilds it into something else, and at the end of the day, all the bricks go
into a bin, never to be separated again.  Until now.

There are a number of places on the web where one can get pictures of Lego
bricks if you know their ID number.  For example:
  * lego.com
  * rebrickable.com
  * more stuff here

There are also a number of places where you can get the lego ID numbers if you
know the name and number of a kit.  For example:
  * someplace
  * someplace else

Until now, getting the reverse information was possible only by consulting an
expert, or doing a long series of Google searches.  Once this all works, that
will no longer be the case.  

At this time, only a small subset of all lego bricks can be identified.

### Step 1: Take a bunch of pictures
As a proof of concept, I have taken over 1600 photos of 56 different lego
bricks, under different lighting conditions, in different positions, and with
different backgrounds.  Over the years, Lego has produced over 10,000 different
brick shapes.  As a future exploration, I am looking at whether one can save
images from something like the Lego Digital Designer to generate training
data for the classifier.

### Step 2: Train a convolutional neural network
To manage the training of the convolutional neural network, I am using the
Keras front-end with TensorFlow back-end.  Initially, all training and testing
was run and saved through Jupyter Notebooks.

After several iterations, I concluded that training a model from scratch was
not going to be feasible with the resources I had available.  I therefore
turned to a partially pre-trained model.

As a base model, I am using Inception V3 with additional layers attached to
the exit layer.  Each incarnation of training was done on a MacBook Pro with
16 GB RAM and 2.5 GHz Intel Core i7 processor.  Approximately 5 hours of
training was required to reach over 99% correct classification on the training
data set (80-20 split) and 85% correct classification on the test set.

### Step 3: Pipeline for new pictures


### Step 4: Construction of usable applications
