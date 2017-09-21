# bricks_from_bins_to_kits

## Saved Models

This folder contains the trained convolutional neural net models which
are the core of the classifier system.  There is one set for color images
and another for grayscale.

At this time, the saved models are .h5 files from the Keras-TensorFlow
system, and are too big to upload to GitHub.  What I have done is saved the
models in an "S3 bucket" on AWS -- just follow the links below and save these
files in this folder

https://s3-us-west-2.amazonaws.com/two-squared-sigma-x-2/keras_inception_all_re-trained-grey.h5

https://s3-us-west-2.amazonaws.com/two-squared-sigma-x-2/keras_inception_all_re-trained.h5


Also saved in this folder are pickled label dictionaries. These connect the
indices of the model outputs to brick name labels, and were each generated with
their associated TensorFlow models.
