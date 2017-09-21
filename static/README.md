# bricks_from_bins_to_kits

## Static files

This folder stores the supplementary css, Javascript, and image files used
by app.py.

Within the *static/images/saved_brick_predictions/* folder are all the images
stored by the classifier showing new pictures, predicted brick IDs and weights,
and the exemplar photos for each prediction.  NOTE: Due to limited space, the
 *app.py* file checks this directory after each classifier call, and deletes
 older images to save space.
