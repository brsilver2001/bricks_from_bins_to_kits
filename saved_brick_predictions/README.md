# bricks_from_bins_to_kits

## Saved prediction results

This folder contains results of classification process.

For each photo taken, results are a set of weights corresponding to each of the
56 classes, on a scale of 0-100%.  Since a completely random prediction would
give each class around a 2% weight, anything less than around 3% can probably
be treated as "not in the picture".

The program displays the original picture next to the top 8 predictions, showing
photo exemplars and brick ID numbers as well as their weights.  

![](3_bricks_temp.png)

In the example shown, the most prominent, high-contrast image is the arch, which
is also the only one completely in the frame.  The classifier correctly identifies
it as [brick 4743](https://brickset.com/parts/design-4743).
The next highest prediction is a different arch -- the program frequently gets
these confused.  After that, we have correct classifications of the two other
bricks in the frame, followed by increasingly low-order guesses.
