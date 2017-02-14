###Project Steps

#### These are the steps by which I implemeted my vehicle detection pipeline (using custom built CV features).

1. Choose the feature representation that best serves the purpose of distinguishing between the two classes - cars and not cars - For this, I chose 3 feature representation methods and concatenate all 3 to form a single feature vector.
  * Spatial arrangement of color in the image - HSV color space
  * Histogram of color in the image - HSV color space
  * HOG (Histogram of Gradient Features) - All 3 channels of the input.
2. Train a linear classifier (SVM chosen here since it has been shown to work well with HOG features in literature) for the features obtained above.
3. Given a new test image, run multiple proposals(windows) and classify them proposals as one or the other class.
4. To ensure not to miss out on detecting a car object in the image, the proposals checked for 'carness' have to be of varying sizes and spread throughout the viable region of the image where we expect to see cars.
5. Using a simple heatmap technique sum up the multiple overlapping detections (bound to happen because of step 4) and obtain a maximal window which summarizes all overlapping windows for each object.
6. Once the image processing pipeline is ready, process the test video.

#### The output images at different stages of the pipeline (for a randomly chosen test image) can be found in output_images_from_pipeline directory.

#### The vehicle-detection.ipynb notebook is self contained with code and writeup.

The dataset is provided by Udacity. Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train the classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.
