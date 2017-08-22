## Vehicle Detection

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car.png
[image2]: ./output_images/notcar.png
[image3]: ./output_images/hog.png
[image4]: ./output_images/hog_car.png
[image5]: ./output_images/features.png
[image6]: ./output_images/sliding_window_example.png
[image7]: ./output_images/sliding_window_scales.png
[image8]: ./output_images/heatmap.png
[image9]: ./output_images/labelmap.png
[image10]: ./output_images/test_output.png

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 3rd-6th cell of the Jupyter notebook `find_car.ipynb`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1] ![alt text][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

![alt text][image3]

Here is an example HOG for a car image using the `YCrCb` color space and HOG parameters of `orientations=11`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image4]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters. From my observation, the main factor is the colorspace. I tried RGB, HLS, HSV, LUV, YUV and YCrCb along with `orient` 9 and 11, `pixels_per_cell` (8, 8) and (16, 16). The linear SVM I trained later showed that this particular set of parameters performed the best with an accuracy of 98.93%

```
CSPACE = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
ORIENT = 11
PIX_PER_CELL = 8
CELL_PER_BLOCK = 2
HOG_CHANNEL = 'ALL' # Can be 0, 1, 2, or "ALL"
```

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM `LinearSVC()` using the HOG, spatial and hist features obtained through the above mentioned parameters. I initially tried with only HOG and no color features. With the above parameters, it achieved an accuracy of over 97% already. Then I experimented with color features and increased the accuracy by 1-2%. In real use cases where performance is more important than a 1-2% accuracy improvement, the HOG may be good enough.

The code can be found in the notebook in the `Train model` section. The features for an example image is shown below

![alt text][image5]

The HOG, spatial and hist features have different scales and must be normalized before training. I used the `StandardScaler()` to normalize them to be zero-mean and unit-variance. This is an essential step for any model training. When doing validation and testing, **the validation and test data should also be normalized before making a prediction**. The corresponding code can be found in the `Train model` section and the `Find cars` section.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search runs square patches of the image with overlap and uses the classifier to check if the patch is a car image. Here’s an example using a 96 x 96 patch and 50% overlap:

![alt text][image6]

Further improvements can be made since there are only certain regions in the image where we want to look for cars. I excluded the top part which is the sky, and chose 4 scales: 1, 1.5, 2 and 3.5 because they appear to give relatively good results in the heatmap voting step.

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.  

![alt text][image7]

Here's an example result showing the heatmap and labeled map

![alt text][image8]
![alt text][image9]

#### 3. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on scale 1, 1.5, 2 and 3.5 using YCrCb ALL-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a relatively nice result.  Here are some example images:

![alt text][image10]

---

### Video Implementation

#### Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](https://youtu.be/qNMrJdLmYyk)

---

### Discussion

I learned about the traditional computer vision approach using HOG, color features and SVM to classify/segment cars out of images. It's quite encouraging that even simple methods like this can do pretty well in this task. However, the disadvantages of this approach are also obvious -- the features do not represent the essense of car images well enough and the model is not complex enough to capture the levels of that essence. More robust methods are needed. Luckily, we have deep neural nets to address that. More complex and automated feature extraction and better models are available. 

From some readings, it appears the state-of-the-art approach is YOLO (You Only Look Once). There is a great writeup and implementation produced by a fellow SDCND student [here](https://medium.com/@ksakmann/vehicle-detection-and-tracking-using-hog-features-svm-vs-yolo-73e1ccb35866). I'm interested in implementing this approach in my future efforts.


