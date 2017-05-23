**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Example of one of each of the `vehicle` and `non-vehicle` classes displayed in output on cell XXXX of IPython notebook.

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Example using HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`is displayed in cell XXXX of IPython notebook.


####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and finally settled for below parameters through trial and error and through some help from the Udacity forums.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a SVM using default rbf kernal and probability. I got accuracy of about 99.55% on test set. I also used different combination of hog features, color hist and Spatial Binning and then scaled them accordingly to form the train and test sets.  

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search on bottom half of the image since car are not likely to found in upper half of image. I used three different window sizes. Function for Sliding window search is present in cell XXXX of IPython notebook. pipeline is present in cell XXXX of IPython notebook, which uses three different window size searches. 

For small windows :-

      1) Window Size - (64, 64)
      2) X Start Stop - (640, 1280)
      3) Y Start Stop - (300, 450)
      4) Overlap - (0.6, 0.6)

For medium windows :-

      1) Window Size - (96, 96)
      2) X Start Stop - (640, 1280)
      3) Y Start Stop - (300, 500)
      4) Overlap - (0.8, 0.8)

For large windows :-

      1) Window Size - (128, 128)
      2) X Start Stop - (640, 1280)
      3) Y Start Stop - (400, 550)
      4) Overlap - (0.85, 0.85)


![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on using Spatial Binning, HOG features plus histograms of color in the feature vector, which provided a good result. Examples of final output is displayed in cell XXXX of IPython notebook. 

Below parameters are used for the Spatial Binning, HOG features plus histograms of color.

      1) Color Channel - RGB
      2) Orientation bins - 8
      3) Pixels per cell - 8
      4) Cells per block - 2
      5) Histogram Bins - 48
      6) Histogram Bins Range - (0, 256)
      7) Spatial Image Size - (16, 16)


### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  
Example of heatmap and boundbox overlaid on the test images is displayed in of cell XXXX of IPython notebook. 

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Below are the few probelms I faced.

1) matplotlib reads png images on scale of 0 to 1. To add to the confusion, matplotlib image will read .jpg images in on a scale of 0 to 255. Eventhough this was mentioned as caution in the project tips, I missed it initially and it created problem. I was getting good accuracy with test result with SVM classifier. But on test images I was unable to find the vehicles.
2) Removing false positives is another problem I faced. I have used some techniques mentioned above to remove false positives but still it's not compeletely removed. 

Pipleline will face problems with more complicated situations. I beleive combining this with deep learning and doing further tuning with parameters and combination of hog, color space etc will be able to improve the pipeline.

