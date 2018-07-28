# Writeup Behavioral Cloning Project

## Project Goals

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image_nvidia_model]: ./doc/img/cnn-architecture-624x890.png "Model Visualization"
[image_left_camera]: ./doc/img/left_2018_07_19_22_52_45_926.jpg "Left Camera Image"
[image_center_camera]: ./doc/img/center_2018_07_19_22_52_45_988.jpg "Center Camera Image"
[image_right_camera]: ./doc/img/right_2018_07_19_22_52_45_926.jpg "Right Camera Image"
[image_cropped_road]: ./doc/img/center_2018_07_19_22_52_45_988_cropped.jpg "Cropped Raod"

[image_yuv_luma]: ./doc/img/center_2018_07_19_22_52_45_988-YCbCr_ITU_R709_luma.jpg "Luma Channel"
[image_yuv_blue]: ./doc/img/center_2018_07_19_22_52_45_988-YCbCr_ITU_R709_blueness.jpg "Blue Channel"
[image_yuv_yellow]: ./doc/img/center_2018_07_19_22_52_45_988-YCbCr_ITU_R709_yellowness.jpg "Blue Channel"

[image_original_dist]: ./doc/img/historgram_original_distribution.png "Original distribution"
[image_normalized_dist]: ./doc/img/historgram_normalized_distribution.png "Normalized distribution"

[image_final_loss]: ./doc/img/loss.png "Training and Validation Loss"

[video_autonomous_driving]: ./doc/img/video-to-gif.gif "Autonomous Driving"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### Input Images.

The input images come from three cameras "mounted" on the left, center and right of the car.

![Left Camera Image][image_left_camera] ![Center Camera Image][image_center_camera] ![Right Camera Image][image_right_camera]

As the direction of the road should lead a driver, the images will be cropped to focus on the road parts.

![Cropped road][image_cropped_road]

The resulting images will be normalized and converted from RGB color space to YUV.

YUV images:
![Luma][image_yuv_luma] ![Blue][image_yuv_blue] ![Yelow][image_yuv_yellow]

#### Correction of steering angle

The recored steering angle of left and right camera images must be corrected. It looks like the car is on the right side of the road, if you look at a camera image from the left side, even the car in the center of the raod.

I have found the best correction value be experiment. I have build a simple network and choose the best correction value by traing a model with a range of correction values and chose the one with the lowest validation loss.

**The correction value I have found is +/- 0.5**

#### Recording

I drove the simulated car several times on both tracks in forward direction and in reverse direction. Additonally, I recorded only curves and several "rescue" situations to bring the car back to the track.

**I collected finially 97701 training images**,consisting of left, center and right camera images.

#### Choosing images

Driving straight is by far overrepresented. Thus, I have shuffled the images and stopped accepping new images for certain steering angle bins.

The first histogram shows the original distribution, the second histogram shows the normalized distribution (with corrected, steering angles and additional flipped images)

![Original Destribution][image_original_dist] ![Normalized Distribution][image_normalized_dist]

**68213 left after filtering**

#### Image augumenation

I assumed that flipping images would be a usefull way increase the amount of training examples. I have tested this with a simple model. The validation lost decreased a bit, especially in the first epochs. Further it helped to have an even distribution of left and right turns.

**This doubled the amount of images to 136426 images**

#### Model search

I started with a very simple model, one convulution later, max pooling and a small fully connected network, to see effects of paramter changes quickly. I used also only a small subset of training data.

At the next step I compared the small network with a LeNet implementation. The learning converged way quicker and the finial validation loss outperformed the simple network (valiation loss: 0.0142 / validation loss: 0.0130).

The results from [Convolutional Neuronal Network (CNN) Architecture](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) from Nvidia looked promising. I wanted to see if the network configuration is also helpful for this car simulation. I didn't see yet an improvment with validation loss. But the car drove much better in autonomunous mode simulation.

All these experiments have in common that the models overfits with a small subset of image data. Giving more training data helped to reduce overfitting. Adding a dropout layer after the convolution layer decreased the performance instead.

#### Final Model

The final model consists for the follwing layers:

* Preprocessing
 * Image cropping to focus on the road.
 * Image normalization using per_image_standardization
 * Convert images to YUV color space
* [Convolutional Neuronal Network (CNN) Architecture](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) from Nvidia
 * Convolutional layer with 2x2 stride and 5x5 Kernel
   * 5 x 5 Kernel, Output Depth 24, Subsampling 2 x 2, Activation Function: Elu
   * 5 x 5 Kernel, OUtput Depth 36, Subsampling 2 x 2, Activation Function: Elu
   * 5 x 5 Kernel, Output Depth 48, Subsampling 2 x 2, Activation Function: Elu
 * Convolutional layer without stride and 3x3 Kernel
   * 3 x 3 Kernel, Output Depth 64, Activation Function: Elu
   * 3 x 3 Kernel, Output Depth 64, Activation Function: Elu
 * Fully Connected Network
   * Flatten
   * Layer 100 Nodes
   * Layer 50 Nodes
   * Layer 10 Nodes
   * Layer 1 Nodes

![Nvidial Nework Architecture][image_nvidia_model]

The training and validation loss descresses with this model like shown in the chart below.

![Training and Validation Loss][image_final_loss]

### Result Video

This is an excerpt of the autonomonous driving of track one.

![Autonomous Driving][video_autonomous_driving]

### Links
* [Convolutional Neuronal Network (CNN) Architecture](https://devblogs.nvidia.com/deep-learning-self-driving-cars/)
* [Udacity Self Driving Car Simulator](https://github.com/udacity/self-driving-car-sim)



