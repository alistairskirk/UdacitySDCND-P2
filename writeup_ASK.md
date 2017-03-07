## Traffic Sign Recognition Project
#### Alistair Kirk January 2017 Cohort
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[scatter]: ./outputfigs/InputDataScatter.png "Scatter Plot Image of Input Data"
[spectral1]: ./outputfigs/SpectralOutput1.png "Spectral Analysis 1"
[spectral2]: ./outputfigs/SpectralOutput2.png "Spectral Analysis 2"
[jitter]: ./outputfigs/PP-Jitter.png "Pre-Process Jittering"
[rotation]: ./outputfigs/PP-Rotation.png "Pre-Process Rotation"
[exposure]: ./outputfigs/PP-Exposure.png "Pre-Process Exposure"
[predictions]: ./outputfigs/inputpredictions_real.png "Real Life Predictions"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/alistairskirk/UdacitySDCND-P2/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

Number of training examples = 31367
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a scatter plot showing the number of samples per feature. There is a significant difference in quantity between the different classes which will likely guide the neural network to either favour those more numerous examples, or at least not be very good at classifying those that are least numerous. An effort to consider for future studies would be to generate additional synthetic data for those low quantity features and try to equalize the data set. This has not been considered in this project.

The spectral images give an idea of what the input images look like in different colour intensities. Initially there was a plan to try and use the colour channels as additional feature information but only grayscale images were considered due to time constraints.

![Scatter Plot Image of Data][scatter]
![Spectral1][spectral1]
![Spectral2][spectral2]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the code cells [4,5,6,7,8] of the IPython notebook.

As a first step, I added a jitter function to distort a randomly selected, but defined, fraction of the input data set and generate new synthetic data. This was based on the suggestion from the LeNet architecture paper. Comparisons of different trial runs suggest that this does not have a strong effect on the overall accuracy, but does help the model reach a high validation accuracy sooner.

Here is an example of a traffic sign image before and after jitter.

![Jitter Example][jitter]

Secondly I generate additional synthetic data by applying a random rotation to a randomly selected set of input data. I created this function because all signs except keep left/right, should be rotationally invariant, and will help classifying the roundabout sign. Unfortunately this may be adding unneccessary complexity and not generating enough data for the model to reap the benefits. Comparisons of different trial runs using different rotational angles and intensities suggest that the final accuracy in the test case cannot be conclusively improved, and is not worth the additional processing time required, but it is included for the record and future experimentation.

Here is an example of a traffic sign image before and after rotation.

![Rotation Example][Rotation]

Thirdly I normalized all images to grayscale using OpenCV function cv2.COLOR-RGB2GRAY. The grayscale results were then normalized to between 0 and 1 by dividing all elements by 255. This normalization was preferred to keep the values positive as opposed to centering around 0, due to some issues with how the exposure algorithms work (it was suggested in the course material to subtract 128 and then divide by 128 to normalize between +- 1 around 0, but the exposure algorithm requires values > 0).

As a last step I corrected all the images for exposure by using skimage import exposure algorithm. This was shown to improve the validation and test accuracy by a few extra points. The exposure correction algorithm is quite time consuming, so a progress bar was added.

Here is an example of a traffic sign image before and after grayscaling and normalization, and exposure correction.

![Exposure Example][exposure]

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by using sklearn's train_test_split as shown in the 8th cell of the notebook.

My final training set had 51897 number of images. My validation set and test set had 12975 and 12630 number of images.

The previous section (found above) describes the different ways I augmented the data set, with before and after images (using Jitter and Rotation to increase the data set size). 

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the tenth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28*28*6 	|
| RELU					| RELU Activation								|
| Dropout				| 50% Dropout after activation					|
| Max pooling	      	| 2x2 stride, VALID padding, outputs 14x14x6	|
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 10*10*16 	|
| RELU					| RELU Activation								|
| Dropout				| 50% Dropout after activation					|
| Max pooling	      	| 2x2 stride, VALID padding, outputs 5x5x16		|
| Flatten 			    | Flatten all layers to connect 400x1			|
| Fully connected		| Input = 400 Output = 120						|
| RELU					| RELU Activation								|
| Dropout				| 50% Dropout after activation					|
| Fully connected		| Input = 120 Output = 84						|
| RELU					| RELU Activation								|
| Dropout				| 50% Dropout after activation					|
| Fully connected		| Input = 84 Output = 10						|
|						|												|
 
This architecture was based on the LeNet architecure that was supplied in the course material. I applied RELU activation layers and dropout between each set at a rate of %50 based on recommendations from the course material. Time constraints prohibited experimentation with different architectures but in the future I would like to investigate the effect of changing dropout rate and to try and further understand how to integrate parallel structures (e.g. where the first conv layer output it directly passed to the final fully connected layer.

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the [11 and 12] cell of the ipython notebook. 

To train the model, I used an Adam Optimizer as was used in the classical LeNet architecture. 
Batch size was 128, with 200 epochs set as the upper limit. The learning rate was set at 0.001.

I experimented with different learning rates, even changing learning rates depending on the accuracy of a given epoch, but that did not seem to improve the final results and only served to increase the training time.

I introduced a set of boolean conditions that if the training accuracy plateaued after a certain point it would break. This methodology is not advisable though as it likely not possible to properly guage the effect of tweaking hyper-parameters on the final accuracy. In the future I would investigate cross-entropy values as a possible kickout value as suggested by peers and research online.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training/validation set accuracy of 96.3%
* test set accuracy of 94.9%
* real world test accuracy: 90% (9/10)

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

I selected the LeNet architecture as a basis, as it was used during the course material and I understand the basic concepts. My first week was spent assembling this classical architecture and ensuring that I could get the training and validation functions to work. I believed that the LeNet architecture was applicable because there are many similarities between the concept of hand writtens number classification and traffic sign classification, such as a fairly rigid and consistent visual structure and translational invariance.

My initial tests had an accuracy of approximately 90% or lower on the test set, and I sought to improve this by adding RELU and dropout between all layers of the LeNet architecture. I had no reason to prefer this approach initially other than guessing but it seemed to improve the final results to the low 90's. 

I attempted to further improve the model by using similar methods found in the provided LeNet architecture paper, such as the jittering of the training images to generate more synthetic data, and to apply an exposure correction because there were a significant number of images with poor lighting. This significantly improved my test set accuracy upwards of 96%.

I tried to refine the model further by adding a rotational pre-processing technique, where additional synthetic data was created by rotating some of the training images by 90 degrees, as most traffic signs should be rotationally invariant (except the keep ahead, keep left and right signs in this dataset, which are deliberately stripped from rotation). This marginally improved the test accuracy in some experiments. 

Additionally, I believe that a large source of error comes from the supplied training set having a large difference between the amount of sample data per feature. There were approximately 2,250 examples for Type 1 (30 km/h speed limit) and 250 examples for Type 0 (20 km/h speed limit). This can help explain the error in the real life test of the model (explained later here) where the model accurately predicts the 30 km/h sign but does not do so well for the 20 km/h sign. In the future I would look at improving the training data set by equalizing the number of samples per feature.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are ten German traffic signs that I found on the web, including their predictions:

![Real Life Predictions][predictions] 

The first image might be difficult to classify because as explained previously, the number of samples for 20 km/h were an order of magnitude less than the highest number of samples per feature.

The classifier also sometimes had trouble telling the difference between the Pedestrian or Children Crossing signs, and the Traffic Signal/Caution Ahead (Exclamation Point) signs, I believe it is because they look very similar, especially at 32x32 resolution.

An interesting future study would be to increase the resolution of the input images to 64x64 and examine the effect on the accuracy.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the [18] cell of the Ipython notebook.

See prediction image above that shows the images and their predictions. The roundabout sign is clearly challenging, and I suspect that more samples of the roundabout in different orientations would help to improve the image recognition capability for that particular sign.

The model was able to correctly guess 9 of the 10 traffic signs, which gives an accuracy of 90%. This compares favorably to the accuracy on the test set.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for outputting probability predictions on my final model is located in the [22nd] cell of the Ipython notebook.

Here I comment on some interesting features instead of all the images, as there were 10:

For the first image, we expected a 20 km/h speed limit, which was successfully detected. It is interesting to note the lesser probable answers were still speed signs.

The second image follows the same logic as the first.

It is interesting to note that the algorithm correctly detects Dangerous Curve Left (3rd Image), and it has a relatively high second guess, at ~65% confidence that it is a Bumpy Road Sign. It is also interesting that when performing a google image search for Bumpy Road German Sign, that a Dangerous Curve Left sign comes up also. 

I could do much more on this project, but it is already overdue, so if you're still reading, thank you very much for your time!