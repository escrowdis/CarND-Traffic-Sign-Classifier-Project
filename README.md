# **Traffic Sign Recognition**

[//]: # (Image References)

[hist_train]: ./img/hist_train.png "Histogram of Training Set"
[hist_valid]: ./img/hist_valid.png "Histogram of Validation Set"
[hist_test]:  ./img/hist_test.png "Histogram of Test Set"
[acc_lenet_rgb_valid]: ./img/acc_valid_rgb.png "Accuracy of Validation using LeNet with RGB Color Space Image"
[acc_lenet_grey_valid]: ./img/acc_valid_grey.png "Accuracy of Validation using LeNet with grey scale Image"
[signs_rgb]: ./img/signs_rgb.png "Signs with RGB Color Space"
[signs_grey]: ./img/signs_grey.png "Signs with Grey Scale"
[signs_test_rgb]: ./img/test_signs_rgb.png "Signs for Testing with RGB Color Space"
[result_signs_test]: ./img/test_signs_result.png "Result of Testing Signs"

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?
    **A:** 34799
* The size of the validation set is ?
    **A:** 4410
* The size of test set is ?
    **A:** 12630
* The shape of a traffic sign image is ?
    **A:** 32x32x3
* The number of unique classes/labels in the data set is ?
    **A:** 43, from 0 to 42.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. These charts show that size of dataset is not even at all which might cause model not robust enough. I think to generate more data by rotating, tilting, and translating the images is a better way to make a even dataset for the model which I do not have enough time to accomplish this now.

![][hist_train]

![][hist_valid]

![][hist_test]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I used LeNet to train models with RGB color and gray scale images to decide which type to use. Both performances look similar to each other under same parameters (only Convolution 1's input dimension is different by image depth). I prefer to use RGB as default which may contain more details for the traffic sign. Because the color represents different meaning on the sign such as red color was to make road user be cautious more than the yellow, blue one.

[parameters]
epoch: 20
batch_size: 128
learn_rate: 0.001
mu: 0
sigma: 0.1
keep_prob: Not used

|RGB|Gray|
|---|---|
| ![][acc_lenet_rgb_valid] | ![][acc_lenet_grey_valid] |

Here is an example of a traffic sign image before and after grayscaling.

![][signs_rgb]
![][signs_grey]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x64 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 26x26x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 13x13x64 				|
| Dropout               | keep_prob: 0.5                                |
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 11x11x32 	|
| RELU					|												|
| Dropout               | keep_prob: 0.5                                |
| Flattern              | Output 3872                                   |
| Fully connected		| Output 512 									|
| RELU					|												|
| Fully connected		| Output 256 									|
| RELU					|												|
| Dropout               | keep_prob: 0.5                                |
| Fully connected		| Output 43 (classes num)    					|
| Softmax				|              									|

with parameters listed below:
epoch: 30
batch_size: 128
learn_rate: 0.001
mu: 0
sigma: 0.1
keep_prob: 0.5

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, there are some critical parameters need to be concerned: epoch, batch size, learning rate, network depth, keep probability,...etc.
In epoch, accuracy was enhanced rapidly in first 5 epochs, and the rest of epochs is going to hit 100%. I've tested from 10 to 100. After 25, the accuracy is toggling. I've tried 100, 128, and 256 on batch size which did not show significant change from accuracy, but you can see the accuracy went up smoother when batch size is bigger. Learning rate and keep probability just use default value 0.001 and 0.5, respectively.
After that, the most important one is the depth of network. First of all, I tried LeNet to train the data, but the performance was stuck around 90%. Then I assumed it's under-fitting, so I expanded first layer's depth which make the performance worse. I've generated lots of low performance models during adjusting the network... After all, I think first layer should have deep depth, and there should have some max pooling in the middle layers for data reduction.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
    **A:** 99.99% (0.9998850542778441)
* validation set accuracy of ?
    **A:** 98.23% (0.9823129298735638)
* test set accuracy of ?
    **A:** 97.08% (0.970783856211441)

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
    **A:** I'm not expert in this field, so I try to use LeNet at beginning.
* What were some problems with the initial architecture?
    **A:** Networks may differ by different problem. Which make LeNet hard to get higher performance on traffic sign detection. I've made lots of trivial changes which usually make the network dump trashes.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
    **A:** At first, I think LeNet's depth is not enough which might lead to underfitting, so I tried to go deeper which made the accuracy thrived as expected. Then go on trial and error hell.....
* Which parameters were tuned? How were they adjusted and why?
    **A:** As mentioned above, lots of parameters were adjusted. The most important one is the depth of each network. I assumed the machine thinks as human when met something new at the first time. It will observe some general like appearances (shape, color,...etc), and then move on details like symbols or corners. To observe these information, I decided to set first layer's depth to 64 to provide as much as information. After that, I tried to shrink the depth gradually in convolution networks but it looks not very well, maybe the wrong tuning. Furthermore, a convolution network was added at layer4 but removed which did not make performance significantly better.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
    **A:** I'm not pretty sure about this@@. As I said, this is a trial and error hell.... But I'm still not pretty sure how to manipulate dropout function which sometimes destroyed my network....

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][signs_test_rgb]

I found a STOP sign with some taped words which may fail to recognize under high resolution. I assumed that one will be the only fail.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alter][result_signs_test]

The model was able to correctly guess 7 of the 11 traffic signs, which gives an accuracy of 63.6%. The result was not good as test set. Surprisingly, #0 was failed but #2, assumed the hardest one, was detected successfully. I think it's affected by how the images were cropped. The signs in the dataset was smaller than those I downloaded from the website.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

Most probabilities of top 1 of test images were close to 1.0 (unknown).

There's a list showing top three soft max probabilities of test set.

|Image No.|GT|Prob.|Top 1|Prob.|Top 2|Prob.|Top 3|
|-----------|----------------------------------------|------------|-----------------------------------------|-----------|-----------------------------------------|-----------|--------------------------------------------|
|0          |Speed limit (20km/h)                    | 0.998469   |Speed limit (30km/h)                     |0.001259   |Speed limit (50km/h)                     |0.000144   |Speed limit (20km/h)                        |  
|1          |Stop                                    | 1.000000   |Stop                                     |0.000000   |No entry                                 |0.000000   |Speed limit (30km/h)                        |
|2          |Stop                                    | 0.569766   |Stop                                     |0.423999   |Yield                                    |0.005624   |Priority road                               |
|3          |Double curve                            | 0.916270   |Dangerous curve to the left              |0.067865   |Speed limit (60km/h)                     |0.009600   |Right-of-way at the next intersection       |
|4          |Road work                               | 0.999992   |Road work                                |0.000008   |Right-of-way at the next intersection    |0.000000   |Beware of ice/snow                          |
|5          |Road work                               | 1.000000   |Road work                                |0.000000   |Dangerous curve to the right             |0.000000   |No passing for vehicles over 3.5 metric tons|
|6          |Children crossing                       | 0.999322   |Children crossing                        |0.000678   |Right-of-way at the next intersection    |0.000000   |Pedestrians                                 |
|7          |Speed limit (50km/h)                    | 0.999998   |Speed limit (30km/h)                     |0.000002   |Speed limit (50km/h)                     |0.000000   |Speed limit (20km/h)                        |
|8          |Speed limit (70km/h)                    | 0.995594   |Speed limit (30km/h)                     |0.002369   |Speed limit (20km/h)                     |0.000990   |Speed limit (50km/h)                        |
|9          |Speed limit (70km/h)                    | 0.861492   |Speed limit (70km/h)                     |0.119362   |Speed limit (30km/h)                     |0.009179   |Speed limit (50km/h)                        |
|10         |Speed limit (120km/h)                   | 0.758382   |Roundabout mandatory                     |0.098380   |Bicycles crossing                        |0.061641   |Road work                                   |

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
