# MNIST Hand-Written Digits Classification using Artificial Neural Networks (ANN) in Python

<img align="center" src="images/banner-03.png" width="1000" >

## 1. Objective

The objective of this project is to develop an Artificial Neural Network (ANN) to classify hand-written digits using the widely used MNIST data set.

## 2. Motivation

The MNIST handwritten digit classification problem is a standard dataset used in computer vision and deep learning.

Although the dataset is effectively solved, it can be used as the basis for learning and practicing how to develop, evaluate, and use simple artificial neural networks for image classification from scratch. 

In this section, we shall demonstrate how to develop a simple artificial neural network for handwritten digit classification from scratch, including:

* How to prepare the input training and test data 
* How to deploy the model
* How to use the trained model to make predictions
* How to evaluate its performance

## 3. Data

The MNIST database of handwritten digits, is widely used for training and evaluating various supervised machine and deep learning models [1]:

* It has a training set of 60,000 examples
* It has test set of 10,000 examples
* It is a subset of a larger set available from NIST. 
* The digits have been size-normalized and centered in a fixed-size image.
* It is a good database for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting.
* The original black and white images from NIST were size normalized and resized to 28x28 binary images.

Sample images from the MNIST data set are illustrated next:
* There are significant variations how digits are handwritten by different people
* The same digit may be written quite differently by different people
* More significantly, different handwritten digits may appear similar, such as the 0, 5 and 6 or the 7 and 9.

<img align="center" src="images/MNIST-sample-images.png" width="1000" >

## 4. Development


In this section, we shall demonstrate how to develop a simple artificial neural network for handwritten digit classification from scratch, including:

* How to prepare the input training and test data 
* How to deploy the model
* How to use the trained model to make predictions
* How to evaluate its performance

* Author: Mohsen Ghazel (mghazel)
* Date: April 5th, 2021

* Project: MNIST Handwritten Digits Classification using Artificial Neural Networks (ANN):


The objective of this project is to demonstrate how to develop a simple artificial neural network to classify images of hand-written digits, from 0-9:


We shall apply the standard Machine and Deep Learning model development and evaluation process, with the following steps:

* Load the MNIST dataset of handwritten digits:
* 60,000 labelled training examples
* 10,000 labelled test examples
* Each handwritten example is 28x28 pixels binary image.


Build a simple ANN model
Train the selected ML model
Deploy the trained on the test data
Evaluate the performance of the trained model using evaluation metrics:
Accuracy
Confusion Matrix
Other metrics derived form the confusion matrix.

### 4.1. Part 1: Imports and global variables
#### 4.1.1. Standard scientific Python imports:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># numpy</span>
<span style="color:#800000; font-weight:bold; ">import</span> numpy <span style="color:#800000; font-weight:bold; ">as</span> np
<span style="color:#696969; "># matplotlib</span>
<span style="color:#800000; font-weight:bold; ">import</span> matplotlib<span style="color:#808030; ">.</span>pyplot <span style="color:#800000; font-weight:bold; ">as</span> plt
<span style="color:#696969; "># opencv</span>
<span style="color:#800000; font-weight:bold; ">import</span> cv2
<span style="color:#696969; "># tensorflow</span>
<span style="color:#800000; font-weight:bold; ">import</span> tensorflow <span style="color:#800000; font-weight:bold; ">as</span> tf
<span style="color:#696969; "># random number generators values</span>
<span style="color:#696969; "># seed for reproducing the random number generation</span>
<span style="color:#800000; font-weight:bold; ">from</span> random <span style="color:#800000; font-weight:bold; ">import</span> seed
<span style="color:#696969; "># random integers: I(0,M)</span>
<span style="color:#800000; font-weight:bold; ">from</span> random <span style="color:#800000; font-weight:bold; ">import</span> randint
<span style="color:#696969; "># random standard unform: U(0,1)</span>
<span style="color:#800000; font-weight:bold; ">from</span> random <span style="color:#800000; font-weight:bold; ">import</span> random
<span style="color:#696969; "># time</span>
<span style="color:#800000; font-weight:bold; ">import</span> datetime
<span style="color:#696969; "># I/O</span>
<span style="color:#800000; font-weight:bold; ">import</span> os
<span style="color:#696969; "># sys</span>
<span style="color:#800000; font-weight:bold; ">import</span> sys

<span style="color:#696969; "># check for successful package imports and versions</span>
<span style="color:#696969; "># python</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Python version : {0} "</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span>sys<span style="color:#808030; ">.</span>version<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># OpenCV</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"OpenCV version : {0} "</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span>cv2<span style="color:#808030; ">.</span>__version__<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># numpy</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Numpy version  : {0}"</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span>np<span style="color:#808030; ">.</span>__version__<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># tensorflow</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Tensorflow version  : {0}"</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span>tf<span style="color:#808030; ">.</span>__version__<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>


Python version <span style="color:#808030; ">:</span> <span style="color:#008000; ">3.7</span><span style="color:#808030; ">.</span><span style="color:#008c00; ">10</span> <span style="color:#808030; ">(</span>default<span style="color:#808030; ">,</span> Feb <span style="color:#008c00; ">20</span> <span style="color:#008c00; ">2021</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">21</span><span style="color:#808030; ">:</span><span style="color:#008c00; ">17</span><span style="color:#808030; ">:</span><span style="color:#008c00; ">23</span><span style="color:#808030; ">)</span> 
<span style="color:#808030; ">[</span>GCC <span style="color:#008000; ">7.5</span><span style="color:#808030; ">.</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span> 
OpenCV version <span style="color:#808030; ">:</span> <span style="color:#008000; ">4.1</span><span style="color:#808030; ">.</span><span style="color:#008c00; ">2</span> 
Numpy version  <span style="color:#808030; ">:</span> <span style="color:#008000; ">1.19</span><span style="color:#808030; ">.</span><span style="color:#008c00; ">5</span>
Tensorflow version  <span style="color:#808030; ">:</span> <span style="color:#008000; ">2.4</span><span style="color:#808030; ">.</span><span style="color:#008c00; ">1</span>
</pre>

#### 4.1.2. Global variables


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># -set the random_state seed = 101 for reproducibilty</span>
random_state_seed <span style="color:#808030; ">=</span> <span style="color:#008c00; ">101</span>

<span style="color:#696969; "># the number of visualized images</span>
num_visualized_images <span style="color:#808030; ">=</span> <span style="color:#008c00; ">25</span>
</pre>


### 4.2. Load MNIST Dataset

#### 4.2.1 Load the MNIST dataset

* Load the MNIST dataset of handwritten digits:
* 60,000 labelled training examples
* 10,000 labelled test examples
* Each handwritten example is 28x28 pixels binary image.


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Load in the data: MNIST</span>
mnist <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>datasets<span style="color:#808030; ">.</span>mnist
<span style="color:#696969; "># mnist.load_data() automatically splits traing and test data sets</span>
<span style="color:#808030; ">(</span>x_train<span style="color:#808030; ">,</span> y_train<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span>x_test<span style="color:#808030; ">,</span> y_test<span style="color:#808030; ">)</span> <span style="color:#808030; ">=</span> mnist<span style="color:#808030; ">.</span>load_data<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
</pre>

### 4.2. Explore training and test images:

#### 4.2.1. Display the number and shape of the training and test subsets:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># Training data:</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># the number of training images</span>
num_train_images <span style="color:#808030; ">=</span> x_train<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Training data:"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"x_train.shape: "</span><span style="color:#808030; ">,</span> x_train<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Number of training images: "</span><span style="color:#808030; ">,</span> num_train_images<span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Image size: "</span><span style="color:#808030; ">,</span> x_train<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">:</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>

<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># Test data:</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># the number of test images</span>
num_test_images <span style="color:#808030; ">=</span> x_test<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Test data:"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"x_test.shape: "</span><span style="color:#808030; ">,</span> x_test<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Number of test images: "</span><span style="color:#808030; ">,</span> num_test_images<span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Image size: "</span><span style="color:#808030; ">,</span> x_test<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">:</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>

<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># Training data:</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># the number of training images</span>
num_train_images <span style="color:#808030; ">=</span> x_train<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Training data:"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"x_train.shape: "</span><span style="color:#808030; ">,</span> x_train<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Number of training images: "</span><span style="color:#808030; ">,</span> num_train_images<span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Image size: "</span><span style="color:#808030; ">,</span> x_train<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">:</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>

<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># Test data:</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># the number of test images</span>
num_test_images <span style="color:#808030; ">=</span> x_test<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Test data:"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"x_test.shape: "</span><span style="color:#808030; ">,</span> x_test<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Number of test images: "</span><span style="color:#808030; ">,</span> num_test_images<span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Image size: "</span><span style="color:#808030; ">,</span> x_test<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">:</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>
</pre>


#### 4.2.2. Display the targets/classes:

* The classification of the digits should be: 0 to 9:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Classes/labels:"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">'The target labels: '</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#808030; ">(</span>np<span style="color:#808030; ">.</span>unique<span style="color:#808030; ">(</span>y_train<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>

<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Classes<span style="color:#44aadd; ">/</span>labels<span style="color:#808030; ">:</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
The target labels<span style="color:#808030; ">:</span> <span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span> <span style="color:#008c00; ">1</span> <span style="color:#008c00; ">2</span> <span style="color:#008c00; ">3</span> <span style="color:#008c00; ">4</span> <span style="color:#008c00; ">5</span> <span style="color:#008c00; ">6</span> <span style="color:#008c00; ">7</span> <span style="color:#008c00; ">8</span> <span style="color:#008c00; ">9</span><span style="color:#808030; ">]</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
</pre>


#### 4.2.3. Examine the number of images for each class of the training and testing subsets:

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Create a histogram of the number of images in each class/digit:</span>
<span style="color:#800000; font-weight:bold; ">def</span> plot_bar<span style="color:#808030; ">(</span>y<span style="color:#808030; ">,</span> loc<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'left'</span><span style="color:#808030; ">,</span> relative<span style="color:#808030; ">=</span><span style="color:#074726; ">True</span><span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
    width <span style="color:#808030; ">=</span> <span style="color:#008000; ">0.35</span>
    <span style="color:#800000; font-weight:bold; ">if</span> loc <span style="color:#44aadd; ">==</span> <span style="color:#0000e6; ">'left'</span><span style="color:#808030; ">:</span>
        n <span style="color:#808030; ">=</span> <span style="color:#44aadd; ">-</span><span style="color:#008000; ">0.5</span>
    <span style="color:#800000; font-weight:bold; ">elif</span> loc <span style="color:#44aadd; ">==</span> <span style="color:#0000e6; ">'right'</span><span style="color:#808030; ">:</span>
        n <span style="color:#808030; ">=</span> <span style="color:#008000; ">0.5</span>
     
    <span style="color:#696969; "># calculate counts per type and sort, to ensure their order</span>
    unique<span style="color:#808030; ">,</span> counts <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>unique<span style="color:#808030; ">(</span>y<span style="color:#808030; ">,</span> return_counts<span style="color:#808030; ">=</span><span style="color:#074726; ">True</span><span style="color:#808030; ">)</span>
    sorted_index <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>argsort<span style="color:#808030; ">(</span>unique<span style="color:#808030; ">)</span>
    unique <span style="color:#808030; ">=</span> unique<span style="color:#808030; ">[</span>sorted_index<span style="color:#808030; ">]</span>
     
    <span style="color:#800000; font-weight:bold; ">if</span> relative<span style="color:#808030; ">:</span>
        <span style="color:#696969; "># plot as a percentage</span>
        counts <span style="color:#808030; ">=</span> <span style="color:#008c00; ">100</span><span style="color:#44aadd; ">*</span>counts<span style="color:#808030; ">[</span>sorted_index<span style="color:#808030; ">]</span><span style="color:#44aadd; ">/</span><span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>y<span style="color:#808030; ">)</span>
        ylabel_text <span style="color:#808030; ">=</span> <span style="color:#0000e6; ">'% count'</span>
    <span style="color:#800000; font-weight:bold; ">else</span><span style="color:#808030; ">:</span>
        <span style="color:#696969; "># plot counts</span>
        counts <span style="color:#808030; ">=</span> counts<span style="color:#808030; ">[</span>sorted_index<span style="color:#808030; ">]</span>
        ylabel_text <span style="color:#808030; ">=</span> <span style="color:#0000e6; ">'count'</span>
         
    xtemp <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>arange<span style="color:#808030; ">(</span><span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>unique<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
    plt<span style="color:#808030; ">.</span>bar<span style="color:#808030; ">(</span>xtemp <span style="color:#44aadd; ">+</span> n<span style="color:#44aadd; ">*</span>width<span style="color:#808030; ">,</span> counts<span style="color:#808030; ">,</span> align<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'center'</span><span style="color:#808030; ">,</span> alpha<span style="color:#808030; ">=</span><span style="color:#008000; ">.7</span><span style="color:#808030; ">,</span> width<span style="color:#808030; ">=</span>width<span style="color:#808030; ">)</span>
    plt<span style="color:#808030; ">.</span>xticks<span style="color:#808030; ">(</span>xtemp<span style="color:#808030; ">,</span> unique<span style="color:#808030; ">,</span> rotation<span style="color:#808030; ">=</span><span style="color:#008c00; ">45</span><span style="color:#808030; ">)</span>
    plt<span style="color:#808030; ">.</span>xlabel<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'digit'</span><span style="color:#808030; ">)</span>
    plt<span style="color:#808030; ">.</span>ylabel<span style="color:#808030; ">(</span>ylabel_text<span style="color:#808030; ">)</span>
 
plt<span style="color:#808030; ">.</span>suptitle<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Frequency of images per digit'</span><span style="color:#808030; ">)</span>
plot_bar<span style="color:#808030; ">(</span>y_train<span style="color:#808030; ">,</span> loc<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'left'</span><span style="color:#808030; ">)</span>
plot_bar<span style="color:#808030; ">(</span>y_test<span style="color:#808030; ">,</span> loc<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'right'</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>legend<span style="color:#808030; ">(</span><span style="color:#808030; ">[</span>
    <span style="color:#0000e6; ">'train ({0} images)'</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span><span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>y_train<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> 
    <span style="color:#0000e6; ">'test ({0} images)'</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span><span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>y_test<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> 
<span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
</pre>

<img align="center" src="images/train-test-images-distributions.png" width="1000" >

#### 4.2.4. Visualize some of the training and test images and their associated targets:

* First implement a visualization functionality to visualize the number of randomly selected images:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">"""</span>
<span style="color:#696969; "># A utility function to visualize multiple images:</span>
<span style="color:#696969; ">"""</span>
<span style="color:#800000; font-weight:bold; ">def</span> visualize_images_and_labels<span style="color:#808030; ">(</span>num_visualized_images <span style="color:#808030; ">=</span> <span style="color:#008c00; ">25</span><span style="color:#808030; ">,</span> dataset_flag <span style="color:#808030; ">=</span> <span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
  <span style="color:#696969; ">"""To visualize images.</span>
<span style="color:#696969; "></span>
<span style="color:#696969; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Keyword arguments:</span>
<span style="color:#696969; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- num_visualized_images -- the number of visualized images (deafult 25)</span>
<span style="color:#696969; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- dataset_flag -- 1: training dataset, 2: test dataset</span>
<span style="color:#696969; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Return:</span>
<span style="color:#696969; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- None</span>
<span style="color:#696969; ">&nbsp;&nbsp;"""</span>
  <span style="color:#696969; ">#--------------------------------------------</span>
  <span style="color:#696969; "># the suplot grid shape:</span>
  <span style="color:#696969; ">#--------------------------------------------</span>
  num_rows <span style="color:#808030; ">=</span> <span style="color:#008c00; ">5</span>
  <span style="color:#696969; "># the number of columns</span>
  num_cols <span style="color:#808030; ">=</span> num_visualized_images <span style="color:#44aadd; ">//</span> num_rows
  <span style="color:#696969; "># setup the subplots axes</span>
  fig<span style="color:#808030; ">,</span> axes <span style="color:#808030; ">=</span> plt<span style="color:#808030; ">.</span>subplots<span style="color:#808030; ">(</span>nrows<span style="color:#808030; ">=</span>num_rows<span style="color:#808030; ">,</span> ncols<span style="color:#808030; ">=</span>num_cols<span style="color:#808030; ">,</span> figsize<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">8</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">10</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
  <span style="color:#696969; "># set a seed random number generator for reproducible results</span>
  seed<span style="color:#808030; ">(</span>random_state_seed<span style="color:#808030; ">)</span>
  <span style="color:#696969; "># iterate over the sub-plots</span>
  <span style="color:#800000; font-weight:bold; ">for</span> row <span style="color:#800000; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#808030; ">(</span>num_rows<span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
      <span style="color:#800000; font-weight:bold; ">for</span> col <span style="color:#800000; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#808030; ">(</span>num_cols<span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
        <span style="color:#696969; "># get the next figure axis</span>
        ax <span style="color:#808030; ">=</span> axes<span style="color:#808030; ">[</span>row<span style="color:#808030; ">,</span> col<span style="color:#808030; ">]</span><span style="color:#808030; ">;</span>
        <span style="color:#696969; "># turn-off subplot axis</span>
        ax<span style="color:#808030; ">.</span>set_axis_off<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
        <span style="color:#696969; ">#--------------------------------------------</span>
        <span style="color:#696969; "># if the dataset_flag = 1: Training data set</span>
        <span style="color:#696969; ">#--------------------------------------------</span>
        <span style="color:#800000; font-weight:bold; ">if</span> <span style="color:#808030; ">(</span> dataset_flag <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">1</span> <span style="color:#808030; ">)</span><span style="color:#808030; ">:</span> 
          <span style="color:#696969; "># generate a random image counter</span>
          counter <span style="color:#808030; ">=</span> randint<span style="color:#808030; ">(</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">,</span>num_train_images<span style="color:#808030; ">)</span>
          <span style="color:#696969; "># get the training image</span>
          image <span style="color:#808030; ">=</span> x_train<span style="color:#808030; ">[</span>counter<span style="color:#808030; ">,</span><span style="color:#808030; ">:</span><span style="color:#808030; ">]</span>
          <span style="color:#696969; "># get the target associated with the image</span>
          label <span style="color:#808030; ">=</span> y_train<span style="color:#808030; ">[</span>counter<span style="color:#808030; ">]</span>
        <span style="color:#696969; ">#--------------------------------------------</span>
        <span style="color:#696969; "># dataset_flag = 2: Test data set</span>
        <span style="color:#696969; ">#--------------------------------------------</span>
        <span style="color:#800000; font-weight:bold; ">else</span><span style="color:#808030; ">:</span> 
          <span style="color:#696969; "># generate a random image counter</span>
          counter <span style="color:#808030; ">=</span> randint<span style="color:#808030; ">(</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">,</span>num_test_images<span style="color:#808030; ">)</span>
          <span style="color:#696969; "># get the test image</span>
          image <span style="color:#808030; ">=</span> x_test<span style="color:#808030; ">[</span>counter<span style="color:#808030; ">,</span><span style="color:#808030; ">:</span><span style="color:#808030; ">]</span>
          <span style="color:#696969; "># get the target associated with the image</span>
          label <span style="color:#808030; ">=</span> y_test<span style="color:#808030; ">[</span>counter<span style="color:#808030; ">]</span>
        <span style="color:#696969; ">#--------------------------------------------</span>
        <span style="color:#696969; "># display the image</span>
        <span style="color:#696969; ">#--------------------------------------------</span>
        ax<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>image<span style="color:#808030; ">,</span> cmap<span style="color:#808030; ">=</span>plt<span style="color:#808030; ">.</span>cm<span style="color:#808030; ">.</span>gray_r<span style="color:#808030; ">,</span> interpolation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'nearest'</span><span style="color:#808030; ">)</span>
        <span style="color:#696969; "># set the title showing the image label</span>
        ax<span style="color:#808030; ">.</span>set_title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'y ='</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#808030; ">(</span>label<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> size <span style="color:#808030; ">=</span> <span style="color:#008c00; ">8</span><span style="color:#808030; ">)</span>
</pre>

##### 4.2.4.1. Visualize some of the training images and their associated targets:

<img align="center" src="images/25-train-images.png" width="1000" >

### 4.2.5) Normalize the training and test images to the interval: [0, 1]:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Normalize the training images</span>
x_train <span style="color:#808030; ">=</span> x_train <span style="color:#44aadd; ">/</span> <span style="color:#008000; ">255.0</span>
<span style="color:#696969; "># Normalize the test images</span>
x_test <span style="color:#808030; ">=</span> x_test <span style="color:#44aadd; ">/</span> <span style="color:#008000; ">255.0</span>
</pre>


### 4.3. Part 3: Develop the ANN model architecture

Develop the structure of the ANN model to classify the MINIST images:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Build the model</span>
model <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>models<span style="color:#808030; ">.</span>Sequential<span style="color:#808030; ">(</span><span style="color:#808030; ">[</span>
  <span style="color:#696969; ">#-----------------------------------------------------------------------------</span>
  <span style="color:#696969; "># Input layer: </span>
  <span style="color:#696969; ">#-----------------------------------------------------------------------------</span>
  <span style="color:#696969; "># - Input images size: (28,28) grayscale images</span>
  tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Flatten<span style="color:#808030; ">(</span>input_shape<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">28</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> 
  <span style="color:#696969; ">#-----------------------------------------------------------------------------</span>
  <span style="color:#696969; "># Layer # 1: </span>
  <span style="color:#696969; ">#-----------------------------------------------------------------------------</span>
  <span style="color:#696969; "># - 128 neurons</span>
  <span style="color:#696969; "># -  relu activation function </span>
  <span style="color:#696969; ">#-----------------------------------------------------------------------------</span>
  tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Dense<span style="color:#808030; ">(</span><span style="color:#008c00; ">128</span><span style="color:#808030; ">,</span> activation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'relu'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> 
  <span style="color:#696969; ">#-----------------------------------------------------------------------------</span>
  <span style="color:#696969; "># Layer # 2: </span>
  <span style="color:#696969; ">#-----------------------------------------------------------------------------</span>
  <span style="color:#696969; "># Dropout layer wit: p = 0.2</span>
  tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Dropout<span style="color:#808030; ">(</span><span style="color:#008000; ">0.2</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> 
  <span style="color:#696969; ">#-----------------------------------------------------------------------------</span>
  <span style="color:#696969; "># Output layer: </span>
  <span style="color:#696969; ">#-----------------------------------------------------------------------------</span>
  <span style="color:#696969; ">#  - 10 classes (0 - 9)</span>
  <span style="color:#696969; ">#  - activation function for multi-class classification: softwax</span>
  <span style="color:#696969; ">#-----------------------------------------------------------------------------</span>
  tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers<span style="color:#808030; ">.</span>Dense<span style="color:#808030; ">(</span><span style="color:#008c00; ">10</span><span style="color:#808030; ">,</span> activation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'softmax'</span><span style="color:#808030; ">)</span> 
<span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
</pre>


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Print the model summary</span>
model<span style="color:#808030; ">.</span>summary<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>

Model<span style="color:#808030; ">:</span> <span style="color:#0000e6; ">"sequential"</span>
_________________________________________________________________
Layer <span style="color:#808030; ">(</span><span style="color:#400000; ">type</span><span style="color:#808030; ">)</span>                 Output Shape              Param <span style="color:#696969; ">#   </span>
<span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span>
flatten <span style="color:#808030; ">(</span>Flatten<span style="color:#808030; ">)</span>            <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">784</span><span style="color:#808030; ">)</span>               <span style="color:#008c00; ">0</span>         
_________________________________________________________________
dense <span style="color:#808030; ">(</span>Dense<span style="color:#808030; ">)</span>                <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">128</span><span style="color:#808030; ">)</span>               <span style="color:#008c00; ">100480</span>    
_________________________________________________________________
dropout <span style="color:#808030; ">(</span>Dropout<span style="color:#808030; ">)</span>            <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">128</span><span style="color:#808030; ">)</span>               <span style="color:#008c00; ">0</span>         
_________________________________________________________________
dense_1 <span style="color:#808030; ">(</span>Dense<span style="color:#808030; ">)</span>              <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">10</span><span style="color:#808030; ">)</span>                <span style="color:#008c00; ">1290</span>      
<span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span>
Total params<span style="color:#808030; ">:</span> <span style="color:#008c00; ">101</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">770</span>
Trainable params<span style="color:#808030; ">:</span> <span style="color:#008c00; ">101</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">770</span>
Non<span style="color:#44aadd; ">-</span>trainable params<span style="color:#808030; ">:</span> <span style="color:#008c00; ">0</span>
</pre>

### 4.4. Part 4: Compile the ANN model

* Compile the ANN model, developed above:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Compile the ANN model</span>
model<span style="color:#808030; ">.</span><span style="color:#400000; ">compile</span><span style="color:#808030; ">(</span>optimizer<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'adam'</span><span style="color:#808030; ">,</span> <span style="color:#696969; "># the optimizer: Gradient descent version (adam vs. SGD, etc.)</span>
              loss<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'sparse_categorical_crossentropy'</span><span style="color:#808030; ">,</span> <span style="color:#696969; "># use for multi-class classification models</span>
              metrics<span style="color:#808030; ">=</span><span style="color:#808030; ">[</span><span style="color:#0000e6; ">'accuracy'</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span> <span style="color:#696969; "># performance evaluation metric</span>
</pre>


### 4.5. Part 5: Train/Fit the model

* Start training the compiled ANN model:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Train the model</span>
<span style="color:#696969; "># - Train for 100 epochs:</span>
r <span style="color:#808030; ">=</span> model<span style="color:#808030; ">.</span>fit<span style="color:#808030; ">(</span>x_train<span style="color:#808030; ">,</span> y_train<span style="color:#808030; ">,</span> validation_data<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span>x_test<span style="color:#808030; ">,</span> y_test<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> epochs<span style="color:#808030; ">=</span><span style="color:#008c00; ">100</span><span style="color:#808030; ">)</span>
</pre>


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;">Epoch <span style="color:#008c00; ">1</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">1875</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1875</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">4</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">2</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">1.7302</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.5456</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.6326</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8480</span>
Epoch <span style="color:#008c00; ">2</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">1875</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1875</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">3</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">2</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.6060</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8383</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.4226</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8902</span>
Epoch <span style="color:#008c00; ">3</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">1875</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1875</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">3</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">2</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.4503</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8726</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.3552</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9023</span>
Epoch <span style="color:#008c00; ">4</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">1875</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1875</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">3</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">2</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.3936</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8865</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.3196</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9105</span>
Epoch <span style="color:#008c00; ">5</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">1875</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1875</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">3</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">2</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.3599</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8956</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.2979</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9142</span>
<span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span>
<span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span>
Epoch <span style="color:#008c00; ">95</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">1875</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1875</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">3</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">2</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0634</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9796</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0772</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9757</span>
Epoch <span style="color:#008c00; ">96</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">1875</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1875</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">3</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">2</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0675</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9798</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0782</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9753</span>
Epoch <span style="color:#008c00; ">97</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">1875</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1875</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">3</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">2</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0657</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9794</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0788</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9748</span>
Epoch <span style="color:#008c00; ">98</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">1875</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1875</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">3</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">2</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0642</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9803</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0776</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9759</span>
Epoch <span style="color:#008c00; ">99</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">1875</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1875</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">4</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">2</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0657</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9791</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0768</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9759</span>
Epoch <span style="color:#008c00; ">100</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">1875</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1875</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">3</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">2</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0650</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9792</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0772</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9755</span>
</pre>


### 4.6. Part 6: Evaluate the model

* Evaluate the trained ANN model on the test data using different evaluation metrics:
  * Loss function
  * Accuracy
  * Confusion matrix.

#### 4.6.1. Loss function:

* Display the variations of the training and validation loss function with the number of epochs:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Plot loss per iteration</span>
<span style="color:#800000; font-weight:bold; ">import</span> matplotlib<span style="color:#808030; ">.</span>pyplot <span style="color:#800000; font-weight:bold; ">as</span> plt
plt<span style="color:#808030; ">.</span>plot<span style="color:#808030; ">(</span>r<span style="color:#808030; ">.</span>history<span style="color:#808030; ">[</span><span style="color:#0000e6; ">'loss'</span><span style="color:#808030; ">]</span><span style="color:#808030; ">,</span> label<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'loss'</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>plot<span style="color:#808030; ">(</span>r<span style="color:#808030; ">.</span>history<span style="color:#808030; ">[</span><span style="color:#0000e6; ">'val_loss'</span><span style="color:#808030; ">]</span><span style="color:#808030; ">,</span> label<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'val_loss'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
plt<span style="color:#808030; ">.</span>legend<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
plt<span style="color:#808030; ">.</span>xlabel<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Epoch Iteration'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
plt<span style="color:#808030; ">.</span>ylabel<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Loss'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
</pre>

<img align="center" src="images/loss-function.png" width="1000" >

#### 4.6.2. Accuracy:

* Display the variations of the training and validation accuracy with the number of epochs:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Plot accuracy per iteration</span>
plt<span style="color:#808030; ">.</span>plot<span style="color:#808030; ">(</span>r<span style="color:#808030; ">.</span>history<span style="color:#808030; ">[</span><span style="color:#0000e6; ">'accuracy'</span><span style="color:#808030; ">]</span><span style="color:#808030; ">,</span> label<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'acc'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
plt<span style="color:#808030; ">.</span>plot<span style="color:#808030; ">(</span>r<span style="color:#808030; ">.</span>history<span style="color:#808030; ">[</span><span style="color:#0000e6; ">'val_accuracy'</span><span style="color:#808030; ">]</span><span style="color:#808030; ">,</span> label<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'val_acc'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
plt<span style="color:#808030; ">.</span>legend<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
plt<span style="color:#808030; ">.</span>xlabel<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Epoch Iteration'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
plt<span style="color:#808030; ">.</span>ylabel<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Accuracy'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
</pre>


<img align="center" src="images/accuracy.png" width="1000" >


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Compute the model accuracy on the test data</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span>model<span style="color:#808030; ">.</span>evaluate<span style="color:#808030; ">(</span>x_test<span style="color:#808030; ">,</span> y_test<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>

<span style="color:#008c00; ">313</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">313</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">0</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">1</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0772</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9755</span>
<span style="color:#808030; ">[</span><span style="color:#008000; ">0.07720769196748734</span><span style="color:#808030; ">,</span> <span style="color:#008000; ">0.9754999876022339</span><span style="color:#808030; ">]</span>
</pre>


#### 4.6.3. Confusion Matrix Visualizations:

* Compute the confusion matrix:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Compute the confusion matrix</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#800000; font-weight:bold; ">def</span> plot_confusion_matrix<span style="color:#808030; ">(</span>cm<span style="color:#808030; ">,</span> classes<span style="color:#808030; ">,</span>
                          normalize<span style="color:#808030; ">=</span><span style="color:#074726; ">False</span><span style="color:#808030; ">,</span>
                          title<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'Confusion matrix'</span><span style="color:#808030; ">,</span>
                          cmap<span style="color:#808030; ">=</span>plt<span style="color:#808030; ">.</span>cm<span style="color:#808030; ">.</span>Blues<span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
  <span style="color:#696969; ">"""</span>
<span style="color:#696969; ">&nbsp;&nbsp;This function prints and plots the confusion matrix.</span>
<span style="color:#696969; ">&nbsp;&nbsp;Normalization can be applied by setting `normalize=True`.</span>
<span style="color:#696969; ">&nbsp;&nbsp;"""</span>
  <span style="color:#800000; font-weight:bold; ">if</span> normalize<span style="color:#808030; ">:</span>
      cm <span style="color:#808030; ">=</span> cm<span style="color:#808030; ">.</span>astype<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'float'</span><span style="color:#808030; ">)</span> <span style="color:#44aadd; ">/</span> cm<span style="color:#808030; ">.</span><span style="color:#400000; ">sum</span><span style="color:#808030; ">(</span>axis<span style="color:#808030; ">=</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span><span style="color:#808030; ">[</span><span style="color:#808030; ">:</span><span style="color:#808030; ">,</span> np<span style="color:#808030; ">.</span>newaxis<span style="color:#808030; ">]</span>
      <span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Normalized confusion matrix"</span><span style="color:#808030; ">)</span>
  <span style="color:#800000; font-weight:bold; ">else</span><span style="color:#808030; ">:</span>
      <span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Confusion matrix, without normalization'</span><span style="color:#808030; ">)</span>

  <span style="color:#696969; "># Display the confusuon matrix</span>
  <span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span>cm<span style="color:#808030; ">)</span>
  <span style="color:#696969; "># display the confusion matrix</span>
  plt<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>cm<span style="color:#808030; ">,</span> interpolation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'nearest'</span><span style="color:#808030; ">,</span> cmap<span style="color:#808030; ">=</span>cmap<span style="color:#808030; ">)</span>
  plt<span style="color:#808030; ">.</span>title<span style="color:#808030; ">(</span>title<span style="color:#808030; ">)</span>
  plt<span style="color:#808030; ">.</span>colorbar<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
  tick_marks <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>arange<span style="color:#808030; ">(</span><span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>classes<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
  plt<span style="color:#808030; ">.</span>xticks<span style="color:#808030; ">(</span>tick_marks<span style="color:#808030; ">,</span> classes<span style="color:#808030; ">,</span> rotation<span style="color:#808030; ">=</span><span style="color:#008c00; ">45</span><span style="color:#808030; ">)</span>
  plt<span style="color:#808030; ">.</span>yticks<span style="color:#808030; ">(</span>tick_marks<span style="color:#808030; ">,</span> classes<span style="color:#808030; ">)</span>
  
  fmt <span style="color:#808030; ">=</span> <span style="color:#0000e6; ">'.2f'</span> <span style="color:#800000; font-weight:bold; ">if</span> normalize <span style="color:#800000; font-weight:bold; ">else</span> <span style="color:#0000e6; ">'d'</span>
  thresh <span style="color:#808030; ">=</span> cm<span style="color:#808030; ">.</span><span style="color:#400000; ">max</span><span style="color:#808030; ">(</span><span style="color:#808030; ">)</span> <span style="color:#44aadd; ">/</span> <span style="color:#008000; ">2.</span>
  <span style="color:#800000; font-weight:bold; ">for</span> i<span style="color:#808030; ">,</span> j <span style="color:#800000; font-weight:bold; ">in</span> itertools<span style="color:#808030; ">.</span>product<span style="color:#808030; ">(</span><span style="color:#400000; ">range</span><span style="color:#808030; ">(</span>cm<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> <span style="color:#400000; ">range</span><span style="color:#808030; ">(</span>cm<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
      plt<span style="color:#808030; ">.</span>text<span style="color:#808030; ">(</span>j<span style="color:#808030; ">,</span> i<span style="color:#808030; ">,</span> format<span style="color:#808030; ">(</span>cm<span style="color:#808030; ">[</span>i<span style="color:#808030; ">,</span> j<span style="color:#808030; ">]</span><span style="color:#808030; ">,</span> fmt<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span>
               horizontalalignment<span style="color:#808030; ">=</span><span style="color:#0000e6; ">"center"</span><span style="color:#808030; ">,</span>
               color<span style="color:#808030; ">=</span><span style="color:#0000e6; ">"white"</span> <span style="color:#800000; font-weight:bold; ">if</span> cm<span style="color:#808030; ">[</span>i<span style="color:#808030; ">,</span> j<span style="color:#808030; ">]</span> <span style="color:#44aadd; ">&gt;</span> thresh <span style="color:#800000; font-weight:bold; ">else</span> <span style="color:#0000e6; ">"black"</span><span style="color:#808030; ">)</span>

  plt<span style="color:#808030; ">.</span>tight_layout<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
  plt<span style="color:#808030; ">.</span>ylabel<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'True label'</span><span style="color:#808030; ">)</span>
  plt<span style="color:#808030; ">.</span>xlabel<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Predicted label'</span><span style="color:#808030; ">)</span>
  plt<span style="color:#808030; ">.</span>show<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>

<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Predict the targets for the test data</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
p_test <span style="color:#808030; ">=</span> model<span style="color:#808030; ">.</span>predict<span style="color:#808030; ">(</span>x_test<span style="color:#808030; ">)</span><span style="color:#808030; ">.</span>argmax<span style="color:#808030; ">(</span>axis<span style="color:#808030; ">=</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># construct the confusion matrix</span>
cm <span style="color:#808030; ">=</span> confusion_matrix<span style="color:#808030; ">(</span>y_test<span style="color:#808030; ">,</span> p_test<span style="color:#808030; ">)</span>
<span style="color:#696969; "># plot the confusion matrix</span>
plot_confusion_matrix<span style="color:#808030; ">(</span>cm<span style="color:#808030; ">,</span> <span style="color:#400000; ">list</span><span style="color:#808030; ">(</span><span style="color:#400000; ">range</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">10</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> <span style="color:#074726; ">False</span><span style="color:#808030; ">,</span> <span style="color:#0000e6; ">'Confusion matrix'</span><span style="color:#808030; ">,</span> plt<span style="color:#808030; ">.</span>cm<span style="color:#808030; ">.</span>Greens<span style="color:#808030; ">)</span>
</pre>


<img align="center" src="images/Confusion-matrix.JPG" width="1000" >

### 4.6.4. Examine some of the misclassified digits:

* Display some of the misclassified digits:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># - Find the indices of all the mis-classified examples</span>
misclassified_idx <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>where<span style="color:#808030; ">(</span>p_test <span style="color:#44aadd; ">!=</span> y_test<span style="color:#808030; ">)</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span> <span style="color:#696969; "># select the index</span>
<span style="color:#696969; "># setup the subplot grid for the visualized images</span>
 <span style="color:#696969; "># the suplot grid shape</span>
num_rows <span style="color:#808030; ">=</span> <span style="color:#008c00; ">5</span>
<span style="color:#696969; "># the number of columns</span>
num_cols <span style="color:#808030; ">=</span> num_visualized_images <span style="color:#44aadd; ">//</span> num_rows
<span style="color:#696969; "># setup the subplots axes</span>
fig<span style="color:#808030; ">,</span> axes <span style="color:#808030; ">=</span> plt<span style="color:#808030; ">.</span>subplots<span style="color:#808030; ">(</span>nrows<span style="color:#808030; ">=</span>num_rows<span style="color:#808030; ">,</span> ncols<span style="color:#808030; ">=</span>num_cols<span style="color:#808030; ">,</span> figsize<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">6</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">8</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># set a seed random number generator for reproducible results</span>
seed<span style="color:#808030; ">(</span>random_state_seed<span style="color:#808030; ">)</span>
<span style="color:#696969; "># iterate over the sub-plots</span>
<span style="color:#800000; font-weight:bold; ">for</span> row <span style="color:#800000; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#808030; ">(</span>num_rows<span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
  <span style="color:#800000; font-weight:bold; ">for</span> col <span style="color:#800000; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#808030; ">(</span>num_cols<span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
    <span style="color:#696969; "># get the next figure axis</span>
    ax <span style="color:#808030; ">=</span> axes<span style="color:#808030; ">[</span>row<span style="color:#808030; ">,</span> col<span style="color:#808030; ">]</span><span style="color:#808030; ">;</span>
    <span style="color:#696969; "># turn-off subplot axis</span>
    ax<span style="color:#808030; ">.</span>set_axis_off<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># select a random mis-classified example</span>
    counter <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>random<span style="color:#808030; ">.</span>choice<span style="color:#808030; ">(</span>misclassified_idx<span style="color:#808030; ">)</span>
    <span style="color:#696969; "># get test image </span>
    image <span style="color:#808030; ">=</span> x_test<span style="color:#808030; ">[</span>counter<span style="color:#808030; ">,</span><span style="color:#808030; ">:</span><span style="color:#808030; ">]</span>
    <span style="color:#696969; "># get the true labels of the selected image</span>
    label <span style="color:#808030; ">=</span> y_test<span style="color:#808030; ">[</span>counter<span style="color:#808030; ">]</span>
    <span style="color:#696969; "># get the predicted label of the test image</span>
    yhat <span style="color:#808030; ">=</span> p_test<span style="color:#808030; ">[</span>counter<span style="color:#808030; ">]</span>
    <span style="color:#696969; "># display the image </span>
    ax<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>image<span style="color:#808030; ">,</span> cmap<span style="color:#808030; ">=</span>plt<span style="color:#808030; ">.</span>cm<span style="color:#808030; ">.</span>gray_r<span style="color:#808030; ">,</span> interpolation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'nearest'</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># display the true and predicted labels on the title of teh image</span>
    ax<span style="color:#808030; ">.</span>set_title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Y = %i, $</span><span style="color:#0f69ff; ">\h</span><span style="color:#0000e6; ">at{Y}$ = %i'</span> <span style="color:#44aadd; ">%</span> <span style="color:#808030; ">(</span><span style="color:#400000; ">int</span><span style="color:#808030; ">(</span>label<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> <span style="color:#400000; ">int</span><span style="color:#808030; ">(</span>yhat<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> size <span style="color:#808030; ">=</span> <span style="color:#008c00; ">8</span><span style="color:#808030; ">)</span>
</pre>

<img align="center" src="images/25-missclassified-images.png" width="1000" >

### 4.7. Part 7: Display a final message after successful execution completion:

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># display a final message</span>
<span style="color:#696969; "># current time</span>
now <span style="color:#808030; ">=</span> datetime<span style="color:#808030; ">.</span>datetime<span style="color:#808030; ">.</span>now<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># display a message</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Program executed successfully on: '</span><span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#808030; ">(</span>now<span style="color:#808030; ">.</span>strftime<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"%Y-%m-%d %H:%M:%S"</span><span style="color:#808030; ">)</span> <span style="color:#44aadd; ">+</span> <span style="color:#0000e6; ">"...Goodbye!</span><span style="color:#0f69ff; ">\n</span><span style="color:#0000e6; ">"</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>

Program executed successfully on<span style="color:#808030; ">:</span> <span style="color:#008c00; ">2021</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">04</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">05</span> <span style="color:#008c00; ">19</span><span style="color:#808030; ">:</span><span style="color:#008c00; ">42</span><span style="color:#808030; ">:</span><span style="color:#008000; ">07.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span>Goodbye!
</pre>


## 5. Analysis

In view of the presented results, we make the following observations:

* The simple designed ANN achieves a surprisingly high accuracy of the MNIST data classification.
* The few misclassifications appear reasonable:
* It is reasonable to confuse 9 with 4, 9, 
* It is reasonable to confuse 9 with 7,
* It is reasonable to confuse 2 with 7, etc. 


## 6. Future Work

We plan to explore the following related issues:

* To explore ways of improving the performance of this simple ANN, including fine-tuning the following hyper-parameters:
* The number of neurons
* The dropout rate
* The optimizer
* The learning rate.


## 7. References

1. Yun Lecun, Corina Cortes, Christopher J.C. Buges. The MNIST database if handwritten digits. Retrieved from: http://yann.lecun.com/exdb/mnist/.
2. Prateek Goyal. MNIST dataset using Dee Leaning algorithm (ANN). Retrieved from: https://medium.com/@prtk13061992/mnist-dataset-using-deep-learning-algorithm-ann-c6f83aa594f5.
3. Tyler Elliot Bettilyon. How to classify MNIST digits with different neural network architectures. Retrieved from: https://medium.com/tebs-lab/how-to-classify-mnist-digits-with-different-neural-network-architectures-39c75a0f03e3.
4. Mönchmeyer Anracon. A simple Python program for an ANN to cover the MNIST dataset: A starting point. Retrieved from: https://linux-blog.anracom.com/2019/09/29/a-simple-program-for-an-ann-to-cover-the-mnist-dataset-i/.
5. Orhan G. Yalcin. Image Classification in 10 Minutes with MNIST Dataset Using Convolutional Neural Networks to Classify Handwritten Digits with TensorFlow and Keras | Supervised Deep Learning. Retrieved from: https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d.












:




