# Diabetes-Onset-Detection-using-Deep-Learning-Grid-Search

## DATASET
The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset.All patients here are females at least 21 years old of Pima Indian heritage.Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.

## DATASET PREPROCESSING
There were initially 700+ records but some records had _glucose, bmi, skin thickness, insulin,blood pressure equal to zero which makes no sense._ So those records were removed and left were some 320 records.

Also the dataset had a problem that _some attributes on dataset had values in very different ranges; Example:no of pregnancy from 0 to 17 while insulin level from 14 to 846_ So to prevent heavy weighing of one variable over other just because of variable's size, **the data had to be [standardized](https://medium.com/@rrfd/standardize-or-normalize-examples-in-python-e3f174b65dfc)**.

## RESULTS
Using **Grid Search**, I optimized/tuned the hyperparameters(selecting the best one from specified) for our Neural Network Model:

- [Batch Size](#batch-size)
- [Number of Epochs](#number-of-epochs)
- [Learning Rate](#learning-rate)
- [Dropout Rate (Regularization)](#dropout-rate)
- [Kernel Initializors (uniform/normal/zero)](#kernel-initializors)
- [Activation Functions (softmax/relu/tanh/linear)](#activation-functions)
- [Number of Neurons in each hidden layer](#number-of-neurons-in-each-hidden-layer)(in the two layers of model)

While trying to find out the best parameters through Grid Search and KFold Cross Validation, I found out that ***The accuracy on training dataset was higher than accuracy on cross validation set, implying Overfitting, So I used Regularization to overcome that by tuning the droput rate*** . 

Overall, ***the model showed an classification accuracy of 79%***

### [DEEP LEARNING](https://towardsdatascience.com/why-deep-learning-is-needed-over-traditional-machine-learning-1b6a99177063)
#### Non-Linear Hypothesis
Consider a supervised learning classification problem where you have a training set like this. If you want to apply logistic regression to this problem, one thing you could do is apply logistic regression with a lot of nonlinear features like. <br/>
![](images/1.png)<br/>
g as usual is the sigmoid function, and we can include lots of polynomial terms like these. And, if you include enough polynomial terms then, maybe you can get a hypotheses that separates the positive and negative examples. This particular method works well when you have only, say, two features - x1 and x2 - because you can then include all those polynomial terms of x1 and x2. But for many interesting machine learning problems, we would have a lot more features than just two.

Example say we have 100 features, x1-x100,, and, **if you were to include all the quadratic terms**, that is the second order polynomial terms, there would be a lot of them. There would be terms like x<sub>1</sub><sup>2</sup>,x<sub>1</sub>x<sub>2</sub>,x<sub>1</sub>x<sub>3</sub> upto x<sub>1</sub>x<sub>100</sub> and then you have x<sub>1</sub><sup>2</sup>, x<sub>2</sub>x<sub>3</sub>, x<sub>2</sub>x<sub>4</sub> upto x<sub>2</sub>x<sub>100</sub> and so on.<br/>
*This way for n = 100, you will end up with 5000 features.(O(n<sub>2</sub>) roguhly n<sub>2</sub>/2). So including all the quadratic features doesn't seem like a good idea, because that is a lot of features and you might end up overfitting the training set, and it can also be computationally expensive*
- One thing you could do is include only a subset of these, so if you include only the features x<sub>1</sub><sup>2</sup>, x<sub>2</sub><sup>2</sup>, x<sub>3</sub><sup>2</sup> up to maybe x<sub>100</sub><sup>2</sup>, then the number of features is much smaller, but this is not enough features and certainly won't let you fit the complex data set like above.

**For many machine learning problems, n will be pretty large. Here's an example:**
#### The Problem of Computer Vision
Suppose you want to use machine learning to train a classifier to examine an image and tell whether or not the image is a car. To understand why computer vision is hard let's zoom into a small part of the image like that area where the little red rectangle is. It turns out that where you and I see a car, the computer sees is this matrix, of pixel intensity values that tells us the brightness of each pixel in the image.** So the computer vision problem is to look at this matrix of pixel intensity values, and tell us that these numbers represent the door handle of a car.** <br/>
![](images/2.png)<br/>

Keepinf the pixel1 and pixel2 positions same for the dataset, we'll find is that the cars and non-cars end up lying in different regions of the space, and we need non-linear hypotheses to try to separate out the two classes.<br/>
![](images/3.png)<br/>

If each images is 50x50 pixels, **dimension of feature space,N = 2500(for grayscale; 7500 for 3 channel RGB)**<br/>
So, feaure vector is the list of the pixel intensities/brightness( values between say 0 to 255) for 2500 pixels. ***So if we try to learn a non-linear hypothesis including all the quadratic features, Total Number of Features = 3 million***

#### NEURAL NETWORKS
The brain does so many different and amazing things. It seems like if you want to mimic the brain you have to write lots of different pieces of software to mimic all of these different fascinating, amazing things(seeing, hearing, taste,do math, speak), but there is this fascinating hypothesis that the way the brain does all of these different things is not with like a thousand different programs, but instead, with just a single learning algorithm. This is just a hypothesis but let me share with you some of the evidence for this.<br/>
>Neuroscientists have done the following fascinating experiments where you cut the wire from the ears to the auditory cortex and you re-wire,in this case an animal's brain, so that the signal from the eyes to the optic nerve eventually gets routed to the auditory cortex. If you do this it turns out, the auditory cortex will learn to see. 

##### MODEL REPRESENTATION
In the Brain, at a simplistic level, a neuron is a computational unit that gets a number of inputs through its input wires(deondrites) and does some computation and then it sends outputs via its axon to other nodes(other neurons in the brain.)

![](images/4.png)<br/>
- x = [ x1, x2, x3....]  are inputs and theta = [theta1, theta2, theta3,...]  are the weights/parameters. 
- Showing x0(the bias unit/the (+1) unit)depends on notation conveniency.
- Activation function in the neural network terminology is just another term for function for the required non-linearity, g
(z)(sigmoid activation function,etc).<BR/>
**FOR OVERSIMPLIFICATION AND UNSERSTANDING PURPOSES: ASSUME g(z) IS THE LOGISTIC/SIGMOID FUNCTION.**

So here are the **computations** represented by the above diagram:<br/>
![](images/5.png)<br/>

##### FORWARD PROPAGATION: VECTORIZED IMPLEMENTATION
Calculating the value of hypothesis (parameterized by theta), by going from input layer => Hidden Layer(s)(the activation units in them) => output layer, is called Forward Prop.

![](images/6.png)<br/>
**The above image is for my notes purposes**

![](images/7.png)<br/>
If you look at the last two layers only, The hypothesis will look a lot like logistic regression where instead of featues x1,x2,x3... we have these features, a1, a2, a3(see image for correct notation) and have capital theta<sup>(2)</sup> instead of lowercase theta. **The features fed into logistic regression(a1,a2..) are these values computed by the hidden layer. *and they themselves are learned as functions of the input.Concretely, the function mapping from layer 1 to layer 2, is determined by some other set of parameters, theta<sup>1</sup>.***<b/>
![](images/8.png)<br/>

>Depending on what parameters is choosen for theta<sup>1</sup>, complex features can be learned and therefore you can end up with a better hypotheses than if you were constrained to use the raw features x1, x2 or x3 or if you will constrained to choose the polynomial terms. **But instead, this algorithm has the flexibility to try to learn whatever features at once, using these a1, a2, a3 in order to feed into this last unit that's essentially a logistic regression here.**

In ANN architecture with multiple layers, in each layer we can learn even more complex features from the input features recieved from previous layer.

### EXAMPLES
Consider the following problem where we have features x1 and x2 that are binary values:<br/>
In this example, only two positive examples and two negative examples have been drawn. Think of this as a simplified version of a more complex learning problem where we may have a bunch of positive examples in the upper right and lower left and a bunch of negative examples denoted by the circles. **And what we'd like to do is learn a non-linear division of boundary that may need to separate the positive and negative examples.**

![](images/12.png)<br/>
**Concretely, label class y = x1 XOR x2; It turns out that these specific examples work out a little bit better if we use x1 XNOR x2 
Thus, y = 1 when both x1,x2 are either 0(false) or 1(true) and y = 0 if any one of them is 1(true)**

By adjusting the weights, and x(input feature vector) values:

![](images/9.png)<br/>
![](images/10.png)<br/>
![](images/11.png)<br/>

**Finally,**<br/>
![](images/13.png)<br/>
>And this is a sort of intuition about why neural networks can compute pretty complicated functions.<br/> That when you have multiple layers you have relatively simple function of the inputs of the second layer. But the third layer I can build on that to complete even more complex functions, and then the layer after that can compute even more complex functions.

### NEURAL NETWORKS for MULTI-CLASSIFICATION PROBLEM
For Example:Hand-Written Digit Recognition, 

The way we do multiclass classification in a neural network is essentially an extension of the one versus all method.

**Let's say that we have a computer vision example, where we're trying to recognize, four categories of objects: Given an image, decide if it is a pedestrian, a car, a motorcycle or a truck.**<br/>

![](images/14.png)<br/>
So, the output now is actually needs to be a vector of four numbers and what we try to do is:
- get the 1st output unit to classify: is the image a pedestrian, yes or no.
- get the 2mrd unit to classify: is the image a car, yes or no.
- get the 3rd unit to classify: is the image a motorcycle, yes or no, and
- get the 4th unit to classify: is the image a truck, yes or no.
And thus, when the image is of a pedestrian, we would ideally want the network to output [1, 0, 0, 0, ] when it is a car we want it to output [ 0, 1, 0, 0,] when this is a motorcycle, we want it to output[ 0, 0, 1, 0] and so on.

**So this is just like the "one versus all" method that I described in my earlier project on [logistic regression](https://github.com/kunal-visoulia/Classification-Using-Logistic-Regression), and here we have essentially four logistic regression classifiers, each of which is trying to recognize one of the four classes that we want to distinguish amongst**

![](images/15.png)<br/>


### GRID SEARCH CV
### KFOLD (UPDATE PREV REPO)

### Batch Size
### Number of Epochs
### Learning Rate
### Dropout Rate
### Kernel Initializers
### Activation Functions
### Number of Neurons in each hidden layer
