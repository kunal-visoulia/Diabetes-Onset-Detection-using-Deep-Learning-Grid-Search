# Diabetes-Onset-Detection-using-Deep-Learning-Grid-Search

## DATASET
The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset.All patients here are females at least 21 years old of Pima Indian heritage.Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.

## DATASET PREPROCESSING
There were initially 700+ records but some records had _glucose, bmi, skin thickness, insulin,blood pressure equal to zero which makes no sense._ So those records were removed and left were some 320 records.

Also the dataset had a problem that _some attributes on dataset had values in very different ranges; Example:no of pregnancy from 0 to 17 while insulin level from 14 to 846_ So to prevent heavy weighing of one variable over other just because of variable's size, **the data had to be [standardized](https://medium.com/@rrfd/standardize-or-normalize-examples-in-python-e3f174b65dfc)**.

## RESULTS
Using **Grid Search**, I optimized/tuned the hyperparameters(selecting the best one from specified) for our Neural Network Model:
https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/'

- [Number of Epochs](#number-of-epochs)
- [Batch Size](#batch-size)
- [Learning Rate](#learning-rate): For adam optimizer
    - [Loss Function](#the-loss-function)    
    - [Optimizer](#optimizer)
- [Dropout Rate (Regularization)](#dropout-rate)
- [Kernel Initializors (uniform/normal/zero)](#kernel-initializors)
- [Activation Functions (softmax/relu/tanh/linear)](#activation-functions)
- [Number of Neurons in each hidden layer](#number-of-neurons-in-each-hidden-layer)(in the two layers of model)

While trying to find out the best parameters through Grid Search and KFold Cross Validation, I found out that ***The accuracy on training dataset was higher than accuracy on cross validation set, implying Overfitting, So I used Regularization to overcome that by tuning the droput rate*** . 

Overall, ***the model showed an classification accuracy of 79%***
  
### [GridSearchCV]((https://scikit-learn.org/stable/modules/grid_search.html))
Exhaustive search over specified parameter values for an estimator by considering all parameters combinations.

**[Hyperparameters vs Parameters](https://scikit-learn.org/stable/modules/grid_search.html)**:Hyper-parameters are parameters that are not directly learnt within estimators. In scikit-learn they are passed as arguments to the constructor of the estimator classes. Typical examples include C, kernel and gamma for Support Vector Classifier, alpha for Lasso, etc. A typical set of hyperparameters for NN include the number and size of the hidden layers, weight initialization scheme, learning rate and its decay, dropout, Learning Rate and gradient clipping threshold, etc.<br/>
Parameters are those which would be learned by the machine like Weights and Biases.

### [Defining the Deep Learning model(in Keras)](https://keras.io/getting-started/sequential-model-guide/)
1. There are two ways to build Keras models: sequential and functional.
    - The sequential API allows you to create models layer-by-layer for most problems. It is limited in that it does not           allow you to create models that share layers or have multiple inputs or outputs.
    - The functional API allows you to create models that have a lot more flexibility as you can easily define models where         layers connect to more than just the previous and next layers. In fact, you can connect layers to (literally) any other       layer. As a result, creating complex networks such as siamese networks and residual networks become possible.

2. Specifying the input shape: The model needs to know what input shape it should expect. For this reason, the first layer in a Sequential model (and only the first, because following layers can do automatic shape inference) needs to receive information about its input shape.

3. Compilation:Before training a model, you need to **configure the learning process**, which is done via the compile method. It receives three arguments:
    - An optimizer. This could be the string identifier of an existing optimizer (such as rmsprop or adagrad), or an instance       of the Optimizer class.
    - A loss function. This is the objective that the model will try to minimize. It can be the string identifier of an             existing loss function (such as categorical_crossentropy or mse), or it can be an objective function. 
    - A list of metrics. For any classification problem you will want to set this to metrics=['accuracy']. A metric could be       the string identifier of an existing metric or a custom metric function. 
    
>***Keras models are trained on Numpy arrays of input data and labels.***

source: https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9<br/>
        https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/<br/>
We need terminologies like epochs, batch size, iterations only when the data is too big which happens all the time in machine learning and we can’t pass all the data to the computer at once. So, to overcome this problem we need to divide the data into smaller sizes and give it to our computer one by one and update the weights of the neural networks at the end of every step to fit it to the data given.
### Number of Epochs
>**One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE.**

Passing the entire dataset through a neural network is not enough. And we need to pass the **full dataset multiple times to the same neural network**. But keep in mind that we are using a limited dataset and to optimise the learning we are using Gradient Descent which is an iterative process. So, updating the weights with single pass or one epoch is not enough(and leads to underfitting).As the number of epochs increases, more number of times the weight are changed in the neural network and the curve goes from underfitting to optimal to overfitting curve.

The right number of epochs is related to how diverse your dataset is.

Since one epoch is too big to feed to the computer at once we divide it in several smaller batches.

### Batch Size
Total number of training examples present in a single batch or The batch size is a hyperparameter that defines the number of samples to work through before updating the internal model parameters.

***Batch:*** At the end of the batch, the predictions are compared to the expected output variables and an error is calculated. From this error, the update algorithm is used to improve the model, e.g. move down along the error gradient.

>- Batch Gradient Descent: Batch Size = Size of Training Set <br/> 
>- Stochastic Gradient Descent: Batch Size = 1 <br/>
>- Mini-Batch Gradient Descent: 1 < Batch Size < Size of Training Set

***Iterations:*** Iterations is the number of batches needed to complete one epoch.<br/>

>***We can divide the dataset of 2000 examples into batches of 500 then it will take 4 iterations to complete 1 epoch.<br/> or Assume you have a dataset with 200 samples (rows of data) and you choose a batch size of 5 and 1,000 epochs.This means that the dataset will be divided into 40 batches, each with five samples. The model weights will be updated after each batch of five samples.This also means that one epoch will involve 40 batches or 40 updates to the model.With 1,000 epochs, the model will be exposed to or pass through the whole dataset 1,000 times. That is a total of 40,000 batches during the entire training process.***

### Learning Rate
Using the Adam optimizer, Hypertuning the learning rate.I used binary_crossentropy as loss function. <br/>
  1. [The Loss Function](https://blog.algorithmia.com/introduction-to-loss-functions/):  it’s a method of evaluating how        well your algorithm models your dataset. As you change pieces of your algorithm to try and improve your model, your loss    function will tell you if you’re getting anywhere.
      https://medium.com/data-science-group-iitr/loss-functions-and-optimization-algorithms-demystified-bb92daff331c
      - Regressive loss functions: They are used in case of regressive problems, that is when the target variable is                 continuous. Most widely used regressive loss function is Mean Square Error. Other loss functions are:
            1. Absolute error — measures the mean absolute value of the element-wise difference between input;
            2. Smooth Absolute Error — a smooth version of Abs Criterion.
      - Classification loss functions:The output variable in classification problem is usually a probability value f(x),             called the score for the input x. Generally, the magnitude of the score represents the confidence of our prediction.         The target variable y, is a binary variable, 1 for true and -1 for false. 
        On an example (x,y), the margin is defined as yf(x). The margin is a measure of how correct we are. Most                     classification losses mainly aim to maximize the margin. Some classification algorithms are:
            1. Binary Cross Entropy 
            2. Negative Log Likelihood
            3. Margin Classifier
            4. Soft Margin Classifier

      - Embedding loss functions: It deals with problems where we have to measure whether two inputs are similar or                 dissimilar. Some examples are:
            1. L1 Hinge Error- Calculates the L1 distance between two inputs.
            2. Cosine Error- Cosine distance between two inputs.

**During the training process, we tweak and change the parameters (weights) of our model to try and minimize that loss function, and make our predictions as correct as possible. But how exactly do you do that? How do you change the parameters of your model, by how much, and when?  *This is where optimizers come in.***    
  2. [Optimizer](https://blog.algorithmia.com/introduction-to-optimizers/): They tie together the loss function and model      parameters by updating the model in response to the output of the loss function. In simpler terms, optimizers shape and      mold your model into its most accurate possible form by futzing with the weights. The loss function is the guide to the      terrain, telling the optimizer when it’s moving in the right or wrong direction.
   > think of a hiker trying to get down a mountain with a blindfold on. It’s impossible to know which direction to go in,        but there’s one thing she can know: if she’s going down (making progress) or going up (losing progress). Eventually, if      she keeps taking steps that lead her downwards, she’ll reach the base.Similarly, it’s impossible to know what your          model’s weights should be right from the start. But with some trial and error based on the loss function (whether the        hiker is descending), you can end up getting there eventually.
  
   Popular optimizers are Gradient Descent((backpropagation is basically gradient descent implemented on a network),            Stochastic Gradient Descent. Other optimizers based on gradient descent:
   1. Adagrad: Adagrad adapts the learning rate specifically to individual features, that means that some of the weights in       your dataset will have different learning rates than others.<br/>
      This works really well for sparse datasets where a lot of input examples are missing.<br/>
      Adagrad has a major issue though: the adaptive learning rate tends to get really small over time. *Some other optimizers below seek to eliminate this problem.*

   2. RMSprop: RMSprop is similar to Adaprop, which is another optimizer that seeks to solve some of the issues that Adagrad       leaves open. Instead of letting all of the gradients accumulate for momentum, it only accumulates gradients in a fixed       window. 

   3. Adam: Adam stands for adaptive moment estimation, and is another way of using past gradients to calculate current           gradients. Adam also utilizes the concept of momentum by adding fractions of previous gradients to the current one.         This optimizer has become pretty widespread, and is practically accepted for use in training neural nets.
   
### [Dropout Rate](https://medium.com/@amarbudhiraja/https-medium-com-amarbudhiraja-learning-less-to-learn-better-dropout-in-deep-machine-learning-74334da4bfc5)
Dropout is an approach to regularization in neural networks which helps reducing interdependent learning amongst the neurons
by ignoring units (i.e. neurons) during the training phase of certain set of neurons which is chosen at random. These units are not considered during a particular forward or backward pass. <br/>
This is to **prevent over-fitting**, A fully connected layer occupies most of the parameters, and hence, neurons develop co-dependency amongst each other during training which curbs the individual power of each neuron leading to over-fitting of training data.

In machine learning, Regularization reduces over-fitting by adding a penalty to the loss function. By adding this penalty, the model is trained such that it does not learn interdependent set of features weights. Regularization techniques: [L1 (Laplacian) and L2 (Gaussian) penalties](https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c).<bt/>
**Training Phase:**: For each hidden layer, for each training sample, for each iteration, ignore (zero out) a random fraction, p, of nodes (and corresponding activations).

**Testing Phase:** Use all activations, but reduce them by a factor p (to account for the missing activations during training).

>Some Observations:<br/>
>1. Dropout forces a neural network to learn more robust features that are useful in conjunction with many different random subsets of the other neurons.
>2. Dropout roughly doubles the number of iterations required to converge. However, training time for each epoch is less.
>3. With H hidden units, each of which can be dropped, we have 2^H possible models. In testing phase, the entire network is considered and each activation is reduced by a factor p.

### [Kernel Initializers](https://machinelearningmastery.com/why-initialize-a-neural-network-with-random-weights/)
Initializations define the way to set the initial random weights of Keras layers.<br/>
- Zeros: Initializer that generates tensors initialized to 0.
- Ones: Initializer that generates tensors initialized to 1.
- Constant: Initializer that generates tensors initialized to a constant value.
- RandomNormal: Initializer that generates tensors with a normal distribution.
- RandomUniform: Initializer that generates tensors with a uniform distribution.

**Why Not Set Weights to Zero?**<br/>
We can use the same set of weights each time we train the network; for example, you could use the values of 0.0 for all weights. In this case, the equations of the learning algorithm would fail to make any changes to the network weights, and the model will be stuck. It is important to note that the **bias weight in each neuron is set to zero by default, not a small random value**. Specifically, nodes that are side-by-side in a hidden layer connected to the same inputs must have different weights for the learning algorithm to update the weights. This is often referred to as the need to **break symmetry during training**.
>Perhaps the only property known with complete certainty is that the initial parameters need to “break symmetry” between different units. If two hidden units with the same activation function are connected to the same inputs, then these units must have different initial parameters. If they have the same initial parameters, then a deterministic learning algorithm applied to a deterministic cost and model will constantly update both of these units in the same way.

**When to Initialize to the Same Weights?**<br/>
We could use the same set of random numbers each time the network is trained.This would not be helpful when evaluating network configurations. It may be helpful in order to train the same final set of network weights given a training dataset in the case where a model is being used in a production environment.

**[Initializing the biases](http://cs231n.github.io/neural-networks-2/)**: It is possible and common to initialize the biases to be zero, since the asymmetry breaking is provided by the small random numbers in the weights. For ReLU non-linearities, some people like to use small constant value such as 0.01 for all biases because this ensures that all ReLU units fire in the beginning and therefore obtain and propagate some gradient.<br/>
One of the most straightforward is initialization of weight and bias values and typical advice is to randomly initialize weights (to break symmetry) and initialize biases to zero.

https://becominghuman.ai/basics-of-neural-network-bef2ba97d2cf
Biases are weights added to hidden layers. They too are randomly initialised and updated in similar manner as the hidden layer. While the role of hidden layer is to map the shape of the underlying function in the data, the role of bias is to laterally shift the learned function so it overlaps with the original function.

https://medium.com/usf-msds/deep-learning-best-practices-1-weight-initialization-14e5c0295b94
It is important to note that setting biases to 0 will not create any troubles as non zero weights take care of breaking the symmetry and even if bias is 0, the values in every neuron are still different.

### [Activation Functions](https://towardsdatascience.com/secret-sauce-behind-the-beauty-of-deep-learning-beginners-guide-to-activation-functions-a8e23a57d046)
![](https://cdn-images-1.medium.com/max/1600/1*p_hyqAtyI8pbt2kEl6siOQ.png)<br/>
 Activation functions are used to introduce non-linearity to neural networks. It squashes the values in a smaller range viz. a Sigmoid activation function squashes values between a range 0 to 1. 


- Identity or Linear Activation Function : simplest activation function of all. It applies identity operation on your data and output data is proportional to the input data. Problem with linear activation function is that it’s derivative is a constant and it’s gradient will be a constant too and the descent will be on a constant gradient.
- Sigmoid or Logistic activation function(Soft Step)
- Hyperbolic tangent (TanH):  It looks like a scaled sigmoid function. Data is centered around zero, so the derivatives will be higher. Tanh quickly converges than sigmoid and logistic activation functions
-  Linear Unit(ReLU) — It trains 6 times faster than tanh. Output value will be zero when input value is less than zero. If input is greater than or equal to zero, output is equal to the input. When the input value is positive, derivative is 1, hence there will be no squeezing effect which occurs in the case of backpropagating errors from the sigmoid function.
-  Softmax functions convert a raw value into a posterior probability. This provides a measure of certainty. It squashes the outputs of each unit to be between 0 and 1, just like a sigmoid function. But it also divides each output such that the total sum of the outputs is equal to 1. The output of the softmax function is equivalent to a categorical probability distribution, it tells you the probability that any of the classes are true
>**Conclusion**: ReLU and it’s variants should be preferred over sigmoid or tanh activation functions. As well as ReLUs are faster to train. If ReLU is causing neurons to be dead, use Leaky ReLUs or it’s other variants. Sigmoid and tanh suffers from vanishing gradient problem and should not be used in the hidden layers. ReLUs are best for hidden layers. Activation functions which are easily differentiable and easy to train should be used.

### Number of Neurons in each hidden layer
Generally the number of neurons in a layer controls the representational capacity of the network, at least at that point in the topology. Also, generally, a large enough single layer network can approximate any other neural network, at least in theory.
