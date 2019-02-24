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
  
### [GridSearchCV]((https://scikit-learn.org/stable/modules/grid_search.html))
Exhaustive search over specified parameter values for an estimator by considering all parameters combinations.

**[Hyperparameters vs Parameters](https://scikit-learn.org/stable/modules/grid_search.html)**:Hyper-parameters are parameters that are not directly learnt within estimators. In scikit-learn they are passed as arguments to the constructor of the estimator classes. Typical examples include C, kernel and gamma for Support Vector Classifier, alpha for Lasso, etc. A typical set of hyperparameters for NN include the number and size of the hidden layers, weight initialization scheme, learning rate and its decay, dropout and gradient clipping threshold, etc.<br/>
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

https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9
### Batch Size
### Number of Epochs
### Learning Rate
### Dropout Rate
### Kernel Initializers
### Activation Functions
### Number of Neurons in each hidden layer
