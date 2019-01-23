# Diabetes-Onset-Detection-using-Deep-Learning-Grid-Search

## DATASET
The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset.All patients here are females at least 21 years old of Pima Indian heritage.Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.

## DATASET PREPROCESSING
There were initially 700+ records but some records had _glucose, bmi, skin thickness, insulin,blood pressure equal to zero which makes no sense._ So those records were removed and left was some 320 records.

Also the dataset had a problem that _some attributes on dataset had values in very different ranges; Example:no of pregnancy from 0 to 17 while insulin level from 14 to 846_ So to prevent heavy weighing of one variable over other just because of variable's size, **the data had to be standardized**.

## RESULTS
Using **Grid Search**, I optimized/tuned the hyperparameters for our Neural Network Model:

- Batch Size
- Number of Epochs
- Learning Rate
- Dropout Rate(Regularization) 
- Kernel Initializors(uniform/normal/zero)
- Activation Functions(softmax/relu/tanh/linear)
- Number of Neurons in each hidden layer

While trying to find out the best parameters through Grid Search and KFold Cross Validation, I found out that ***The accuracy on training dataset was higher than accuracy on cross validation set, implying Overfitting, So I used Regularization to overcome that by tuning the droput rate*** . 

Overall, ***the model showed an accuracy of 79%***
 
