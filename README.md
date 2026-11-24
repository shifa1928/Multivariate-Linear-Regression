# Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
### Step1
<br>
Load the Boston housing dataset and define the feature matrix (X) and response vector (y).

### Step2
<br>
Split the dataset into training and testing sets (60% train, 40% test).

### Step3
<br>
Create a linear regression object and train the model using the training set.

### Step4
<br>
4. Print the regression coefficients and the variance score (R²) on the test set.


### Step5
<br>
Plot the residual errors for the training and test data, then display the plot.

## Program:
```

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics

# load the boston dataset
diabetes = datasets.load_diabetes(return_X_y=False)

# defining feature matrix(X) and response vector(y)
X = diabetes.data
y = diabetes.target

# splitting X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                     random_state=1)

# create linear regression object
reg = linear_model.LinearRegression()

# train the model using the training sets
reg.fit(X_train, y_train)

# regression coefficients
print('Coefficients: ', reg.coef_)

# variance score: 1 means perfect prediction
print('Variance score: {}'.format(reg.score(X_test, y_test)))

# plot for residual error

## setting style
plt.style.use('fivethirtyeight')

## plotting residual errors in training data
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train,
            color = "green", s = 10, label = 'Train data')

## plotting residual errors in test data
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test,
            color = "blue", s = 10, label = 'Test data')

## plotting line for zero residual error
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)

## plotting legend
plt.legend(loc = 'upper right')

## plot title
plt.title("Residual errors")

## method call for showing the plot
plt.show()




```
## Output:

<img width="671" height="488" alt="image" src="https://github.com/user-attachments/assets/b50d9827-ef48-44e8-bd36-49f12bbfb50f" />



<br>

## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
