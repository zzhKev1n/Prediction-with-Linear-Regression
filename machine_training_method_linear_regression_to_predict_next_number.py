# This script is for the prediction of the numbers in sequence
# Linear regression will be applied for the estimation
# This script applied the machine training method to predict the sequence

# importing the libraries to work with the dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

# input the given values and count the number of the values
# then put those given data into lists
given_values = [25, 36, 45, 54, 72, 83]
number_of_sequence = list(range(1,len(given_values)+1))

# convert the given data into arrays for the fitting process
x = np.asarray(number_of_sequence).reshape(-1, 1)
y = np.asarray(given_values).reshape(-1, 1)

# we need to predict the 7th value of the sequence
# if the prediction is further, more numbers can be added in the list
to_predict_x = [7]
to_predict_x_array = np.asarray(to_predict_x).reshape(-1, 1)

# split the dataset into 2 parts one for training and another for testing
# the test size can be 1/2 or 1/3
# test size cannot be too big, otherwise it will be lack of training
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

# fitting the linear regression to the training set
# implement the linear regression on the training model
# regressor.fit is the training process
#from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# plot the training set results
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Sequence Machine Training Model')
plt.xlabel('Number of Sequence')
plt.ylabel('Sequence')
plt.show()

# visualizing the Test set results
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Sequence Test Model')
plt.xlabel('Number of Sequence')
plt.ylabel('Sequence')
plt.show()

# Predicting the result of the 7th value
y_pred = regressor.predict(to_predict_x_array)
print(y_pred)
