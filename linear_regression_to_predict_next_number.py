# This script is for the prediction of the numbers in sequence
# Linear regression will be applied for the estimation

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

# fine the best fit line for the given data using the sklearn extension
regressor = LinearRegression()
regressor.fit(x, y)

# use the linear regression extension to predict the 7th value on the best fit
# output some more information about the best fit line (slope, y axis intercept)
predicted_y_value = regressor.predict(to_predict_x_array)
slope = regressor.coef_
c = regressor.intercept_
print("Predicted y:\n",predicted_y_value)
print("slope: ",slope)
print("y-intercept (c): ",c)

# visualise the prediction
# plot the given data and the best fit line on a graph
# plot the given data as scatter on the graph
# draw the best fit line with the slope and intercept information above
plt.scatter(x, y, color="blue")
best_fit_line=[ slope*i+c for i in np.append(x, to_predict_x)]
best_fit_line=np.array(best_fit_line).reshape(-1,1)
plt.plot(np.append(x, to_predict_x), best_fit_line, color="red")
plt.title('Predict the next numbers in a given sequence')  
plt.xlabel('x')  
plt.ylabel('Numbers') 
plt.show()