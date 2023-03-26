# Week 1

# This code is to create "Model Representation"
# On how to learn to implement the model  𝑓𝑤,𝑏 for linear regression with one variable.

# The tools we use is  NumPy and Matplotlib.

# As in the lecture, we will use the motivating example of housing price prediction.
# This lab will use a simple data set with only two data points - a house with 1000 square feet(sqft) sold for $300,000 and a house with 2000 square feet sold for $500,000. These two points will constitute our data or training set. In this lab, the units of size are 1000 sqft and the units of price are 1000s of dollars.


# 1. Size (1000 sqft): 1.0 
# Price (1000s of dollars): 300

# 2. Size (1000 sqft): 2.0 
# Price (1000s of dollars): 500


import numpy as np
import matplotlib.pyplot as plt
# plt.style.use("./deeplearning.mplstyle")

#x_train is the input variable (size in 1000 square feet)
#y_train is the target price (price in 1000s of dollar )

x_train = np.array([1.0 , 2.0])
y_train = np.array([300.0, 500.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

#M = as the number of training example. NumPy array have .shape parameter. x_train.shape return a python tuple with an entry for each dimension. 
#this code helps you find out and display the size of the x_train table and the number of examples or items it contains.
#x_train.shape is a command that tells us the size of this table, showing how many rows and columns it has.

print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print("number of training example is : {m}")


# m is the number of training examples
m = len(x_train)
print(f"Number of training examples is: {m}")


# To access a value in a Numpy array, one indexes the array with the desired offset. For example the syntax to access location zero of x_train is x_train[0]. Run the next code block below to get the  𝑖𝑡ℎ
#  training example.
# i = 0 # Change this to 1 to see (x^1, y^1)

x_i = x_train[1]
y_i = y_train[1]
print(f"(x^({1}), y^({1})) = ({x_i}, {y_i})")

# Plotting the data
# You can plot these two points using the scatter() function in the matplotlib library, as shown in the cell below.

# The function arguments marker and c show the points as red crosses (the default is blue dots).
# You can use other functions in the matplotlib library to set the title and labels to display

#plot data points
# plt.scatter(x_train, y_train, marker="x", c="r")
# #plot the title
# plt.title("Housing Price Market")
# #set y_axis and x_axis label 
# plt.ylabel("price(in 1000 of dollars)")
# plt.xlabel("size(1000 sqft)")

# plt.show()

# As described in lecture, the model function for linear regression (which is a function that maps from x to y) is represented as

# 𝑓𝑤,𝑏(𝑥(𝑖))=𝑤𝑥(𝑖)+𝑏(1)
# The formula above is how you can represent straight lines - different values of  𝑤
#   and  𝑏
#   give you different straight lines on the plot.

w = 200
b = 100
print(f"w: {w}")
print(f"b: {b}")

# Now, let's compute the value of  𝑓𝑤,𝑏(𝑥(𝑖))
#   for your two data points. You can explicitly write this out for each data point as -

# for  𝑥(0)
#  , f_wb = w * x[0] + b

# for  𝑥(1)
#  , f_wb = w * x[1] + b

# For a large number of data points, this can get unwieldy and repetitive. So instead, you can calculate the function output in a for loop as shown in the compute_model_output function below.

# Note: The argument description (ndarray (m,)) describes a Numpy n-dimensional array of shape (m,). (scalar) describes an argument without dimensions, just a magnitude.
# Note: np.zero(n) will return a one-dimensional numpy array with  𝑛
#   entries

def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      y (ndarray (m,)): target values
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb

tmp_f_wb = compute_model_output(x_train, w, b,)

# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()

#predictions
#now we have a model, we could use predictions using these : 

w = 200
b = 100
x_i = 1.2
cost_1200sqft = w*x_i+ b
print(f"${cost_1200sqft:.0f} thousand dollars")


# w = 200
# b = 100
# print(f"w: {w}")
# print(f"b: {b}")

# tmp_f_wb = compute_model_output(x_train, w, b)
# plt.plot(x_train, tmp_f_wb, c="b", label="our prediction")
# plt.scatter(x_train, y_train, marker="x", c="r", label="Actual values")
# plt.title("Housing prices")
# plt.ylabel("price("in 1000 of dollars)")
# plt.xlabel("size(1000 sqft)")
# plt.show()
