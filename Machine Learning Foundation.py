import numpy as np
import matplotlib.pyplot as plt
# plt.style.use("./deeplearning.mplstyle")

#x_train is the input variable (size in 1000 square feet)
#y_train is the target price (price in 1000s of dollar )

x_train = np.array([1.0 , 2.0])
y_train = np.array([300.0, 500.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")


# m is the number of training examples
m = len(x_train)
print(f"Number of training examples is: {m}")
