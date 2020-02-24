# implementation of sigmoid function and perceptron model
import numpy as np
import matplotlib.pyplot as plt


def draw(x1, x2):
    temp = plt.plot(x1, x2)
    plt.pause(0.01)
    temp[0].remove()


def sigmoid(score):
    return 1 / (1 + np.exp(-score))


# cross entropy
def error(line_parameters, points, y):
    m = points.shape[0]
    p = sigmoid(points * line_parameters)
    cross_entropy = (1 / m) * (np.log(p).T * y + np.log(1 - p).T * (1 - y))
    return cross_entropy


def gradient_descent(line_parameters, points, y, alpha):
    m = points.shape[0]
    for i in range(500):
        p = sigmoid(points * line_parameters)
        gradient = (points.T * (p - y)) * (alpha / m)
        line_parameters = line_parameters - gradient
        w1 = line_parameters.item(0)
        w2 = line_parameters.item(1)
        b = line_parameters.item(2)
        x1 = np.array([points[:, 0].min(), points[:, 0].max()])
        x2 = -b / w2 + x1 * (-w1 / w2)
        draw(x1, x2)


# creating our own dataset with random data points
# %matplotlib inline

n_pts = 100
bias = np.ones(n_pts)
np.random.seed(0)  # this is important as it keeps the random values same each time
top_reigon = np.array([np.random.normal(10, 2, n_pts), np.random.normal(12, 2, n_pts), bias]).T
bottom_region = np.array([np.random.normal(5, 2, n_pts), np.random.normal(6, 2, n_pts), bias]).T
all_points = np.vstack((top_reigon, bottom_region))
# dataset prepared.

# creating an intial linear separation
"""
Hardcoded Part, SKIP-
w1=-0.2
w2=-0.35
b=3.5
"""
line_parameters = np.matrix([np.zeros(3)]).T
# x1=np.array([bottom_region[:, 0].min(),top_reigon[:, 0].max()])
# from w1*x1+w2*x2+b=0
# x2=-b/w2+x1*(-w1/w2)
linear_combination = all_points * line_parameters
# print(linear_combination)
y = np.array([np.zeros(n_pts), np.ones(n_pts)]).reshape(n_pts * 2, 1)
print(y)

plt.scatter(top_reigon[:, 0], top_reigon[:, 1], color='r')
plt.scatter(bottom_region[:, 0], bottom_region[:, 1], color='b')
alpha = float(input("Enter Number Of Iterations:"))
gradient_descent(line_parameters, all_points, y, alpha)
plt.show()
print(all_points)

print(error(line_parameters, all_points, y))
# gives the trueness of our perceptron model
