import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import seaborn as sns
from matplotlib.animation import FuncAnimation

np.random.seed(42)

class classifier():
    def __init__(self):
        self.m, self.n = 0,0
        self.theta = 0
        self.eps = 1e-6
    
    def L(self, X, Y, theta = None):
        if(theta == None):
            theta = self.theta
        h_theta_x = 1/(1+np.exp(-X@theta))
        return (np.dot(Y, np.log(h_theta_x)) + np.dot((1-Y), np.log(1-h_theta_x)))

    def grad_L(self, X, Y, theta = None):
        if(theta == None):
            theta = self.theta
        h_theta_x = 1/(1+np.exp(-X@theta))
        return (X.T @ (Y-h_theta_x))

    def H_inv(self, X, Y, theta = None):
        if(theta == None):
            theta = self.theta
        h_theta_x = 1/(1+np.exp(-X@theta))
        H = (X.T * h_theta_x*(1-h_theta_x)) @ X
        if np.linalg.det(H) < self.eps:
            return None
        return np.linalg.inv(H)


    def fit(self, X,Y):
        self.m = X.shape[0]
        X = np.hstack((np.ones((self.m,1)), X))
        self.n = X.shape[1]
        self.theta = np.zeros((self.n,))
        H_inv = self.H_inv(X,Y)
        L = self.L(X,Y)
        grad = 10

        while(type(H_inv) != type(None) and np.max(np.abs(grad)) > self.eps):
            grad = self.grad_L(X,Y)
            self.theta += H_inv @ grad
            H_inv = self.H_inv(X,Y)
            print(L)
            L = self.L(X,Y)
    
        return self.theta


X = pd.read_csv("Q3/logisticX.csv", header=None)
X = (X-np.mean(X,axis = 0))/np.std(X,axis=0) # normailse
# X = (X-np.min(X,axis = 0))/(np.max(X,axis=0)-np.min(X,axis = 0)) # normailse
Y = pd.read_csv("Q3/logisticY.csv", header=None)
Y_train = np.array(Y)
Y_train = Y_train.reshape((Y_train.shape[0],))

model = classifier()
theta = model.fit(np.array(X), Y_train)
print("theta = ", theta)
def line(x, theta):
    return theta[0]/theta[2]-theta[1]/theta[2]*x

x = [np.min(X[0]), np.max(X[0])]
y = [line(p,theta) for p in x]


plt.scatter(np.array(X[0].get(X.index[Y[0] == 1])), np.array(X[1].get(X.index[Y[0] == 1])),marker='.',label = "class 1")
plt.scatter(np.array(X[0].get(X.index[Y[0] == 0])), np.array(X[1].get(X.index[Y[0] == 0])),marker='.',label = "class 0")
plt.plot(x,y, c = 'r',label = "decision boundary")

plt.legend()
plt.xlabel("x0")
plt.ylabel("x1")
plt.title("Decision Boundary with True Data Labels")
plt.show()
