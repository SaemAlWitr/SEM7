import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import seaborn as sns
from matplotlib.animation import FuncAnimation

df_type = type(pd.DataFrame())

class gda:
    def __init__(self):
        self.m, self.n = 0,0
        self.m0, self.m1 = 0, 0
        self.covar = 0
        self.covar_0 = 0
        self.covar_1 = 0
        self.mu_0 = 0
        self.mu_1 = 0
    def eval_normal(self, x, covar, mu): # evaluates the multinomial normal N(mu, covar) at x
        det = np.linalg.det(covar)
        p = 1/((2*np.pi)**(self.n/2)*np.sqrt(det))*np.exp(-0.5* (x-mu) @ np.linalg.solve(covar, (x-mu)))
        return p

    def pxgivy(self, x, y): # p(x|y = y)
        if y == 1:
            return self.eval_normal(x, self.covar_1, self.mu_1)
        else:
            return self.eval_normal(x, self.covar_0, self.mu_0)
        
    def fit(self, X, Y, same_covar = False):
        self.m = X.shape[0]
        self.n = X.shape[1]

        idx_0 = [i for i in range(self.m) if Y[i] == 0]
        idx_1 = [i for i in range(self.m) if Y[i] == 1]

        self.m0, self.m1 = len(idx_0), len(idx_1)

        self.mu_0 = np.mean(X[idx_0], axis = 0)
        self.mu_1 = np.mean(X[idx_1], axis = 0)
        
        if same_covar:
            covar = ((X[idx_0]-self.mu_0).T @ (X[idx_0]-self.mu_0) + \
                    (X[idx_1]-self.mu_1).T @ (X[idx_1]-self.mu_1))/self.m
            self.covar = self.covar_0 = self.covar_1 = covar
        else:
            self.covar_0 = ((X[idx_0]-self.mu_0).T @ (X[idx_0]-self.mu_0))/self.m0
            self.covar_1 = ((X[idx_1]-self.mu_1).T @ (X[idx_1]-self.mu_1))/self.m1

    def get_boundary(self, x0, linear = False):
        assert(self.n == 2)
        if linear:
            covar_inv = np.linalg.inv(self.covar)
            n = (self.mu_1-self.mu_0) @ covar_inv # normal to line
            c = (self.mu_0 @ covar_inv @ self.mu_0 - self.mu_1 @ covar_inv @ self.mu_1)/2 + np.log(self.m1/self.m0)
            if n[1] == 0:
                return [c for i in x0]
            x1 = [(c - n[0]*x)/n[1] for x in x0]
            return x1
        else:
            covar_0_inv = np.linalg.inv(self.covar_0)
            covar_1_inv = np.linalg.inv(self.covar_1)
            A = (covar_0_inv-covar_1_inv)/2
            b = covar_1_inv @ self.mu_1 - covar_0_inv @ self.mu_0
            c = .5*(self.mu_0 @ covar_0_inv @ self.mu_0 - self.mu_1 @ covar_1_inv @ self.mu_1) +\
                 0.5*(np.log(np.linalg.det(self.covar_0)/np.linalg.det(self.covar_1))) + np.log(self.m1/self.m0)
            coeffs = lambda x: [A[1,1], b[1]+2*A[0,1]*x, A[0,0]*x**2+b[0]*x+c]
            x1r1 = []
            x1r2 = []
            for x in x0:
                roots = np.roots(coeffs(x))
                x1r1.append(min(roots))
                x1r2.append(max(roots))
            return [x1r1, x1r2]
            



X = pd.read_csv("Q4/q4x.dat", header=None, delim_whitespace=True)
Y = pd.read_csv("Q4/q4y.dat", header=None, delim_whitespace=True)

# normalise X
X = (X-np.mean(X, axis = 0))/np.std(X, axis = 0)

# make data in Y numerical, Alaska = 0, Canada = 1
names = {0:"Alaska", 1:"Canada"}
Y[1] = 0
Y.loc[Y.index[Y[0] == names[1]], 1] = 1
# drop the name columns
Y.drop(columns=[0], inplace=True)

X_train = np.array(X)
Y_train = np.array(Y).reshape((Y.shape[0],))

model = gda()
# same_covar=True for 1
model.fit(X_train, Y_train, same_covar=True)
mu_0, mu_1, covar = model.mu_0, model.mu_1, model.covar

print(f"1. mu_0 = {mu_0}, mu_1 = {mu_1}\n \033[4mvar-covar matrix\033[0m \n{covar}")

# plot the points for 2
plt.scatter(np.array(X[0].get(X.index[Y[1] == 1])), np.array(X[1].get(X.index[Y[1] == 1])),marker='.',label = names[1])
plt.scatter(np.array(X[0].get(X.index[Y[1] == 0])), np.array(X[1].get(X.index[Y[1] == 0])),marker='x',label = names[0])

# get the boundary for 3
x0 = [np.min(X_train[:,0]), np.max(X_train[:,0])]
x1 = model.get_boundary(x0, linear=True) # get x1 for the min and max of x0 to get the linear decision boundary
plt.plot(x0, x1, label = "linear decision boundary")

# new model. allowing different var-covar matrices
model1 = gda()
model1.fit(X_train, Y_train, same_covar=False)
mu_0, mu_1, covar_0, covar_1 = model1.mu_0, model1.mu_1, model1.covar_0, model1.covar_1
# print the parameters obtained
print(f"4. mu_0 = {mu_0}, mu_1 = {mu_1}\n \033[4mvar-covar matrix for label 0\033[0m \n{covar_0}\
      \n\033[4mvar-covar matrix for label 1\033[0m \n{covar_1}")

x0 = np.linspace(np.min(X_train[:,0]), np.max(X_train[:,0]),100)
x1r1, x1r2 = model1.get_boundary(x0, linear=False)
# plt.plot(x0, x1r1, label = "qudratic decision boundary branch 1", c='c') # branch 2 is the main branch
plt.plot(x0, x1r2, label = "qudratic decision boundary",c='k') 

plt.legend()
plt.xlabel("Ring diameter (fresh water)")
plt.ylabel("Ring diameter (marine water)")
plt.title("Visualization of Linear and Quadratic Decision Boundaries")

plt.show()


