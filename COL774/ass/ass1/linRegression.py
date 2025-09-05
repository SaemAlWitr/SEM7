import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import seaborn as sns
from matplotlib.animation import FuncAnimation

parser = argparse.ArgumentParser()
parser.add_argument("-lr", type=float, default=0.1)
parser.add_argument("-e", type=float, default=1e-6)
parser.add_argument("--verbose", action="store_true")

args = parser.parse_args()

class regressor():
    def __init__(self):
        self.theta = 0
        self.lr = 0.01
        self.m, self.n = 0, 0
        self.eps = 1e-6

    def J(self, X, Y,theta):
        return (np.sum(np.square((Y-(X@theta.T)))))/2/self.m
    
    def gradiant(self, X, Y, theta):
        return X.T@((X@theta.T) - Y)/self.m
    
    def fit(self, X, Y, lr = 0.01, verbose = 0):
        self.m = X.shape[0]
        self.lr = lr
        X = np.hstack((np.ones((self.m,1)), X)) # X = [1|X]
        self.n = X.shape[1]
        self.theta = np.zeros((self.n,))
        cost = self.J(X, Y,self.theta)
        # previous_cost = -np.inf
        history = [(cost,self.theta[0], self.theta[1])]
        iter = 1
        grad = self.gradiant(X, Y, self.theta)
        while np.max(np.abs(grad)) > self.eps:
            if verbose:
                print(iter, cost)
            self.theta = self.theta - grad.T*self.lr
            # previous_cost = cost
            cost = self.J(X, Y,self.theta)
            grad = self.gradiant(X, Y, self.theta)

            history.append((cost,self.theta[0], self.theta[1]))
            iter += 1
        return history
    
    def predict(self, X):
        return (X @ self.theta).T

def plot(X, Y, model, history = None, hypothesis = 0, cost3d = 0, cost2d = 0, freq = 200):
    if(len(X.shape) > 1):
        X.reshape((max(X.shape), ))
    if hypothesis:
        X_plot = np.linspace(np.min(X),np.max(X),2)
        Y_plot = model.theta[0]+model.theta[1]*X_plot
        plt.figure(figsize=(6,4))
        plt.plot(X_plot, Y_plot,label = 'hypothesis')
        sns.scatterplot(x = X[::2], y = Y[::2],marker='o',color='red', alpha = 0.4,label = 'data-points')
        plt.title("Plot of datapoints and the hypothesis function \n lr = "+str(model.lr))
        plt.xlabel("Acidity")
        plt.xlabel("Density")
        plt.legend()
        plt.show()
    if cost3d or cost2d:
        theta0 = np.linspace(model.theta[0]-30,model.theta[1]+15,300)
        theta1 = np.linspace(model.theta[1]-30,model.theta[1]+30,300)
        theta0, theta1 = np.meshgrid(theta0, theta1)
        mse = np.zeros_like(theta0)
        y_pred = theta0[None, :, :] + theta1[None,:,:] * X[:, None, None]
        mse = np.mean((Y[:,None, None] - y_pred)**2/2, axis = 0)
        if cost3d:
            sns.set_theme(style="whitegrid")
            fig = plt.figure()
            ax = fig.add_subplot(111,projection = '3d')
            ax.set_title("3D mesh plot for J(θ) \n lr = "+str(model.lr))
            p = ax.plot_surface(theta0, theta1, mse, cmap="viridis", edgecolor="none", alpha=0.5, label = 'J(θ)')
            fig.colorbar(p, ax=ax, pad=0.01, label="MSE")
            scat = ax.scatter(history[:0,1], history[:0,2], history[:0,0], c = 'blue', alpha = 0.6, s = 40, marker = '.', depthshade = True, label = 'gradient descent trajectory')
            def update(frameno):
                theta0 = history[:frameno,1]
                theta1 = history[:frameno,2]
                mse = history[:frameno,0]
                scat._offsets3d = (theta0,theta1,mse)
                scat.set_array(mse)
                return scat,
            anim = FuncAnimation(fig, update, frames=np.arange(1, len(history)+1), interval=200, blit=False)
            ax.set_xlabel("θ0 (intercept)")
            ax.set_ylabel("θ1 (slope)")
            ax.set_zlabel("MSE")
            ax.legend()
            ax.view_init(elev=30, azim=30)
            plt.show()
        if cost2d:
            fig = plt.figure()
            ax2 = fig.add_subplot()
            cont = ax2.contour(theta0, theta1, mse, levels=70, cmap="viridis",alpha = 0.7)
            ax2.set_title("Contour plot for J(θ) \n lr = "+str(model.lr))
            fig.colorbar(cont, ax=ax2, label="MSE")
            scat2 = ax2.scatter(history[:0,1], history[:0,2], c='blue', alpha=0.8, s=40, marker='.',label = 'gradient descent trajectory')
            def init2():
                scat2.set_offsets(np.empty((0, 2)))
                return scat2,
            def update2(frameno):
                theta0 = history[:frameno,1]
                theta1 = history[:frameno,2]
                scat2.set_offsets(np.column_stack((theta0,theta1)))
                return scat2,
            anim2 = FuncAnimation(fig, update2, init_func=init2, frames=np.arange(1, len(history)+1), interval=freq, blit=False)
            ax2.set_xlabel("θ0 (intercept)")
            ax2.set_ylabel("θ1 (slope)")
            ax2.legend()
            plt.show()

# load the data
X, Y = 0, 0
with open("Q1/linearX.csv","r") as f:
    s = f.read()
    X = np.array(list(map(float, s[:-1].split('\n'))))
with open("Q1/linearY.csv","r") as f:
    s = f.read()
    Y = np.array(list(map(float, s[:-1].split('\n'))))

X_train = X.reshape((X.shape[0],1))
assert(len(X_train.shape) == 2 and len(Y.shape) == 1)

# visualize the training data
if(args.verbose):
    plt.scatter(X.T, Y)
    plt.show()

model = regressor()
# save the history (cost, past_theta_values)
history = np.array(model.fit(X_train,Y,lr = args.lr,verbose = args.verbose))
print(f"lr = {model.lr}, stopping criterion: l_infinity(grad) < {model.eps}, final parameters = {model.theta}")
plot(X,Y,model,history,hypothesis=1,cost3d=1,cost2d=1)

for lr in [0.001, 0.025, 0.1]:
    model = regressor()
    history = np.array(model.fit(X_train,Y,lr = lr))
    if(args.verbose):
        print(lr, history.shape,model.theta)
    plot(X,Y,model,history,cost2d=1,freq=200) # faster updates for small lr's
    