import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, '../')
from src.svm import HuberSVM

class Simulated_Demo:
    def __init__(self, verbose=True, plot=False)
        self.X, self.Xy = sklearn.datasets.make_circles(noise=0.2, factor=0.5, random_state=1)
        self.X = StandardScaler().fit_transform(X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=.4, random_state=42)
        self.y_train = 2*self.y_train - 1
        self.y_test = 2*self.y_test - 1
        if verbose:
            print('Number of training examples:', self.X_train.shape[0])
            print('Number of test examples:', self.X_test.shape[0])
        if plot:
            x_min, x_max = self.X[:, 0].min() - .5, self.X[:, 0].max() + .5
            y_min, y_max = self.X[:, 1].min() - .5, self.X[:, 1].max() + .5
            h = .02  # step size in the mesh
            self.xx, self.yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))

            self.cm = plt.cm.RdBu
            self.cm_bright = ListedColormap(['#FF0000', '#0000FF'])
            # Plot the training points
            ax = plt.subplot()
            ax.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train, cmap=self.cm_bright,
                       edgecolors='k')
            # and testing points
            ax.scatter(self.X_test[:, 0], self.X_test[:, 1], c=self.y_test, cmap=self.cm_bright, alpha=0.1,
                       edgecolors='k')
            ax.set_xlim(self.xx.min(), self.xx.max())
            ax.set_ylim(self.yy.min(), self.yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            plt.show()
    def evaluate(self, beta, X_test, y_test, kernel, **kwargs):
        n_test = len(y_test)
        y_pred = np.zeros(n_test)
        y_vals = np.zeros(n_test)
        for i in range(n_test):
            y_vals[i] = np.dot(kernel(self.X_train, X_test[i, :].reshape(1, -1), **kwargs).reshape(-1), beta)
        y_pred = np.sign(y_vals)
        return np.mean(y_pred != y_test), y_vals  # return error and values from before applying cutoff
    
    def plot(self, betas, kernel, **kwargs):
        Zs = np.c_[self.xx.ravel(), self.yy.ravel()]
        error, test_values = misclassification_error(betas[-1, :], self.X_train, self.X_test, self.y_test, kernel, **kwargs)
        Z = misclassification_error(betas[-1, :], X_train, Zs, [0]*len(Zs), kernel, **kwargs)[1]
        print('Misclassification error when lambda =', kwargs['lambda'], ':', error)

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax = plt.subplot()
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot also the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   edgecolors='k', alpha=0.2)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        plt.show()