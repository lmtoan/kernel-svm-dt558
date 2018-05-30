import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, '../')
from src.svm import HuberSVM

def sim_demo(verbose=True, plot_contour=False):
    def evaluate(beta, X_train, X_test, y_test, kernel, **kwargs):
        n_test = len(y_test)
        y_pred = np.zeros(n_test)
        y_vals = np.zeros(n_test)
        for i in range(n_test):
            y_vals[i] = np.dot(kernel(X_train, X_test[i, :].reshape(1, -1), **kwargs).reshape(-1), beta)
        y_pred = np.sign(y_vals)
        return np.mean(y_pred != y_test), y_vals  # return error and values from before applying cutoff
    
    def evaluate_plot(betas, kernel, plot_contour, **kwargs):
        error, test_values = evaluate(betas[-1, :], X_train, X_test, y_test, kernel, **kwargs)
        print('Misclassification error when lambda =', kwargs['lambda'], ':', error)
        ax = None
        if plot_contour:
            Zs = np.c_[xx.ravel(), yy.ravel()]
            Z = evaluate(betas[-1, :], X_train, Zs, [0]*len(Zs), kernel, **kwargs)[1]
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
        return error, test_values, ax
        
    X, y = sklearn.datasets.make_circles(noise=0.2, factor=0.5, random_state=1)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)
    y_train = 2*y_train - 1
    y_test = 2*y_test - 1
    if verbose:
        print('Number of training examples:', X_train.shape[0])
        print('Number of test examples:', X_test.shape[0])

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = .02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    if plot_contour:
        # Plot the training points
        ax = plt.subplot()
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.1,
                   edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())

    mysvm = HuberSVM(**{'plot': False, 'kernel_choice': 'rbf', 'sigma': 1, 'max_iter': 10, 'lambda': 0.8, 'margin': 1})
    mysvm.fit(X_train, y_train)
    beta_vals = mysvm.beta_vals
    train_cache = mysvm.cache
    errors, test_vals, contour_ax = evaluate_plot(beta_vals, mysvm.gram, plot_contour, **train_cache)
    plt.show()
    
if __name__=='__main__':
    sim_demo(verbose=True, plot_contour=False)