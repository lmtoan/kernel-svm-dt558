"""
This module implements ..., as described in Nesterov and Polyak (2006) and also
the adaptive cubic regularization algorithm described in Cartis et al. (2011). This code solves the cubic subproblem
according to slight modifications of Algorithm 7.3.6 of Conn et. al (2000). Cubic regularization solves unconstrained
minimization problems by minimizing a cubic upper bound to the function at each iteration.

Implementation by Toan Luong
toanlm@uw.edu
June 2018
"""

from tqdm import tqdm
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits

import sys
sys.path.insert(0, '../')
from src.svm import KernelSVM
    
def real_demo(verbose=True, plot_progress=True):
    def filter_pair(label_pair, x, y):
        """ Filter a multi-class dataset into binary dataset given a label pair.
        """
        mask = np.isin(y, label_pair)
        x_bin, y_bin = x[mask].copy(), y[mask].copy()
        y_bin[y_bin==label_pair[0]] = 1.0
        y_bin[y_bin==label_pair[1]] = -1.0
        return x_bin, y_bin

    def evaluate(beta, X_train, X_test, kernel, **kwargs):
        n_test = X_test.shape[0]
        y_pred = np.zeros(n_test)
        y_vals = np.zeros(n_test)
        for i in range(n_test):
            y_vals[i] = np.dot(kernel(X_train, X_test[i, :].reshape(1, -1), **kwargs).reshape(-1), beta)
        return y_vals

    def train_predict(X_train, y_train, X_test, y_test, method='ovo', **config):
        error = None
        label_list = np.unique(y_train)
        if method == 'ovo':
            pred_list = []
            label_pair_list = list(itertools.combinations(label_list, 2))
            for label_pair in tqdm(label_pair_list):
                X_train_bin, y_train_bin = filter_pair(label_pair, X_train, y_train)
                mylinearsvm = KernelSVM(**config)
                mylinearsvm.fit(X_train_bin, y_train_bin)
                beta_vals, train_cache = mylinearsvm.beta_vals, mylinearsvm.cache
                if config['plot']:
                    plt.show(train_cache['plot'])
                scores = evaluate(beta_vals[-1, :], X_train_bin, X_test, mylinearsvm.gram, **train_cache)
                y_pred_bin = np.zeros_like(y_test) + label_pair[-1]
                y_pred_bin[scores >= 0] = label_pair[0]
                pred_list.append(y_pred_bin)
            test_preds = np.array([mode(pi).mode[0] for pi in np.array(pred_list, dtype=np.int64).T])
            error = np.mean(test_preds != y_test)
        elif method == 'ovr':
            score_list = []
            for label in tqdm(label_list):
                y_train_bin = np.zeros_like(y_train) - 1
                y_train_bin[y_train == label] = 1
                mylinearsvm = KernelSVM(**config)
                mylinearsvm.fit(X_train, y_train_bin)
                beta_vals, train_cache = mylinearsvm.beta_vals, mylinearsvm.cache
                if config['plot']:
                    plt.show(train_cache['plot'])
                scores = evaluate(beta_vals[-1, :], X_train, X_test, mylinearsvm.gram, **train_cache)
                score_list.append(scores)
            test_preds = np.argmax(np.stack(score_list, axis=1), axis=1)
            error = np.mean(test_preds != y_test)
        else:
            print("Method Not Implemented")
        return error, beta_vals
    
    print("Download the dataset...")
    digits = load_digits() 
    num_images = digits.images.shape[0]
    X_train, X_test, y_train, y_test = train_test_split(digits.images.reshape(num_images, -1), digits.target, test_size=.4, random_state=42)
    #Standardizing the data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    if verbose:
        print('Number of training examples:', X_train.shape)
        print('Number of test examples:', X_test.shape)
    lambda_list = [1]
    for l in lambda_list:
        error, beta_vals = train_predict(X_train, y_train, X_test, y_test, method='ovo', **{'lambda': l, 'kernel_choice': 'linear', 'sigma': 1, 'plot': plot_progress, 'max_iter': 50})
        print("Lambda = %0.4f. Misclassification Error = %0.4f" %(l, error))

if __name__=='__main__':
    real_demo(verbose=True, plot_progress=False)