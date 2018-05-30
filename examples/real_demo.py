"""
This module demonstrates the 3 types of KernelSVM: linear, polynomial order 3, and rbf,
by applying to the Digits dataset as provided in the following link.

http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html.

All plots are stored in '../images' as default.

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
sys.path.insert(0, '../') # Insert the root directory to path.
from src.svm import KernelSVM
    
def real_demo(plot_cache_path, verbose=True, plot_progress=True):
    """
    Main function to run the real-data demonstration.

    Args:
        plot_cache_path: Path to store plots/images.
        verbose: Set to True for extensive outputs.
        plot_progress: Set to True for progress plots.
    """
    def filter_pair(label_pair, x, y):
        """
        Filter a multi-class dataset into binary dataset given a label pair.

        Args:
            label_pair: [label_x, label_y] to binarize the dataset.
            x: A (n, d) input matrix.
            y: A (d,) label vector.

        Returns:
            x_bin: Example rows of only label_x and label_y.
            y_bin: A (d,) label vector. label_x is 1.0 and label_y is -1.0.
        """
        mask = np.isin(y, label_pair)
        x_bin, y_bin = x[mask].copy(), y[mask].copy()
        y_bin[y_bin==label_pair[0]] = 1.0
        y_bin[y_bin==label_pair[1]] = -1.0
        return x_bin, y_bin

    def evaluate(beta, X_train, X_test, kernel, **train_cache):
        """
        Given a training cache and trained beta/weight vector, compute the predicted scores.
        The result is the dot product between the K(X_train, X_test) and beta.

        Args:
            beta: Final beta/weight vector of shape (d,).
            X_train: Training matrix.
            X_test: Test matrix.
            kernel: Function to compute the gram matrix.
            **train_cache: Keyword arguments cached from training to compute the correct Gram matrix.

        Returns:
            y_vals: Predicted score vector of shape (len(X_test),).
        """
        n_test = X_test.shape[0]
        y_pred = np.zeros(n_test)
        y_vals = np.zeros(n_test)
        for i in range(n_test):
            y_vals[i] = np.dot(kernel(X_train, X_test[i, :].reshape(1, -1), **train_cache).reshape(-1), beta)
        return y_vals

    def multiclass_train(X_train, y_train, X_test, y_test, method='ovo', **config):
        """
        Provide 2 options: one-vs-one ('ovo') and one-vs-rest ('ovr') to train multiclass dataset using Kernel SVM.

        Args:
            X_train: Training input matrix of shape (n,d).
            y_train: Training label vector of shape (n,).
            X_test: Test/validation input matrix of shape (n_test,d).
            y_test: Test/validation label vector of shape (n,).
            method: 'ovo' or 'ovr'.
            **config: Training configurations that are appropriate to the KernelSVM class.
        
        Returns:
            error: Misclassfication error on the test/validation set.
            beta_vals: An array of beta/weight values at each iterations.
        """
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
    # Standardizing the data.
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    if verbose:
        print('Number of training examples:', X_train.shape)
        print('Number of test examples:', X_test.shape)
    # Specify the list of lambdas to experiment.
    lambda_list = [0.01]
    for l in lambda_list:
        error, beta_vals = multiclass_train(X_train, y_train, X_test, y_test, method='ovo', **{'lambda': l, 'kernel_choice': 'linear', 'sigma': 1, 'plot': plot_progress, 'max_iter': 50})
        print("Lambda = %0.4f. Misclassification Error = %0.4f" %(l, error))

if __name__=='__main__':
    cache_path = '../images'
    real_demo(cache_path, verbose=True, plot_progress=False)