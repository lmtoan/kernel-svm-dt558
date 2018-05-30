"""
This module implements cubic regularization of Newton's method, as described in Nesterov and Polyak (2006) and also
the adaptive cubic regularization algorithm described in Cartis et al. (2011). This code solves the cubic subproblem
according to slight modifications of Algorithm 7.3.6 of Conn et. al (2000). Cubic regularization solves unconstrained
minimization problems by minimizing a cubic upper bound to the function at each iteration.

Implementation by Toan Luong
toanlm@uw.edu
June 2018
"""

import sklearn.metrics
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

class HuberSVM:
    def __init__(self, **kwargs):
        self.beta_vals = None
        self.cache = {}
        self.h = None
        self.sigma = None
        self.order = None
        self.lam = None
        self.eta_init = None
        self.max_iter = None
        self.eps = None
        self.kernel_choice = None
        self.plot = False
        
        if 'kernel_choice' in kwargs:
            self.kernel_choice = kwargs['kernel_choice']
        else:
            self.kernel_choice = 'linear'
        
        if self.kernel_choice == 'rbf':
            if 'sigma' in kwargs:
                self.sigma = kwargs['sigma']
            else:
                self.sigma = 1.0
        
        if self.kernel_choice == 'poly':   
            if 'order' in kwargs:
                self.order = kwargs['order']
            else:
                self.order = 2

        if 'margin' in kwargs:
            self.h = kwargs['margin']
        else:
            self.h = 0.5
        
        if 'lambda' in kwargs:
            self.lam = kwargs['lambda']
        else:
            self.lam = 1.0

        if 'max_iter' in kwargs:
            self.max_iter = kwargs['max_iter']
        else:
            self.max_iter = 50
            
        if 'eps' in kwargs:
            self.eps = kwargs['eps']
        else:
            self.eps = 1e-5
            
        if 'eta_init' in kwargs:
            self.eta_init = kwargs['eta_init']
            
        if 'plot' in kwargs:
            self.plot = kwargs['plot']

    def fit(self, X_train, y_train):
        """Fit the model
        """
        # Initialize
        n, d = X_train.shape
        beta_init = np.zeros(n)
        theta_init = np.zeros(n)

        if self.kernel_choice == 'rbf' and self.sigma is None:
            # Set sigma based on pairwise distances.
            dists = sklearn.metrics.pairwise.pairwise_distances(X_train).reshape(-1)
            self.sigma = np.median(dists)

        # Main Loop
        K = self.gram(X_train, X_train, **{'kernel_choice': self.kernel_choice, 'sigma': self.sigma, 
                                           'order': self.order})
        if self.eta_init is None:
            # Set eta_init based on an upper bound on the Lipschitz constant.
            self.eta_init = 1 / scipy.linalg.eigh(2 / n * np.dot(K, K) + 2 * self.lam * K, eigvals=(n - 1, n - 1),
                                             eigvals_only=True)[0]

        self.beta_vals = self.fastgradalgo(beta_init, theta_init, K, y_train, self.lam, self.eta_init, self.max_iter, self.eps)
        if self.plot:
            ax = self.objective_plot(self.beta_vals, K, y_train, self.lam)
            self.cache['plot'] = ax
        self.cache['kernel_choice'] = self.kernel_choice
        self.cache['sigma'] = self.sigma
        self.cache['order'] = self.order
        self.cache['eta_init'] = self.eta_init
        self.cache['lambda'] = self.lam
    
    def gram(self, X, Z, **kwargs):
        """
        Inputs: 
        - X: matrix with observations as rows
        - Z: Another matrix with observations as rows
        - Sigma: kernel bandwidth
        Output: Gram matrix
        """  
        if Z is None:
            Z = X
        if kwargs['kernel_choice'] == 'rbf':
            return np.exp(-1/(2*kwargs['sigma']**2)*((np.linalg.norm(X, axis=1)**2)[:, np.newaxis] + (np.linalg.norm(Z, axis=1)**2)[np.newaxis, :] - 2*np.dot(X, Z.T)))
        elif kwargs['kernel_choice'] == 'linear':
            return X.dot(Z.T)
        elif kwargs['kernel_choice'] == 'poly':
            return (X.dot(Z.T) + 1)**kwargs['order']
        else:
            print("Kernel Not Implemented")
            return X
        
    def fastgradalgo(self, beta_init, theta_init, K, y, lam, eta_init, max_iter, eps):
        beta = beta_init
        theta = theta_init
        eta = eta_init
        grad_theta = self.grad(theta, K, y, lam)
        grad_beta = self.grad(beta, K, y, lam)
        beta_vals = beta
        iter = 0
        while iter < max_iter and np.linalg.norm(grad_beta) > eps:
            eta = self.bt_line_search(theta, K, y, lam, eta=eta)
            beta_new = theta - eta*grad_theta
            theta = beta_new + iter/(iter+3)*(beta_new-beta)
            grad_theta = self.grad(theta, K, y, lam)
            grad_beta = self.grad(beta, K, y, lam)
            beta = beta_new
            iter += 1
            if iter % 1 == 0:
                beta_vals = np.vstack((beta_vals, beta_new))
        return beta_vals
    
    def obj(self, beta, K, y, lam):
        """
        Inputs:
        - beta: Vector to be optimized
        - K: Gram matrix consisting of evaluations of the kernel k(x_i, x_j) for i,j=1,...,n
        - y: Labels y_1,...,y_n corresponding to x_1,...,x_n
        - lam: Penalty parameter lambda
        Output:
        - Value of the objective function at beta
        """
        h = self.h
        cost_vector = np.zeros(y.shape[0])
        t = K.dot(beta)
        yt = y * t
        cost_vector[yt < 1-h] = (1 - yt[yt < 1 - h])
        cost_vector[np.absolute(1 - yt) <= h] = (((1 + h - yt[np.absolute(1 - yt) <= h])**2) / (4 * h))
        return (1/y.shape[0]) * np.sum(cost_vector) + lam * beta.dot(K).dot(beta)

    def grad(self, beta, K, y, lam):
        """
        Inputs:
        - beta: Vector to be optimized
        - K: Gram matrix consisting of evaluations of the kernel k(x_i, x_j) for i,j=1,...,n
        - y: Labels y_1,...,y_n corresponding to x_1,...,x_n
        - lam: Penalty parameter lambda
        Output:
        - Value of the gradient at beta
        """
        h = self.h
        grad_matrix = np.zeros((y.shape[0], y.shape[0]))
        t = K.dot(beta)
        yt = y * t
        grad_matrix[yt < 1 - h] = (-y[:, np.newaxis] * K)[yt < 1 - h]
        grad_matrix[np.absolute(1 - yt) <= h] = (((-2 / (4 * h)) * (1 + h - yt))[:, np.newaxis] * (y[:, np.newaxis] * K))[np.absolute(1 - yt) <= h]
        return 1/y.shape[0] * np.sum(grad_matrix, axis=0) + 2 * lam * K.dot(beta)

    def bt_line_search(self, beta, K, y, lam, eta=1, alpha=0.5, betaparam=0.8, max_iter=100):
        grad_beta = self.grad(beta, K, y, lam)
        norm_grad_beta = np.linalg.norm(grad_beta)
        found_eta = 0
        iter = 0
        while found_eta == 0 and iter < max_iter:
            if self.obj(beta - eta * grad_beta, K, y, lam) < \
                            self.obj(beta, K, y, lam) - alpha * eta * norm_grad_beta ** 2:
                found_eta = 1
            elif iter == max_iter-1:
                raise ('Max number of iterations of backtracking line search reached')
            else:
                eta *= betaparam
                iter += 1
        return eta
    
    def objective_plot(self, betas, K, y, lam):
        num_points = np.size(betas, 0)
        objs = np.zeros(num_points)
        for i in range(0, num_points):
            objs[i] = self.obj(betas[i, :], K, y, lam)
        fig, ax = plt.subplots()
        ax.plot(np.array(range(num_points)), objs, c='red')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective value')
        ax.set_title('Objective value vs. iteration when lambda=' + str(lam))
        return ax