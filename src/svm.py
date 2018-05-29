import sklearn.metrics
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

class HuberSVM:
    def __init__(self, h=None):
        self.beta_vals = None
        self.cache = {}
        self.h = h
        
    def fit(self, X_train, y_train, lam=1, **config):
        """Fit the model
        """
        # Initialize
        n, d = X_train.shape
        beta_init = np.zeros(n)
        theta_init = np.zeros(n)
        K = None
        sigma = None
        order = None
        eta_init = None
        max_iter = None
        eps = None
        kernel_choice = None
        plot = False
        
        if 'kernel_choice' in config:
            kernel_choice = config['kernel_choice']
        else:
            kernel_choice = 'linear'
        
        if 'sigma' in config:
            sigma = config['sigma']
        elif sigma is None and kernel_choice=='rbf':
            # Set sigma based on pairwise distances.
            dists = sklearn.metrics.pairwise.pairwise_distances(X_train).reshape(-1)
            sigma = np.median(dists)
            
        if 'order' in config:
            order = config['order']
        elif order is None and kernel_choice=='poly':
            order = 2
        
        if 'max_iter' in config:
            max_iter = config['max_iter']
        else:
            max_iter = 50
            
        if 'eps' in config:
            eps = config['eps']
        else:
            eps = 1e-5
            
        if 'eta_init' in config:
            eta_init = config['eta_init']
            
        if 'plot' in config:
            plot = config['plot']
        
        # Main Loop
        K = self.gram(X_train, X_train, **{'kernel_choice': kernel_choice, 'sigma': sigma, 
                                           'order': order})
        if 'eta_init' not in config:
            # Set eta_init based on an upper bound on the Lipschitz constant.
            eta_init = 1 / scipy.linalg.eigh(2 / n * np.dot(K, K) + 2 * lam * K, eigvals=(n - 1, n - 1),
                                             eigvals_only=True)[0]
        self.beta_vals = self.fastgradalgo(beta_init, theta_init, K, y_train, lam, eta_init, max_iter, eps)
        if plot:
            ax = self.objective_plot(self.beta_vals, K, y_train, lam)
            self.cache['plot'] = ax
        self.cache['kernel_choice'] = kernel_choice
        self.cache['sigma'] = sigma
        self.cache['order'] = order
        self.cache['eta_init'] = eta_init
        self.cache['lambda'] = lam
    
    def gram(self, X, Z, **config):
        """
        Inputs: 
        - X: matrix with observations as rows
        - Z: Another matrix with observations as rows
        - Sigma: kernel bandwidth
        Output: Gram matrix
        """  
        if Z is None:
            Z = X
        if config['kernel_choice'] == 'rbf':
            return np.exp(-1/(2*config['sigma']**2)*((np.linalg.norm(X, axis=1)**2)[:, np.newaxis] + (np.linalg.norm(Z, axis=1)**2)[np.newaxis, :] - 2*np.dot(X, Z.T)))
        elif config['kernel_choice'] == 'linear':
            return X.dot(Z.T)
        elif config['kernel_choice'] == 'poly':
            return (X.dot(Z.T) + 1)**config['order']
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
    
    def obj(self, beta, K, y, lam, h=0.5):
        """
        Inputs:
        - beta: Vector to be optimized
        - K: Gram matrix consisting of evaluations of the kernel k(x_i, x_j) for i,j=1,...,n
        - y: Labels y_1,...,y_n corresponding to x_1,...,x_n
        - lam: Penalty parameter lambda
        Output:
        - Value of the objective function at beta
        """
        if self.h is not None:
            h = self.h
        cost_vector = np.zeros(y.shape[0])
        t = K.dot(beta)
        yt = y * t
        cost_vector[yt < 1-h] = (1 - yt[yt < 1 - h])
        cost_vector[np.absolute(1 - yt) <= h] = (((1 + h - yt[np.absolute(1 - yt) <= h])**2) / (4 * h))
        return (1/y.shape[0]) * np.sum(cost_vector) + lam * beta.dot(K).dot(beta)

    def grad(self, beta, K, y, lam, h=0.5):
        """
        Inputs:
        - beta: Vector to be optimized
        - K: Gram matrix consisting of evaluations of the kernel k(x_i, x_j) for i,j=1,...,n
        - y: Labels y_1,...,y_n corresponding to x_1,...,x_n
        - lam: Penalty parameter lambda
        Output:
        - Value of the gradient at beta
        """
        if self.h is not None:
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