#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 01:41:45 2019

@author: kenneth
"""

from __future__ import absolute_import
import numpy as np

class Kernels:
    '''
    Kernels are mostly used for solving
    non-lineaar problems. By projecting/transforming
    our data into a subspace, making it easy to
    almost accurately classify our data as if it were
    still in linear space.
    '''
    def __init__(self):
        return
    
    @staticmethod
    def linear(x1, x2, c = None):
        '''
        Linear kernel
        ----------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :return type: kernel(Gram) matrix
        '''
        if not c:
            c = 0
        else:
            c = c
        return x1.dot(x2.T) + c
    
    
    @staticmethod
    def linear_svdd(x1, x2, c = None):
        '''
        Linear kernel
        ----------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :return type: kernel(Gram) matrix
        '''
        if not c:
            c = 0
        else:
            c = c
        return x1.T.dot(x2) + c
    
    @staticmethod
    def rbf(x1, x2, gamma = None):
        '''
        RBF: Radial basis function or guassian kernel
        ----------------------------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :param: gamma: 1/2(sigma-square)
        :return type: kernel(Gram) matrix
        '''
        if not gamma:
            gamma = 1
        else:
            gamma = gamma
        if x1.ndim == 1 and x2.ndim == 1:
            return np.exp(-gamma * np.linalg.norm(x1 - x2)**2)
        elif (x1.ndim > 1 and x2.ndim == 1) or (x1.ndim == 1 and x2.ndim > 1):
            return np.exp(-gamma * np.linalg.norm(x1 - x2, axis = 1)**2)
        elif x1.ndim > 1 and x2.ndim > 1:
            return np.exp(-gamma * np.linalg.norm(x1[:, np.newaxis] - x2[np.newaxis, :], axis = 2)**2)
       
    @staticmethod
    def laplacian(x1, x2, gamma = None):
        '''
        RBF: Radial basis function or guassian kernel
        ----------------------------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :param: gamma: 1/2(sigma-square)
        :return type: kernel(Gram) matrix
        '''
        if not gamma:
            gamma = 1
        else:
            gamma = gamma
        if x1.ndim == 1 and x2.ndim == 1:
            return np.exp(-gamma * np.linalg.norm(x1 - x2))
        elif (x1.ndim > 1 and x2.ndim == 1) or (x1.ndim == 1 and x2.ndim > 1):
            return np.exp(-gamma * np.linalg.norm(x1 - x2, axis = 1))
        elif x1.ndim > 1 and x2.ndim > 1:
            return np.exp(-gamma * np.linalg.norm(x1[:, np.newaxis] - x2[np.newaxis, :], axis = 2))
        
    
    @staticmethod
    def locguass(x1, x2, d = None, gamma = None):
        '''
        :local guassian
        '''
        if not gamma:
            gamma = 1
        else:
            gamma = gamma
        if not d:
            d = 5
        else:
            d = d
        if x1.ndim == 1 and x2.ndim == 1:
            return (np.maximum(0, 1 - gamma*np.linalg.norm(x1 - x2)/3)**d)*np.exp(-gamma * np.linalg.norm(x1 - x2)**2)
        elif (x1.ndim > 1 and x2.ndim == 1) or (x1.ndim == 1 and x2.ndim > 1):
            return (np.maximum(0, 1 - gamma*np.linalg.norm(x1 - x2, axis = 1)/3)**d)*np.exp(-gamma * np.linalg.norm(x1 - x2, axis = 1)**2)
        elif x1.ndim > 1 and x2.ndim > 1:
            return (np.maximum(0, 1 - gamma*np.linalg.norm(x1[:, np.newaxis] - x2[np.newaxis, :], axis = 2)/3)**d) * np.exp(-gamma * np.linalg.norm(x1[:, np.newaxis] - x2[np.newaxis, :], axis = 2)**2)
    
    @staticmethod
    def chi(x):
        '''
        Using Chisquared from sklearn
        '''
        from sklearn.metrics.pairwise import chi2_kernel
        return chi2_kernel(x)
    
    @staticmethod
    def sigmoid(x1, x2, gamma = None, c = None):
        '''
        logistic or sigmoid kernel
        ----------------------------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :param: gamma: 1/2(sigma-square)
        :return type: kernel(Gram) matrix
        '''
        if not gamma:
            gamma = 1
        else:
            gamma = gamma
        if not c:
            c = 1
        return np.tanh(gamma * x1.dot(x2.T) + c)
    
    
    @staticmethod
    def polynomial(x1, x2, d = None, c = None):
        '''
        polynomial kernel
        ----------------------------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :param: d: polynomial degree
        :return type: kernel(Gram) matrix
        '''
        if not d:
            d = 3
        else:
            d = d
        return (x1.dot(x2.T))**d
    
    @staticmethod
    def cosine(x1, x2):
        '''
        Cosine kernel
        ----------------------------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :return type: kernel(Gram) matrix
        '''
        
        return (x1.dot(x2.T)/np.linalg.norm(x1, 1) * np.linalg.norm(x2, 1))
    
    @staticmethod
    def correlation(x1, x2, gamma = None):
        '''
        Correlation kernel
        ----------------------------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :param: gamma: 1/2(sigma-square)
        :return type: kernel(Gram) matrix
        '''
        if not gamma:
            gamma = 1
        else:
            gamma = gamma
        return np.exp((x1.dot(x2.T)/np.linalg.norm(x1, 1) * np.linalg.norm(x2, 1)) - 1/gamma)
    
    @staticmethod
    def linrbf(x1, x2, gamma = None, op = None):
        '''
        MKL: Lineaar + RBF kernel
        ----------------------------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :param: gamma: 1/2(sigma-square)
        :return type: kernel(Gram) matrix
        '''
        if not gamma:
            gamma = 1
        else:
            gamma = gamma
        if not op:
            op = 'multiply' #add seems like the best performning here
        else:
            op = op
        if op == 'multiply':
            return Kernels.linear(x1, x2) * Kernels.rbf(x1, x2, gamma)
        elif op == 'add':
            return Kernels.linear(x1, x2) + Kernels.rbf(x1, x2, gamma)
        elif op == 'divide':
            return Kernels.linear(x1, x2) / Kernels.rbf(x1, x2, gamma)
        elif op == 'subtract':
            return np.abs(Kernels.linear(x1, x2) - Kernels.rbf(x1, x2, gamma))
        elif op == 'dot':
            return Kernels.linear(x1, x2).dot(10000*Kernels.rbf(x1, x2, gamma).T)
        
    @staticmethod
    def rbfpoly(x1, x2, d = None, gamma = None, op = None):
        '''
        MKL: RBF + Polynomial kernel
        ----------------------------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :param: gamma: 1/2(sigma-square)
        :return type: kernel(Gram) matrix
        '''
        if not gamma:
            gamma = 1
            gamma = gamma
        if not d:
            d = 5
        else:
            d = d
        if not op:
            op = 'multiply'
        else:
            op = op
        if op == 'multiply':
            return Kernels.polynomial(x1, x2, d) * Kernels.rbf(x1, x2, gamma)
        elif op == 'add':
            return Kernels.polynomial(x1, x2, d) + Kernels.rbf(x1, x2, gamma)
        elif op == 'divide':
            return Kernels.polynomial(x1, x2, d) / Kernels.rbf(x1, x2, gamma)
        elif op == 'subtract':
            return np.abs(Kernels.polynomial(x1, x2, d) - Kernels.rbf(x1, x2, gamma))
        elif op == 'dot':
            return Kernels.polynomial(x1, x2, d).dot(10000*Kernels.polynomial(x1, x2, d).T)
        
    @staticmethod
    def rbfcosine(x1, x2, gamma = None, op = None):
        '''
        MKL: RBF + Polynomial kernel
        ----------------------------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :param: gamma: 1/2(sigma-square)
        :return type: kernel(Gram) matrix
        '''
        if not gamma:
            gamma = 1
        else:
            gamma = gamma
        if not op:
            op = 'multiply'
        else:
            op = op
        if op == 'multiply':
            return Kernels.cosine(x1, x2) * Kernels.rbf(x1, x2, gamma)
        elif op == 'add':
            return Kernels.cosine(x1, x2) + Kernels.rbf(x1, x2, gamma)
        elif op == 'divide':
            return Kernels.cosine(x1, x2) / Kernels.rbf(x1, x2, gamma)
        elif op == 'subtract':
            return np.abs(Kernels.cosine(x1, x2) - Kernels.rbf(x1, x2, gamma))
        elif op == 'dot':
            return Kernels.cosine(x1, x2).dot(10000*Kernels.cosine(x1, x2).T)
        
    @staticmethod
    def etakernel(x1, x2, d = None, gamma = None, op = None):
        '''
        MKL: Pavlidis et al. (2001)
        ----------------------------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :param: gamma: 1/2(sigma-square)
        :return type: kernel(Gram) matrix
        '''
        if not gamma:
            gamma = 1
        else:
            gamma = gamma
        if not op:
            op = 'etamul'
        else:
            op = op
        if not d:
            d = 5
        else:
            d = d
        if op == 'eta':
            return Kernels.linrbf(x1, x2).dot(Kernels.rbfpoly(x1, x2))
        elif op == 'etasum':
            return Kernels.linrbf(x1, x2) + Kernels.rbfpoly(x1, x2)
        elif op == 'etamul':
            return Kernels.linrbf(x1, x2) * Kernels.rbfpoly(x1, x2)
        elif op == 'etadiv':
            return Kernels.linrbf(x1, x2) / Kernels.rbfpoly(x1, x2)
        elif op == 'etapoly':
            return Kernels.linrbf(x1, x2).dot(Kernels.rbfpoly(x1, x2)) + Kernels.rbfpoly(x1, x2).dot(Kernels.rbfcosine(x1, x2))
        elif op == 'etasig':
            return Kernels.sigmoid(x1, x2).dot(Kernels.rbf(x1, x2)) + Kernels.rbfpoly(x1, x2).dot(Kernels.sigmoid(x1, x2))
        elif op == 'etaalpha':
            return Kernels.rbf(Kernels.linear(x1, x2).dot(Kernels.rbfpoly(x1, x2)), Kernels.sigmoid(x1, x2) + Kernels.polynomial(x1, x2))
            
    @staticmethod
    def alignment(x1, x2, d = None, gamma = None, op = None):
        '''
        MKL: Cortes et al.
        ----------------------------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :param: gamma: 1/2(sigma-square)
        :return type: kernel(Gram) matrix
        '''
        if not gamma:
            gamma = 1
        else:
            gamma = gamma
        if not op:
            op = 'rbfpoly'
        else:
            op = op
        if not d:
            d = 5
        else:
            d = d
        
        kappa_lin = Kernels.linear(x1, x2) - 1/len(x1)*np.ones((x1.shape[0], x1.shape[0])).dot(np.ones((x1.shape[0], x1.shape[0])).T).dot(Kernels.linear(x1, x2))\
                     - 1/len(x1)*Kernels.linear(x1, x2).dot(np.ones((x1.shape[0], x1.shape[0])).dot(np.ones((x1.shape[0], x1.shape[0])).T)) + \
                     1/len(x1)**2*(np.ones((x1.shape[0], x1.shape[0])).T.dot(Kernels.linear(x1, x2))).dot(np.ones((x1.shape[0], x1.shape[0]))).dot(np.ones((x1.shape[0], x1.shape[0])).dot(np.ones((x1.shape[0], x1.shape[0])).T))
        kappa_rbf = Kernels.rbf(x1, x2) - 1/len(x1)*np.ones((x1.shape[0], x1.shape[0])).dot(np.ones((x1.shape[0], x1.shape[0])).T).dot(Kernels.rbf(x1, x2))\
                     - 1/len(x1)*Kernels.rbf(x1, x2).dot(np.ones((x1.shape[0], x1.shape[0])).dot(np.ones((x1.shape[0], x1.shape[0])).T)) + \
                     1/len(x1)**2*(np.ones((x1.shape[0], x1.shape[0])).T.dot(Kernels.rbf(x1, x2))).dot(np.ones((x1.shape[0], x1.shape[0]))).dot(np.ones((x1.shape[0], x1.shape[0])).dot(np.ones((x1.shape[0], x1.shape[0])).T))    
        kappa_poly = Kernels.polynomial(x1, x2) - 1/len(x1)*np.ones((x1.shape[0], x1.shape[0])).dot(np.ones((x1.shape[0], x1.shape[0])).T).dot(Kernels.polynomial(x1, x2))\
                     - 1/len(x1)*Kernels.polynomial(x1, x2).dot(np.ones((x1.shape[0], x1.shape[0])).dot(np.ones((x1.shape[0], x1.shape[0])).T)) + \
                     1/len(x1)**2*(np.ones((x1.shape[0], x1.shape[0])).T.dot(Kernels.polynomial(x1, x2))).dot(np.ones((x1.shape[0], x1.shape[0]))).dot(np.ones((x1.shape[0], x1.shape[0])).dot(np.ones((x1.shape[0], x1.shape[0])).T))
        kappa_sigmoid = Kernels.sigmoid(x1, x2) - 1/len(x1)*np.ones((x1.shape[0], x1.shape[0])).dot(np.ones((x1.shape[0], x1.shape[0])).T).dot(Kernels.sigmoid(x1, x2))\
                     - 1/len(x1)*Kernels.sigmoid(x1, x2).dot(np.ones((x1.shape[0], x1.shape[0])).dot(np.ones((x1.shape[0], x1.shape[0])).T)) + \
                     1/len(x1)**2*(np.ones((x1.shape[0], x1.shape[0])).T.dot(Kernels.sigmoid(x1, x2))).dot(np.ones((x1.shape[0], x1.shape[0]))).dot(np.ones((x1.shape[0], x1.shape[0])).dot(np.ones((x1.shape[0], x1.shape[0])).T))
        if op == 'linrbf':
            return kappa_lin.dot(kappa_rbf)/np.sqrt(np.sum(kappa_lin.dot(kappa_lin))*np.sum(kappa_rbf.dot(kappa_rbf)))
        elif op == 'rbfpoly':
            return kappa_rbf.dot(kappa_poly)/np.sqrt(np.sum(kappa_rbf.dot(kappa_rbf))*np.sum(kappa_poly.dot(kappa_poly)))
        elif op == 'rbfsig':
            return kappa_rbf.dot(kappa_sigmoid)/np.sqrt(np.sum(kappa_rbf.dot(kappa_rbf))*np.sum(kappa_sigmoid.dot(kappa_sigmoid)))
        elif op == 'polysig':
            return kappa_poly.dot(kappa_sigmoid)/np.sqrt(np.sum(kappa_poly.dot(kappa_poly))*np.sum(kappa_sigmoid.dot(kappa_sigmoid)))
        return
    

class kkMeans(Kernels):
    def __init__(self, k = None, kernel = None, gamma = None, d = None):
        '''Kernel KMeans Algorithm
        :param: k: number of clusters
        :param: kernel
        
        '''
        super().__init__()
        if not gamma:
            gamma = 5
            self.gamma = gamma
        else:
            self.gamma = gamma
        if not d:
            d = 3
            self.d = d
        else:
            self.d = d
        if not k:
            k = 32
            self.k = k
        else:
            self.k = k
        if not kernel:
            kernel = 'linear'
            self.kernel = kernel
        else:
            self.kernel = kernel
        return
    
    def kernelize(self, x1, x2):
        '''
        :params: X: NxD
        '''
        if self.kernel == 'linear':
            return Kernels.linear(x1, x2)
        elif self.kernel == 'rbf':
            return Kernels.rbf(x1, x2, gamma = self.gamma)
        elif self.kernel == 'sigmoid':
            return Kernels.sigmoid(x1, x2, gamma = self.gamma)
        elif self.kernel == 'polynomial':
            return Kernels.polynomial(x1, x2, d = self.d)
        elif self.kernel == 'cosine':
            return Kernels.cosine(x1, x2)
        elif self.kernel == 'correlation':
            return Kernels.correlation(x1, x2, gamma = self.gamma)
        elif self.kernel == 'linrbf':
            return Kernels.linrbf(x1, x2, gamma = self.gamma)
        elif self.kernel == 'rbfpoly':
            return Kernels.rbfpoly(x1, x2, d = self.d, gamma = self.gamma)
        elif self.kernel == 'rbfcosine':
            return Kernels.rbfpoly(x1, x2, d = self.d, gamma = self.gamma)
        elif self.kernel == 'etakernel':
            return Kernels.etakernel(x1, x2, d = self.d, gamma = self.gamma)
        elif self.kernel == 'alignment':
            return Kernels.alignment(x1, x2)
        elif self.kernel == 'laplace':
            return Kernels.laplacian(x1, x2, gamma = self.gamma)
        elif self.kernel == 'locguass':
            return Kernels.locguass(x1, x2, d = self.d, gamma = self.gamma)
        elif self.kernel == 'chi':
            return Kernels.chi(x1)
    
    #compute suqared kernel distance
    def distance(self, kappa):
        self.dist_w = np.zeros(self.k)
        c_k_init = np.ones(kappa.shape[0])
        for ii in range(self.k):
            self.c_k_indices = self.clusters == ii
            self.c_k = np.sum(c_k_init[self.c_k_indices])
            self.c_k_squared = np.square(self.c_k)
            self.kappa_ii = kappa[self.c_k_indices][:, self.c_k_indices]
            self.distance_ii[:, ii] += np.sum(self.kappa_ii)/self.c_k_squared - 2*\
                                    np.sum(kappa[:, self.c_k_indices], axis = 1)/self.c_k
        return self.distance_ii
           
    #fit and return cluster labels
    def fit_predict(self, X, iteration = None, halt = None):
        '''
        :param: X: NxD Feature
        :param: iteration: 100
        :param: tolerance: 1e-3 default
        
        '''
        if not halt:
            halt = 1e-3
            self.halt = halt
        else:
            self.halt = halt
        if not iteration:
            iteration = 100
            self.iteration = iteration
        else:
            self.iteration = iteration
        self.X = X
        N, D = X.shape
        self.distance_ii = np.zeros((N, self.k))
        self.kx = self.kernelize(self.X, self.X)
        self.clusters = np.random.randint(self.k, size = N)
        '''iterate by checking to see if new and previous cluster
        labels are similar. If they are, algorithm converges and halts..
        '''
        for ii in range(self.iteration):
            #compute distance
            self.distance_k = self.distance(self.kx)
            self.prev_clusters = self.clusters
            self.clusters = self.distance_k.argmin(axis=1)
            if 1 - float(np.sum((self.clusters - self.prev_clusters) == 0)) / N < self.halt:
                break
        return self

    def rand_index_score(self, clusters, classes):
        '''Compute the RandIndex
        :param: Clusters: Cluster labels
        :param: classses: Actual class
        :returntype: float
        '''
        from scipy.special import comb
        tp_fp = comb(np.bincount(clusters), 2).sum()
        tp_fn = comb(np.bincount(classes), 2).sum()
        A = np.c_[(clusters, classes)]
        tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
                 for i in set(clusters))
        fp = tp_fp - tp
        fn = tp_fn - tp
        tn = comb(len(A), 2) - tp - fp - fn
        self.randindex = (tp + tn) / (tp + fp + fn + tn)
        return self.randindex
    
#%%
##%% Testing
#from sklearn.datasets import make_circles, make_moons, load_iris
#X, y = load_iris().data, load_iris().target
#X, y = make_moons(n_samples=samples, noise=.05)
#X, y = make_circles(n_samples = 1000, noise = .10, factor=.05)
#kkmeans = kkMeans(k = 32, kernel = 'rbf', gamma = 1).fit_predict(np.array(simscore))
#plt.scatter(X[:, 0], X[:, 1], c = kkmeans.clusters)

