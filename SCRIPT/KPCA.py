#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 12:19:22 2019

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
            gamma = 0.1
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
            gamma = 0.1
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
    

class kPCA(Kernels):
    def __init__(self, k = None, kernel = None):
        super().__init__()
        if not k:
            k = 2
            self.k = k
        else:
            self.k = k
        if not kernel:
            kernel = 'linear'
            self.kernel = kernel
        else:
            self.kernel = kernel
        return
    
    def explained_variance_(self):
        '''
        :Return: explained variance.
        '''
        self.total_eigenvalue = np.sum(self.eival)
        self.explained_variance = [x/self.total_eigenvalue*100 for x in sorted(self.eival, reverse = True)[:self.k]]
        return self.explained_variance
    
    def kernelize(self, x1, x2):
        '''
        :params: x1: NxD
        :params: x2: NxD
        '''
        if self.kernel == 'linear':
            return Kernels.linear(x1, x2)
        elif self.kernel == 'rbf':
            return Kernels.rbf(x1, x2)
        elif self.kernel == 'sigmoid':
            return Kernels.sigmoid(x1, x2)
        elif self.kernel == 'polynomial':
            return Kernels.polynomial(x1, x2)
        elif self.kernel == 'cosine':
            return Kernels.cosine(x1, x2)
        elif self.kernel == 'correlation':
            return Kernels.correlation(x1, x2)
        elif self.kernel == 'linrbf':
            return Kernels.linrbf(x1, x2)
        elif self.kernel == 'rbfpoly':
            return Kernels.rbfpoly(x1, x2)
        elif self.kernel == 'rbfcosine':
            return Kernels.rbfpoly(x1, x2)
        elif self.kernel == 'etakernel':
            return Kernels.etakernel(x1, x2)
        elif self.kernel == 'alignment':
            return Kernels.alignment(x1, x2)
        elif self.kernel == 'laplace':
            return Kernels.laplacian(x1, x2)
        elif self.kernel == 'locguass':
            return Kernels.locguass(x1, x2)
        elif self.kernel == 'chi':
            return Kernels.chi(x1)
        
    def fit(self, X):
        '''
        param: X: NxD
        '''
        self.X = X
        #normalized kernel
        N_N = 1/self.X.shape[0]*np.ones((self.X.shape[0],self.X.shape[0]))
        self.normKernel = self.kernelize(X, X) - N_N.dot(self.kernelize(X, X)) - self.kernelize(X, X).dot(N_N) + N_N.dot(self.kernelize(X, X).dot(N_N))
        self.eival, self.eivect = np.linalg.eig(self.normKernel)
        self.eival, self.eivect = self.eival.real, self.eivect.real
        #sort eigen values and return explained variance
        self.sorted_eigen = np.argsort(self.eival[:self.k])[::-1]
        self.explained_variance = self.explained_variance_()
        #return eigen value and corresponding eigenvectors
        self.eival, self.eivect = self.eival[:self.k], self.eivect[:, self.sorted_eigen]
        self.components_ = self.eivect.T
        return self
        
    def fit_transform(self):
        '''
        Return: transformed data
        '''
        return self.normKernel.dot(self.eivect)
    
    def inverse_transform(self):
        '''
        :Return the inverse of input data
        '''
        self.transformed = self.normKernel.dot(self.eivect)
        return self.transformed.dot(self.components_)
    
    
#%%
#pca = kPCA(k = 32, kernel = 'linear').fit(np.array(simscore))
#pca = pca.components_.T

