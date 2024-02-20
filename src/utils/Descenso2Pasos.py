# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 18:17:05 2020

@author: RAVJ

Gradient Descent in two steps. The idea is to fix one variable, optimizate and
use the solution to optimizate the other one.
"""
import numpy as np
from numpy import linalg as la
import time

import utils.PuntosInteriores as PI

def Jloss(X, W, H, lambd):
    """
    J(W, H) = 0.5*||Y - WH||^2_Frob + 0.5 * lambda (||W||^2_Frob + ||H||^2_Frob) 

    Parameters
    ----------
    X : Numpy array (sparse matrix)
        Matrix that wants to be approximated in a non negative factorization: W H
    W : Numpy array (sparse matrix)
        .
    H : Numpy array (sparse matrix)
        .
    lambd: non negative float
        Penalization of the size of the columns in W and H

    Returns
    -------
    Loss function.

    """
    #get index where are not na values
    ## array with the index
    indicator = np.argwhere(~ np.isnan(X))
    numcalificaciones=len(indicator)
    indirow = indicator[:,0]
    indicol = indicator[:,1]
    
    # ravel values of X and Z = WH
    Y = X[indirow, indicol]
    Z = W @ H
    Z = Z[indirow, indicol]
    
    # cost function
    J1 = la.norm(Y - Z)**2
    J2 = lambd * la.norm(W, ord='fro')**2
    J3 = lambd * la.norm(H, ord='fro')**2
    J4 = J1 + J2 + J3
    return [numcalificaciones, 0.5*J1, 0.5*J2, 0.5*J3, 0.5*J4]

def gradient_step(X, W, H, lambd, typeof=''):
    """
    One step of gradient descendent. 

    Parameters
    ----------
    X : Numpy array (sparse matrix)
         Matrix that wants to be approximated in a non negative factorization.
    Y : Numpy array 
        One of the two non negative matrix approximation of X
    typeof : string (column, row)
        Specify if we want to approximate X columns or X rows.

    Returns
    -------
    Optimized

    """
    # get values
    r, p = X.shape
    k = H.shape[0]
    
    if typeof == 'column':
        # optimize H with W fixed
        # min h' (W'W + lambd*I) h + (-Xj'W)'h  s.a. h > 0, forall h in col(H)
        for j in range(p):
            # print('\nOptimizando H{0}'.format(j))
            
            #  objective function
            Xj = X[:, j].copy().reshape((r, 1))
            Q = W.T @ W + lambd * np.eye(k)
            
            # take out missing values in Xj
            indicator = ~ np.isnan(Xj) # only complete values
            indicator = np.squeeze(indicator)
            # reduce dimensionality 
            Xj = Xj[indicator]
            W_subset = W[indicator]
            
            c = - np.dot(W_subset.T, Xj)
            A = np.eye(k)
            b = np.zeros((k, 1))
            
            # solve for Hj
            hj, *_ = PI.interior_points(Q, A, c, b)
            
            # update H
            H[:,j] = np.squeeze(hj)
        return H
            
    elif typeof == 'row':
        # optimize W with H fixed
        # min w (HH' + lambd*I) w' + (-HXj)'h  s.a. w > 0, forall w in row(W)
        for i in range(r):
            # print('\nOptimizando W{0}'.format(i))
            #  objective function
            Xi = X[i].copy().reshape((p, 1))
            Q = H @ H.T + lambd * np.eye(k)
            
            #take out missing values in Xi
            indicator = ~ np.isnan(Xi) # only complete values
            indicator = np.squeeze(indicator)
            # reduce dimensionality 
            Xi = Xi[indicator]
            H_subset = H[:, indicator] 
            
            c = - np.dot(H_subset, Xi)
            A = np.eye(k)
            b = np.zeros((k, 1))
            
            # solve for wi'
            wi, *_ = PI.interior_points(Q, A, c, b)
            
            # update W
            W[i] = np.squeeze(wi.T)
        return W
    


def gradient2steps(X, k, lambd, maxiter, tol):
    start_time = time.time()
    """
    Two steps gradient descent
    problem: 
        min 0.5*||Y - WH||^2_Frob + 0.5 * lambda (||W||^2_Frob + ||H||^2_Frob)  
        s.t. Wij, Hij >= 0

    Parameters
    ----------
    X : Numpy array (sparse matrix)
        Matrix that wants to be approximated in a non negative factorization: W H
    k : int
        Size of the range of Z = WH.
    lambd: non negative float
        Penalization of the size of the columns in W and H
        
    maxiter : int
        Number of gradient descendant iterations
    tol : float
        Tolarance in aproximation. Recomended a not too low number.

    Returns
    -------
    A tuple with the non negative factorization of the matrixes

    """
    # init values
    r, p = X.shape
    W = np.random.random((r, k)) * 5
    H = np.random.random((k, p)) * 5
    Jotas=[1e30]
    iteration = 0
    J = Jloss(X, W, H, lambd) 
    Jotas.append(J[4])
    
    # two step gradient
    while (Jotas[-2]/J[0]-Jotas[-1]/J[0]) > tol and iteration < maxiter:
        print(f'\nIteración-{iteration}. \n  Costo Medio: {(J[4]/J[0]).round(3)}\n  Costo de Diferencias: {(J[1]).round(3)} \n  Costo de U: {(J[2]).round(3)} \n  Costo de V: {(J[3]).round(3)} \n  Costo Total: {(J[4]).round(3)} \nHan transcurrido {np.round(time.time()-start_time, 3)} segundos')
        
        # first step: optimize with respect columns of X
        print('En columnas')
        H = gradient_step(X, W, H, lambd, typeof='column')
        
        # second step: optimize with respect of rows of X
        print('En renglones \n')
        W = gradient_step(X, W, H, lambd, typeof='row')
        
        # approximation
        J = Jloss(X, W, H, lambd)
        iteration += 1
    
    if iteration == maxiter:
        print('Algoritmo de descenso en 2 pasos no convirgió')
    else:
        print(f'\nEl método convirgió en {iteration}-iteraciones' +
              f'\nIteración-{iteration}. \n  Costo Medio: {(J[4]/J[0]).round(3)}\n  Costo de Diferencias: {(J[1]).round(3)} \n  Costo de U: {(J[2]).round(3)} \n  Costo de V: {(J[3]).round(3)} \n  Costo Total: {(J[4]).round(3)} \nHan transcurrido {np.round(time.time()-start_time, 3)} segundos')
              
    return(W, H, time.time()-start_time, J[4])

























