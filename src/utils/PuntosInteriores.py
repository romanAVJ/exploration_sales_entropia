# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 07:07:48 2020

@author: RAVJ

Interior Points Method in cuadratic programing. 
Karsuh Kun Tucker and perturbation method. 
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la

def interior_points(Q, A, c, b, plot=False):
    """
    Interior point method to solve quadratic programing with restrictions:
        min 0.5 x'Q x + c'x
        s.t. Ax >= b
        
    The idea was taken from Nocedal, chapter 16, second edition.
    
    Roman Alberto Velez Jimenez
    Dec 1, 2020
    
    Parameters
    ----------
    Q : Numpy array (matrix) nxn
        Symmetric positive definite. 
    A : Numpy array (matrix) mxn
        Linear restrictions of x with range r(A) = m.
    c : Numpy array (vector). R^n
        Linear cost
    b : Numpy array (vector). R^m
        Restrictions in Ax

    Returns
    -------
    x : numpy array (vector). R^n
        Linear aproximation of the global minima. i.e. x .= x*
    y : numpy array (vector). R^m
        Slack variable of the restriction Ax - y = b
    mu : numpy array (vector). R^m
        Aproximation to the lagrange multiplier.

    """
    # Inicial parameters
    tol = 1e-6
    maxiter = 250
    iteration = 0
    
    # dimensions
    n = len(c) # dimension of x variable
    m = len(b) # number of inequality restrictions
    
    # inicial parameters
    x = np.ones((n, 1))
    mu = np.ones((m, 1))
    y = np.ones((m, 1))
    gamma = 0.5 * np.dot(mu.T, y) / m #perturbation update
    gamma = np.squeeze(gamma) # make scalar
    e = np.ones((m, 1))
    At = np.transpose(A)
    
    # First Order Necesary Conditions
    fonc = []
    comp = []
    # Perturbated KKT conditions
    H = np.vstack((
                np.dot(Q, x) - np.dot(At, mu) + c,
                np.dot(A, x) - y - b,
                mu * y # Hadamard product
                  ))
    normH = la.norm(H) 
    
    while normH > tol and iteration < maxiter:
        # solve Newton method 
        YminU = (1/y) * mu
        YminU = np.diag(np.squeeze(YminU))
        
        rx = np.dot(Q, x) - np.dot(At, mu) + c
        ry = np.dot(A, x) - y - b
        rmu = mu * y - gamma * e
        
        # solve linear sistem (KKT matrix)
        K = Q + At @ YminU @ A
        ld = -(rx + At @ YminU @ ry + np.dot(At, (1/y) * rmu))
        Dx = la.solve(K, ld)
        
        # substitute for Dy and Dmu
        Dy = np.dot(A, Dx) + ry
        Dmu = -(1/y) * (mu * Dy + rmu)
        
        # cut gradient
        bt = np.where(Dmu < 0, -(mu / Dmu), 1)
        gm = np.where(Dy < 0, -(y / Dy), 1)
        
        alpha = np.min(np.vstack((bt, gm)))
        alpha = 0.9995 * min((1, alpha))
        
        # update parameters
        x += alpha * Dx
        mu += alpha * Dmu
        y += alpha * Dy
        
        gamma = 0.5 * np.dot(mu.T, y) / m # mean of complementarity
        gamma = np.squeeze(gamma)
        # First Order Necesary Conditions
        H = np.vstack((
                np.dot(Q, x) - np.dot(At, mu) + c,
                np.dot(A, x) - y - b,
                mu * y # Hadamard product
                  ))
        normH = la.norm(H)
        
        # next iteration
        iteration += 1
        fonc.append(normH)
        comp.append(2 * gamma)
    
    if plot:
        # graph convergence of interior points
        kiters = list(range(1, iteration + 1))
    
        plt.plot(kiters, fonc, color='darkgreen', label='Condiciones 1er Orden')
        plt.plot(kiters, comp, color='red', label='Complementaridad')
        plt.legend()
        plt.show()
    
    if iteration == maxiter:
        print('\nEl metodo de puntos interiores no covirgió')
    # else:
    #     print('\nEl método convirgió en {0}-iteraciones'.format(iteration) +
    #           '\nLa aproximacion al paso de newton es: {:f}'.format(normH) +
    #           '\nLa complementariedad final es: {:f}'.format(2*gamma)
    #           )
    
    return (x, y, mu)


