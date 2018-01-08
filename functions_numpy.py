# -*- coding: utf-8 -*-
"""
Packages: numpy
Task : Practice functions; sigmoid,first derivative, gradient descent etc
@author: Swetha
"""

import hashlib
import numpy as np
import matplotlib.pyplot as plt


# Compute sigmoid function

def sigmoid(z):
    return 1/(1+np.exp(-z))

np.random.seed(2)
n = 8
d = 4
A = np.random.randn(n,d)
b = np.round(np.random.rand(n))

# Calculate f(x) = (sigmoid(Ax)-b)^2

def f(x):
    return np.sum((sigmoid(A.dot(x))-b)**2)

x = np.zeros(d)

# Derivative of sigmoid function
def sigmoid_deriv(z):
    return 1./(np.exp(-z/2)+np.exp(z/2))**2
sigmoid_deriv(0)

x = np.random.randn(1)
delta = .0001
(sigmoid(x+delta)-sigmoid(x-delta))/(2*delta)

sigmoid_deriv(x)

x = np.array([0, 1, 2])
print sigmoid_deriv(x)

# Derivative of f(x)

def grad_f(x):
    e=(sigmoid(A.dot(x))-b)*sigmoid_deriv(A.dot(x))
    return 2*A.T.dot(e)

x0 = np.random.randn(d)
x0 = np.zeros(d)
delta=1e-6
r = np.random.randn(d)*delta
print (f(x0+r) - f(x0-r))/(2*delta)

print grad_f(x0).dot(r)/delta


def mysig(i):
    a=np.zeros(d)
    a[i]=1.
    return a
np.array([(f(x0+mysig(i)*delta)-f(x0-mysig(i)*delta))/(2*delta) for i in range(d)])
grad_f(x0)


n = 30
d = 10
A = np.random.randn(n,d)
b = np.round(np.random.rand(n))
print np.linalg.norm(A)

# Gradient Descent Function

def grad_descent(x0, f, g, tol=10**(-5),step_size=10,max_iters=10000):

    f0 = f(x0)  # initial value
    g0 = g(x0)  # initial gradient   
    iters=0
    while np.linalg.norm(g0)>tol and iters<max_iters:
        xnew = x0-step_size*g0
        fnew = f(xnew)
        if (fnew < f0):
            f0 = fnew   # the best function value
            x0 = xnew   # the corresponding vector
            g0 = g(x0)  # and, its gradient
        else:
            step_size = step_size/1.2          
    return x0



