# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# The true function
def f_true(x: float) -> float:
    return 6.0 * (np.sin(x + 2) + np.sin(2 * x + 4))

# Get data points
np.random.seed(1)                                          # This can make the result repeatable during debugging
n = 750                                                     # Number of data points
x = np.random.uniform(-7.5, 7.5, n)                         # Training examples, in one dimension
e = np.random.normal(0.0, 5.0, n)                           # Random Gaussian noise
y = f_true(x) + e

plt.figure()
plt.title('Plot the Data')
# Plot the data
plt.scatter(x, y, 12, marker = 'o')

# Plot the true function, which is really "unknown"
x_true = np.arange(-7.5, 7.5, 0.05)
y_true = f_true(x_true)
plt.plot(x_true, y_true, marker = 'None', color = 'r')

# scikit-learn has many tools and utilities for model selection
# Use the fraction in code regradless of document's words
tst_frac = 0.3                                              # Fraction of examples to sample for the test set
val_frac = 0.1                                              # Fraction of examples to sample for the validation set

# Use the same seed = 42
# First, we use train_test_split to partition (x, y) into training and test sets
x_trn, x_tst, y_trn, y_tst = train_test_split(x, y, test_size = tst_frac, random_state = 42)

# Next, we use train_test_split to further partition (x_trn, ytrn) into training and validation sets
x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size = val_frac, random_state = 42)

# Plot the three subsets
plt.figure()
plt.title('Plot Three Subsets')
plt.scatter(x_trn, y_trn, 12, marker = 'o', color = 'orange')
plt.scatter(x_val, y_val, 12, marker = 'o', color = 'green')
plt.scatter(x_tst, y_tst, 12, marker = 'o', color = 'blue')

# Polynomial Basis Functions
def polynomial_transform(x: np.array(float), d: int) -> np.array(float):
    # *** Insert your code here ***
    return np.array([np.logspace(0, d, d+1, base=i) for i in x])

def train_model(phi: np.array(float), y: np.array(float)) -> np.array(float):
    # *** Insert your code here ***
    return np.linalg.inv(phi.transpose().dot(phi)).dot(phi.transpose()).dot(y)

def evaluate_model(phi: np.array(float), y: np.array(float), w: np.array(float)) -> float:
    # *** Insert your code here ***
    return np.mean(np.power((y - np.array([w.dot(i) for i in phi])), 2))

w = {}                                                          # Dictionary to store all the trained models
validationErr = {}                                              # Validation error of the models
testErr = {}                                                    # Test error of all the models

for d in range(3, 25, 3):                                       # Iterate over polynomial degree
    phi_trn = polynomial_transform(x_trn, d)                    # Transform training data into d dimensions
    w[d] = train_model(phi_trn, y_trn)                          # Learn model on training data
    phi_val = polynomial_transform(x_val, d)                    # Transform validation data into d dimensions
    validationErr[d] = evaluate_model(phi_val, y_val, w[d])     # Evaluate model on validation data
    phi_tst = polynomial_transform(x_tst, d)                    # Transform test data into d dimensions
    testErr[d] = evaluate_model(phi_tst, y_tst, w[d])           # Evaluate model on test data

# Plot all the models
plt.figure()
plt.title('Analysis of d of Linear Basis Case')
plt.plot(list(validationErr.keys()), list(validationErr.values()), marker = 'o', linewidth = 3, markersize = 12)
plt.plot(list(testErr.keys()), list(testErr.values()), marker = 's', linewidth = 3, markersize = 12)
plt.xlabel("Ploynomial degree", fontsize = 16)
plt.ylabel("Validation/Test error", fontsize = 16)
plt.xticks(list(validationErr.keys()), fontsize = 12)
plt.legend(['Validation Error', 'Test Error'], fontsize = 16)
plt.axis([2, 25, 15, 65])

plt.figure()
plt.title('Visualize All Learned Models of Linear Basis Case')
plt.plot(x_true, y_true, marker = 'None', linewidth = 5, color = 'k')

for d in range(9, 25, 3):
    x_d = polynomial_transform(x_true, d)
    y_d = x_d @ w[d]
    plt.plot(x_true, y_d, marker = 'None', linewidth = 2)

plt.legend(['true'] + list(range(9, 25, 3)))
plt.axis([-8, 8, -15, 15])

# Radial Basis Functions
def radial_basis_transform(x: np.array(float), b: np.array(float), gamma: float = 0.1) -> np.array(float):
    # *** Insert your code here ***
    return np.array([[(np.e ** (-gamma * ((xi - xj) ** 2))) for xj in b] for xi in x])

def train_ridge_model(phi: np.array(float), y: np.array(float), lam: float) -> np.array(float):
    # *** Insert your code here ***
    return np.linalg.inv(phi.transpose().dot(phi) + lam * np.eye(len(phi))).dot(phi.transpose()).dot(y)

# *** Insert your code here ***
w2 = {}                                                         # Dictionary to store all the trained models for radial basis
validationErr2 = {}                                             # Validation error of the models for radial basis
testErr2 = {}                                                   # Test error of all the models for radial basis

phi_trn2 = radial_basis_transform(x_trn, x_trn)                 # Transform training data into n dimensions
phi_val2 = radial_basis_transform(x_val, x_trn)                 # Transform validation data into n dimensions
phi_tst2 = radial_basis_transform(x_tst, x_trn)                 # Transform test data into n dimensions

for l in range(-3, 4):                                          # Iterate over l
    lam = 10 ** l                                               # Get lambda from l
    w2[l] = train_ridge_model(phi_trn2, y_trn, lam)             # Learn model on training data
    validationErr2[l] = evaluate_model(phi_val2, y_val, w2[l])  # Evaluate model on validation data
    testErr2[l] = evaluate_model(phi_tst2, y_tst, w2[l])        # Evaluate model on test data

# Plot all the models
plt.figure()
plt.title('Analysis of lambda of Radial Basis Case')
plt.plot(list(validationErr2.keys()), list(validationErr2.values()), marker = 'o', linewidth = 3, markersize = 12)
plt.plot(list(testErr2.keys()), list(testErr2.values()), marker = 's', linewidth = 3, markersize = 12)
plt.xlabel("Radial degree", fontsize = 16)
plt.ylabel("Validation/Test error", fontsize = 16)
plt.xticks(list(validationErr2.keys()), [10 ** i for i in list(validationErr2.keys())], fontsize = 12)
plt.legend(['Validation Error', 'Test Error'], fontsize = 16)
plt.axis([-4, 4, 15, 65])

plt.figure()
plt.title('Visualize All Learned Models of Radial Basis Case')
plt.plot(x_true, y_true, marker = 'None', linewidth = 5, color = 'k')

for l in range(-3, 4):
    x_d = radial_basis_transform(x_true, x_trn)
    y_d = x_d @ w2[l]
    plt.plot(x_true, y_d, marker = 'None', linewidth = 2)

plt.legend(['true'] + list(np.logspace(-3, 3, 7)))
plt.axis([-8, 8, -15, 15])

# Wait to check results
plt.show()                                                      # Image will not show without this code