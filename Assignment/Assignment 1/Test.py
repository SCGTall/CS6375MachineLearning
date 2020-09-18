# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# The true function
def f_true(x: float) -> float:
    return 6.0 * (np.sin(x + 2) + np.sin(2 * x + 4))

# Get data points
np.random.seed(1)                                           # This can make the result repeatable
n = 750                                                     # Number of data points
x = np.random.uniform(-7.5, 7.5, n)                         # Training examples, in one dimension
e = np.random.normal(0.0, 5.0, n)                           # Random Gaussian noise
y = f_true(x) + e

# Plot the true function, which is really "unknown"
x_true = np.arange(-7.5, 7.5, 0.05)
y_true = f_true(x_true)

# scikit-learn has many tools and utilities for model selection
# Use the fraction in code regradless of document's words
tst_frac = 0.3                                              # Fraction of examples to sample for the test set
val_frac = 0.1                                              # Fraction of examples to sample for the validation set

# Use the same seed = 42
# First, we use train_test_split to partition (x, y) into training and test sets
x_trn, x_tst, y_trn, y_tst = train_test_split(x, y, test_size = tst_frac, random_state = 42)

# Next, we use train_test_split to further partition (x_trn, ytrn) into training and validation sets
x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size = val_frac, random_state = 42)

# Polynomial Basis Functions 1
def polynomial_transform(x: np.array(float), d: int) -> np.mat(float):
    # *** Insert your code here ***
    return np.mat([np.logspace(0, d, d+1, base=i) for i in x])

def train_model(phi: np.mat(float), y: np.array(float)) -> np.mat(float):
    # *** Insert your code here ***
    return (phi.T * phi).I * phi.T * np.mat(y).T

def evaluate_model(phi: np.mat(float), y: np.array(float), w: np.mat(float)) -> float:
    # *** Insert your code here ***
    return np.mean(np.power((y - w.T * phi.T), 2))

# Polynomial Basis Functions 2
def polynomial_transform2(x: np.array(float), d: int) -> np.array(float):
    # *** Insert your code here ***
    return np.array([np.logspace(0, d, d+1, base=i) for i in x])

def train_model2(phi: np.array(float), y: np.array(float)) -> np.array(float):
    # *** Insert your code here ***
    return np.linalg.inv(phi.transpose().dot(phi)).dot(phi.transpose()).dot(y)

def evaluate_model2(phi: np.array(float), y: np.array(float), w: np.array(float)) -> float:
    # *** Insert your code here ***
    return np.mean(np.power((y - np.array([w.dot(i) for i in phi])), 2))

w = {}                                                          # Dictionary to store all the trained models
validationErr = {}                                              # Validation error of the models
testErr = {}                                                    # Test error of all the models
w2 = {}                                                         # Dictionary to store all the trained models
validationErr2 = {}                                             # Validation error of the models
testErr2 = {}                                                   # Test error of all the models

for d in range(3, 25, 3):                                       # Iterate over polynomial degree
    phi_trn = polynomial_transform(x_trn, d)                    # Transform training data into d dimensions
    phi_trn2 = polynomial_transform2(x_trn, d)
    w[d] = train_model(phi_trn, y_trn)                          # Learn model on training data
    w2[d] = train_model2(phi_trn2, y_trn)
    phi_val = polynomial_transform(x_val, d)                    # Transform validation data into d dimensions
    phi_val2 = polynomial_transform2(x_val, d)
    validationErr[d] = evaluate_model(phi_val, y_val, w[d])     # Evaluate model on validation data
    validationErr2[d] = evaluate_model2(phi_val2, y_val, w2[d])
    phi_tst = polynomial_transform(x_tst, d)                    # Transform test data into d dimensions
    phi_tst2 = polynomial_transform2(x_tst, d)
    testErr[d] = evaluate_model(phi_tst, y_tst, w[d])           # Evaluate model on test data
    testErr2[d] = evaluate_model2(phi_tst2, y_tst, w2[d])

# Plot all the models
plt.figure()
plt.plot(list(validationErr.keys()), list(validationErr.values()), marker = 'o', linewidth = 3, markersize = 12)
plt.plot(list(testErr.keys()), list(testErr.values()), marker = 's', linewidth = 3, markersize = 12)
plt.plot(list(validationErr2.keys()), list(validationErr2.values()), marker = 'o', linewidth = 3, markersize = 6)
plt.plot(list(testErr2.keys()), list(testErr2.values()), marker = 's', linewidth = 3, markersize = 6)
plt.xlabel("Ploynomial degree", fontsize = 16)
plt.ylabel("Validation/Test error", fontsize = 16)
plt.xticks(list(validationErr.keys()), fontsize = 12)
plt.legend(['Validation Error', 'Test Error', 'Validation Error2', 'Test Error2'], fontsize = 16)
plt.axis([2, 25, 15, 70])

print(w[21].T)
print(w2[21])
print(w[24].T)
print(w2[24])

# Wait to check results
plt.show()                                                      # Image will not show without this code