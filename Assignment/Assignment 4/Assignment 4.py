# -*- coding: utf-8 -*-
# 1. Support Vector Machines with Synthetic Data

# DO NOT EDIT THIS FUNCTION; IF YOU WANT TO PLAY AROUND WITH DATA GENERATION,
# MAKE A COPY OF THIS FUNCTION AND THEN EDIT
#
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def generate_data(n_samples, tst_frac=0.2, val_frac=0.2):
    # Generate a non-linear data set
    X, y = make_moons(n_samples=n_samples, noise=0.25, random_state=42)

    # Take a small subset of the data and make it VERY noisy; that is, generate outliers
    m = 30
    np.random.seed(30)  # Deliberately use a different seed
    ind = np.random.permutation(n_samples)[:m]
    X[ind, :] += np.random.multivariate_normal([0, 0], np.eye(2), (m,))
    y[ind] = 1 - y[ind]

    # Plot this data
    cmap = ListedColormap(['#b30065', '#178000'])
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='k')

    # First, we use train_test_split to partition (X, y) into training and test sets
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=tst_frac,
                                                  random_state=42)

    # Next, we use train_test_split to further partition (X_trn, y_trn) into training and validation sets
    X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn, test_size=val_frac,
                                                  random_state=42)

    return (X_trn, y_trn), (X_val, y_val), (X_tst, y_tst)


#
#  DO NOT EDIT THIS FUNCTION; IF YOU WANT TO PLAY AROUND WITH VISUALIZATION,
#  MAKE A COPY OF THIS FUNCTION AND THEN EDIT
#

def visualize(models, param, X, y):
    # Initialize plotting
    if len(models) % 3 == 0:
        nrows = len(models) // 3
    else:
        nrows = len(models) // 3 + 1

    fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(15, 5.0 * nrows))
    cmap = ListedColormap(['#b30065', '#178000'])

    # Create a mesh
    xMin, xMax = X[:, 0].min() - 1, X[:, 0].max() + 1
    yMin, yMax = X[:, 1].min() - 1, X[:, 1].max() + 1
    xMesh, yMesh = np.meshgrid(np.arange(xMin, xMax, 0.01),
                               np.arange(yMin, yMax, 0.01))

    for i, (p, clf) in enumerate(models.items()):
        # if i > 0:
        #   break
        r, c = np.divmod(i, 3)
        ax = axes[r, c]

        # Plot contours
        zMesh = clf.decision_function(np.c_[xMesh.ravel(), yMesh.ravel()])
        zMesh = zMesh.reshape(xMesh.shape)
        ax.contourf(xMesh, yMesh, zMesh, cmap=plt.cm.PiYG, alpha=0.6)

        if (param == 'C' and p > 0.0) or (param == 'gamma'):
            ax.contour(xMesh, yMesh, zMesh, colors='k', levels=[-1, 0, 1],
                       alpha=0.5, linestyles=['--', '-', '--'])

        # Plot data
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='k')
        ax.set_title('{0} = {1}'.format(param, p))


SAVE_IMG = True
SHOW_IMG = False
SHOW_MAT_TRANS = True
# Generate the data
n_samples = 300    # Total size of data set
(X_trn, y_trn), (X_val, y_val), (X_tst, y_tst) = generate_data(n_samples)

# a. The effect of the regularization parameter
print("Part 1.a")
# Learn support vector classifiers with a radial-basis function kernel with
# fixed gamma = 1 / (n_features * X.std()) and different values of C
C_range = np.arange(-3.0, 6.0, 1.0)
C_values = np.power(10.0, C_range)

models = dict()
trnErr = dict()
valErr = dict()

# Learn with different C
for C in C_values:
    clf = SVC(C=C, gamma='scale')
    clf.fit(X_trn, y_trn)
    models[C] = clf
    trnErr[C] = 1 - models[C].score(X_trn, y_trn)  # error = 1 - score
    valErr[C] = 1 - models[C].score(X_val, y_val)

if SAVE_IMG:
    plt.savefig('./img/Figure_1.png')
visualize(models, 'C', X_trn, y_trn)
if SAVE_IMG:
    plt.savefig('./img/Figure_2.png')
# Discuss error with C
plt.figure()
plt.title("How Error change with C in SVM")
plt.plot(C_range, [trnErr[c] for c in C_values], marker='o', linewidth=3, markersize=9)
plt.plot(C_range, [valErr[c] for c in C_values], marker='o', linewidth=3, markersize=9)
plt.xlabel("C", fontsize=16)
plt.ylabel("Error", fontsize=16)
plt.xticks(C_range, [("1E" + str(round(c))) for c in C_range], fontsize=12)
plt.axis([min(C_range) - 1, max(C_range) + 1, -0.1, 1])
plt.legend(["Training Error", "Validation Error"], fontsize=16)

# Select the "best" with minimum validation error
C_best, min_err = 0, 1
for (k, v) in valErr.items():
    if (v < min_err):
        min_err = v
        C_best = k
tstAccuracy = models[C_best].score(X_tst, y_tst)
print("Among all C values:")
print("Best C: {0:.5g}, Test Accuracy: {1:.4f}.\n".format(C_best, tstAccuracy))
plt.plot([np.log10(C_best)], [min_err], marker='X', color='red', linewidth=3, markersize=16)
if SAVE_IMG:
    plt.savefig('./img/Figure_3.png')


# b. The effect of the BRF kernel parameter
print("Part 1.b")
# Learn support vector classifiers with a radial-basis function kernel with
# fixed C = 10.0 and different values of gamma
gamma_range = np.arange(-2.0, 4.0, 1.0)
gamma_values = np.power(10.0, gamma_range)

models = dict()
trnErr = dict()
valErr = dict()

# Learn with different gamma
for G in gamma_values:
    clf = SVC(C=10, gamma=G)
    clf.fit(X_trn, y_trn)
    models[G] = clf
    trnErr[G] = 1 - models[G].score(X_trn, y_trn)  # error = 1 - score
    valErr[G] = 1 - models[G].score(X_val, y_val)

visualize(models, 'gamma', X_trn, y_trn)
if SAVE_IMG:
    plt.savefig('./img/Figure_4.png')
# Discuss error with gamma
plt.figure()
plt.title("How Error change with gamma in SVM")
plt.plot(gamma_range, [trnErr[g] for g in gamma_values], marker='o', linewidth=3, markersize=9)
plt.plot(gamma_range, [valErr[g] for g in gamma_values], marker='o', linewidth=3, markersize=9)
plt.xlabel("gamma", fontsize=16)
plt.ylabel("Error", fontsize=16)
plt.xticks(gamma_range, [("1E" + str(round(g))) for g in gamma_range], fontsize=12)
plt.axis([min(gamma_range) - 1, max(gamma_range) + 1, -0.1, 1])
plt.legend(["Training Error", "Validation Error"], fontsize=16)

# Select the "best" with minimum validation error
gamma_best, min_err = 0, 1
for (k, v) in valErr.items():
    if (v < min_err):
        min_err = v
        gamma_best = k
tstAccuracy = models[gamma_best].score(X_tst, y_tst)
print("Among all gamma values:")
print("Best gamma: {0:.5g}, Test Accuracy: {1:.4f}.\n".format(gamma_best, tstAccuracy))
plt.plot([np.log10(gamma_best)], [min_err], marker='X', color='red', linewidth=3, markersize=16)
if SAVE_IMG:
    plt.savefig('./img/Figure_5.png')


# 2. Breast Cancer Diagnosis with Support Vector Machines
print("Part 2")
# Load the Breast Cancer Diagnosis data set; download the files from eLearning
# CSV files can be read easily using np.loadtxt()
trn = np.loadtxt("wdbc_trn.csv", delimiter=',')
X_trn = trn[:, 1:]
y_trn = trn[:, 0]
val = np.loadtxt("wdbc_val.csv", delimiter=',')
X_val = val[:, 1:]
y_val = val[:, 0]
tst = np.loadtxt("wdbc_tst.csv", delimiter=',')
X_tst = tst[:, 1:]
y_tst = tst[:, 0]

C_range = np.arange(-2.0, 5.0, 1.0)
C_values = np.power(10.0, C_range)
gamma_range = np.arange(-3.0, 3.0, 1.0)
gamma_values = np.power(10.0, gamma_range)

models = dict()
trnMat = [["1E{0:d}".format(round(g)) for g in gamma_range]]
trnMat[0].insert(0, "gamma =")
valMat = [["1E{0:d}".format(round(g)) for g in gamma_range]]
valMat[0].insert(0, "gamma =")
# remember all min_err place and use median matrix to find the "best" C, gamma pair.
mem = [[0 for g in gamma_values] for c in C_values]
dic = {}

min_err = 1

# Learn with different C and gamma and select the "best"
for (i, c) in enumerate(C_values):
    trnMat.append(["C = 1E{0:d}".format(round(np.log10(c)))])
    valMat.append(["C = 1E{0:d}".format(round(np.log10(c)))])
    for (j, g) in enumerate(gamma_values):
        clf = SVC(C=c, gamma=g)
        clf.fit(X_trn, y_trn)
        models[(c, g)] = clf
        trnMat[-1].append("{0:4f}".format(1 - models[(c, g)].score(X_trn, y_trn)))
        tmp = 1 - models[(c, g)].score(X_val, y_val)
        if (abs(tmp - min_err) < 0.0000001):  # float equal
            mem[i][j] = 1
            dic[(i, j)] = (c, g)
        elif (tmp < min_err):
            min_err = tmp
            mem = [[0 for g in gamma_values] for c in C_values]
            mem[i][j] = 1
            dic.clear()
            dic[(i, j)] = (c, g)
        valMat[-1].append("{0:4f}".format(tmp))


# Used to find best
def matrix_median(m, i, j):
    tmp = [m[i][j]]
    if (i - 1 >= 0):
        tmp.append(m[i - 1][j])
    else:
        tmp.append(0)
    if (i + 1 < len(m)):
        tmp.append(m[i + 1][j])
    else:
        tmp.append(0)
    if (j - 1 >= 0):
        tmp.append(m[i][j - 1])
    else:
        tmp.append(0)
    if (j + 1 < len(m[0])):
        tmp.append(m[i][j + 1])
    else:
        tmp.append(0)
    tmp.sort()
    return tmp[2]


def blur_matrix(m):
    res = []
    for i in range(len(m)):
        row = []
        for j in range(len(m[0])):
            row.append(matrix_median(m, i, j))
        res.append(row)
    return res


def find_first(m, val):
    for i in range(len(m)):
        for j in range(len(m[0])):
            if (m[i][j] == val): return (i, j)
    return None


# use matrix median to find "best"
def find_best_C_and_gamma(mem):
    if not find_first(mem, 0):  # every mem will come with at least one 1
        return find_first(mem, 1)
    old_m = mem
    if SHOW_MAT_TRANS:
        for line in old_m:
            print(line)
    while (True):  # if all 1, blur will not erase any 1
        new_m = blur_matrix(old_m)
        if SHOW_MAT_TRANS:
            print("Blur to:")
            for line in new_m:
                print(line)
        if not find_first(new_m, 1):
            (i, j) = find_first(old_m, 1)
            if SHOW_MAT_TRANS:
                print("({0}, {1})".format(i, j))
            return (i, j)
        else:
            old_m = new_m


i, j = find_best_C_and_gamma(mem)
C_best, gamma_best = dic[(i, j)]


# Used to print matrix
def print_C_gamma_matrix(mat, selected_i, selected_j, show_best):
    cell_width = 20
    if len(mat) == 0: return
    for i in range(len(mat)):
        s = ""
        for j in range(len(mat[i])):
            if (show_best and i == selected_i + 1 and j == selected_j + 1):  # 1 for offset
                s += ("(" + mat[i][j] + ")<-best").ljust(cell_width, ' ')
            else:
                s += mat[i][j].ljust(cell_width, ' ')
        print(s)


print("Training Errors:")
print_C_gamma_matrix(trnMat, np.where(C_values == C_best)[0], np.where(gamma_values == gamma_best)[0], False)
print("Validation Errors:")
print_C_gamma_matrix(valMat, np.where(C_values == C_best)[0], np.where(gamma_values == gamma_best)[0], True)


tstAccuracy = models[(C_best, gamma_best)].score(X_tst, y_tst)
print("Among all C and gamma value combinations:")
print("Best C: {0:.5g}, Best gamma: {1:.5g}, Test Accuracy: {2:.4f}.\n".format(C_best, gamma_best, tstAccuracy))


# 3. Breast Cancer Diagnosis with k-Nearest Neighbors
print("Part 3")

k_values = [1, 5, 11, 15, 21]
# Learn with different k and select the "best"
models = dict()
trnErr = []
valErr = []
k_best, min_err = 0, 1

# Learn with different k
for k in k_values:
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_trn, y_trn)
    models[k] = clf
    trnErr.append(1 - models[k].score(X_trn, y_trn))  # error = 1 - score
    valErr.append(1 - models[k].score(X_val, y_val))
    if (valErr[-1] < min_err):
        min_err = valErr[-1]
        k_best = k


# Discuss error with k
plt.figure()
plt.title("How Error change with k in KNN")
plt.plot(k_values, trnErr, marker='o', linewidth=3, markersize=9)
plt.plot(k_values, valErr, marker='o', linewidth=3, markersize=9)
plt.xlabel("k", fontsize=16)
plt.ylabel("Error", fontsize=16)
plt.axis([min(k_values) - 1, max(k_values) + 1, -0.1, 1])
plt.legend(["Training Error", "Validation Error"], fontsize=16)

# Select the "best" with minimum validation error
tstAccuracy = models[k_best].score(X_tst, y_tst)
print("Among all k values:")
print("Best k: {0:n}, Test Accuracy: {1:.4f}.".format(k_best, tstAccuracy))
plt.plot([k_best], [min_err], marker='X', color='red', linewidth=3, markersize=16)
if SAVE_IMG:
    plt.savefig('./img/Figure_6.png')


if SHOW_IMG:
    plt.show()
