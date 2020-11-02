# -*- coding: utf-8 -*-
import os
import numpy as np

# global strings
HAM_STR = "ham/"
SPAM_STR = "spam/"
TRAIN_STR = "train/"
TEST_STR = "test/"
STOPWORDS_DIR = "stopwords.txt"
folders = [HAM_STR, SPAM_STR]
# for PUNCTUATION remove ' from string.punctuation
PUNCTUATION = r"""!¡"#$%&()*+,-./:;<=>?¿@[\]^_`{|}~"""
# string.punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""


# build vocabulary, return voc along with train counts
def buildVocabulary(doIgnore: bool):
    ignore_dic = {}
    if doIgnore:
        path = STOPWORDS_DIR
        f = open(path, 'r')
        lines = f.read()
        words = lines.split()
        for word in words:
            ignore_dic[word] = 0
    voc = {}
    # for our model
    # [[file num in ham folder, file num in spam folder],
    # [word num in ham folder, word num in spam folder]]
    train_counts = [[0, 0], [0, 0]]
    for i in range(2):  # ham, spam
        path = TRAIN_STR + folders[i]
        files = os.listdir(path)
        train_counts[0][i] = len(files)
        for file in files:
            if os.path.isdir(file): continue
            f = open(path + file, 'r', encoding='utf-8', errors='ignore')  # UnicodeDecodeError: 2248.2004-09-23.GP.spam.txt
            line = f.readline()
            while line:
                line = line.strip().lower()
                tmp = line.translate(str.maketrans('', '', PUNCTUATION))  # remove punctuation
                words = tmp.split()
                train_counts[1][i] += len(words)
                for word in words:
                    if not word: continue
                    if (doIgnore and word in ignore_dic.keys()): continue
                    if word not in voc.keys():
                        voc[word] = [0, 0]
                    voc[word][i] += 1
                line = f.readline()
            f.close()
    return voc, train_counts


# test function for Naive Bayes
def naiveBayesTest(voc, train_counts):
    class_ps = [np.log(_/sum(train_counts[0])) for _ in train_counts[0]]  # calculate all p in log-scale to avoid underflow
    # for our model
    # [[ham counts in ham folder, spam counts in ham folder],
    # [ham counts in spam folder, spam counts in spam folder]]
    test_counts = [[0, 0], [0, 0]]
    for i in range(2):
        path = TEST_STR + folders[i]
        files = os.listdir(path)
        for file in files:
            if os.path.isdir(file): continue
            f = open(path + file, 'r', encoding='utf-8', errors='ignore')  # UnicodeDecodeError: 2248.2004-09-23.GP.spam.txt
            ps = class_ps.copy()
            line = f.readline()
            while line:
                line = line.strip().lower()
                tmp = line.translate(str.maketrans('', '', PUNCTUATION))  # remove punctuation
                words = tmp.split()
                for word in words:
                    if word not in voc.keys(): continue
                    for j in range(2):
                        ps[j] += np.log((voc[word][j] + 1)/(train_counts[1][j] + len(voc)))
                line = f.readline()
            if (ps[0] >= ps[1]):  # assume ham if equal
                test_counts[i][0] += 1
            else:
                test_counts[i][1] += 1
            f.close()
    return test_counts


# generate x and y vector
def generateXY(voc):
    x, y = [], []
    socket = {}
    for (i, v) in enumerate(voc.keys()):
        socket[v] = i
    xi0 = [0 for _ in range(len(socket))] + [1]  # I put x0 at the end of vector to for the consistency of indexes
    for i in range(2):  # ham, spam
        path = TRAIN_STR + folders[i]
        files = os.listdir(path)
        for file in files:
            xi = xi0.copy()
            if os.path.isdir(file): continue
            f = open(path + file, 'r', encoding='utf-8', errors='ignore')  # UnicodeDecodeError: 2248.2004-09-23.GP.spam.txt
            line = f.readline()
            while line:
                line = line.strip().lower()
                tmp = line.translate(str.maketrans('', '', PUNCTUATION))  # remove punctuation
                words = tmp.split()
                for word in words:
                    if word not in voc.keys(): continue
                    xi[socket[word]] += 1
                line = f.readline()
            x.append(xi)
            y.append(i)
            f.close()
    return x, y, socket


# train w vector
def trainW(x, y, w, hard_limit, lbd, lr):
    print("Progressing ", end='')
    for t in range(hard_limit):
        print("#", end='')
        new_w = w.copy()
        p = []
        for l in range(len(y)):
            sum1 = 0
            for i in range(len(w)):
                if x[l][i] == 0: continue
                sum1 += x[l][i] * w[i]
            if (sum1 > 709):  # prevent overflow
                p.append(1.0)
            else:
                e = np.exp(sum1)
                p.append(e / (1 + e))
        for i in range(len(w)):
            sum2 = 0
            for l in range(len(y)):
                sum2 += x[l][i] * (y[l] - p[l])
            new_w[i] += lr * (sum2 - lbd * w[i])
        w = new_w
    print(" -- 100%")
    return w


# test function for Logistic Regression
def logisticRegressionTest(voc, w, socket):
    # for our model
    # [[ham counts in ham folder, spam counts in ham folder],
    # [ham counts in spam folder, spam counts in spam folder]]
    test_counts = [[0, 0], [0, 0]]
    xi0 = [0 for _ in range(len(socket))] + [1]  # I put x0 at the end of vector to for the consistency of indexes
    for i in range(2):
        path = TEST_STR + folders[i]
        files = os.listdir(path)
        for file in files:
            if os.path.isdir(file): continue
            f = open(path + file, 'r', encoding='utf-8', errors='ignore')  # UnicodeDecodeError: 2248.2004-09-23.GP.spam.txt
            xi = xi0.copy()
            line = f.readline()
            while line:
                line = line.strip().lower()
                tmp = line.translate(str.maketrans('', '', PUNCTUATION))  # remove punctuation
                words = tmp.split()
                for word in words:
                    if word not in voc.keys(): continue
                    xi[socket[word]] += 1
                line = f.readline()
            sum = 0
            for j in range(len(xi)):
                sum += xi[j] * w[j]
            if (sum <= 0):  # assume ham if equal to 0
                test_counts[i][0] += 1
            else:
                test_counts[i][1] += 1
            f.close()
    return test_counts


print("Library Loaded.")
