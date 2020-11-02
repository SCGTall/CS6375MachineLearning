# -*- coding: utf-8 -*-
import Assignment3 as a3

print("Start training...")
voc, train_counts = a3.buildVocabulary(True)  # ignore stop words
x, y, socket = a3.generateXY(voc)
hard_limit = 200
lbds = [0.01, 0.02, 0.05, 0.1]  # 0.01-0.1
lr = 0.01
w0 = [1.0 for _ in range(len(voc) + 1)]
for lbd in lbds:
    w = a3.trainW(x, y, w0, hard_limit, lbd, lr)
    test_counts = a3.logisticRegressionTest(voc, w, socket)
    ham_accuracy = test_counts[0][0] / sum(test_counts[0])
    spam_accuracy = test_counts[1][1] / sum(test_counts[1])
    total_accuracy = (test_counts[0][0] + test_counts[1][1]) / (sum(test_counts[0]) + sum(test_counts[1]))
    print("Accuracy of Logistic Regression with lambda = %.2f (Ignore Stop Words):" % lbd)
    print("Ham: %.5f" % ham_accuracy)
    print("Spam: %.5f" % spam_accuracy)
    print("Total: %.5f" % total_accuracy)
