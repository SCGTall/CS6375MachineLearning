# -*- coding: utf-8 -*-
import Assignment3 as a3

print("Start training...")
voc, train_counts = a3.buildVocabulary(True)  # ignore stop words
test_counts = a3.naiveBayesTest(voc, train_counts)
ham_accuracy = test_counts[0][0] / sum(test_counts[0])
spam_accuracy = test_counts[1][1] / sum(test_counts[1])
total_accuracy = (test_counts[0][0] + test_counts[1][1]) / (sum(test_counts[0]) + sum(test_counts[1]))
print("Accuracy of Naive Bayes (Ignore Stop Words):")
print("Ham: %.5f" % ham_accuracy)
print("Spam: %.5f" % spam_accuracy)
print("Total: %.5f" % total_accuracy)