# Assignment 2 b-e
# ---------
# Use decision_tree.py and other tools to finish Assignment 2 part b-e.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.model_selection import train_test_split
import decision_tree as dt


def get_error_ID3(trn_dir: str, tst_dir: str, max_d: int) -> (float, float):
    # Load the training data
    M = np.genfromtxt(trn_dir, missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt(tst_dir, missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    # Learn a decision tree with a max depth
    decision_tree = dt.id3(Xtrn, ytrn, max_depth=max_d)
    """
    # Pretty print it to console
    dt.pretty_print(decision_tree)

    # Visualize the tree and save it as a PNG image
    dot_str = dt.to_graphviz(decision_tree)
    dt.render_dot_file(dot_str, './my_learned_tree')
    """
    # Compute the test error
    y_pred = [dt.predict_example(x, decision_tree) for x in Xtst]
    tst_err = dt.compute_error(ytst, y_pred)

    # Compute the train error
    y_pred2 = [dt.predict_example(x, decision_tree) for x in Xtrn]
    trn_err = dt.compute_error(ytrn, y_pred2)

    #print("Max Depth = {0}  Train Error = {1:4.2f}%  Test Error = {2:4.2f}%.".format(max_d, trn_err * 100, tst_err * 100))
    return (trn_err, tst_err)


if __name__ == '__main__':
    # part b
    print("part b:")
    # For depth = 1, ..., 10, learn decision trees and compute the average training and test errors on each of the three
    # MONK’s problems. Make three plots, one for each of the MONK’s problem sets, plotting training and testing error
    # curves together for each problem, with tree depth on the x-axis and error on the y-axis.
    trn_sets = ['./monks-1.train', './monks-2.train', './monks-3.train']
    tst_sets = ['./monks-1.test', './monks-2.test', './monks-3.test']
    for i in range(len(trn_sets)):
        res = [[], []]
        for j in range(1, 11):
            error = get_error_ID3(trn_sets[i], tst_sets[i], j)
            res[0].append(error[0])# train
            res[1].append(error[1])# test
        plt.figure()
        plt.title("Analysis of Error in monks-{0}".format((i + 1)))
        plt.plot(np.arange(1, 11), res[0], marker='o', linewidth=3, markersize=12)
        plt.plot(np.arange(1, 11), res[1], marker='s', linewidth=3, markersize=12)
        plt.xlabel("Depth", fontsize=16)
        plt.ylabel("Training/Testing Error", fontsize=16)
        plt.xticks(np.arange(1, 11), fontsize=12)
        plt.legend(["Training Error", "Testing Error"], fontsize=16)
        plt.axis([0, 12, 0, 1])
        print("Problem: monks-{0}  Average Training Error = {1:4.2f}%  Average Testing Error = {2:4.2f}%.".format(
            (i + 1), np.average(res[0]) * 100, np.average(res[1]) * 100))

    # part c
    print("part c:")
    print("Problem: monks-1")
    # For monks-1, report the visualized learned decision tree and the confusion matrix on the test set for
    # depth = 1, 3, 5. You may use scikit-learns’s confusion matrix() function [2].
    # Load the training data
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    matrixes = {}
    print("Print Decision Trees:")
    for i in [1, 3, 5]:
        print("Max Depth = {0}:".format(i))
        # Learn a decision tree with ID3
        decision_tree = dt.id3(Xtrn, ytrn, max_depth=i)

        # Pretty print it to console
        dt.pretty_print(decision_tree)

        # Visualize the tree and save it as a PNG image
        dot_str = dt.to_graphviz(decision_tree)
        dt.render_dot_file(dot_str, './img/monks-1/my_learned_tree_depth{0}'.format(i))

        # Report the Confusion Matrix
        y_pred = [dt.predict_example(x, decision_tree) for x in Xtst]
        matrixes[i] = confusion_matrix(ytst, y_pred)

    print("Print Confusion Matrixes:")
    for i in [1, 3, 5]:
        print("Max Depth = {0}:".format(i))
        print(matrixes[i])

    # part d
    print("part d:")
    print("Problem: monks-1")
    # For monks-1, use scikit-learn’s DecisionTreeClassifier [3] to learn a decision tree using criterion=’entropy’ for
    # depth = 1, 3, 5. You may use scikit-learn’s confusion matrix() function [2].
    # Load the training data
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    matrixes2 = {}
    print("Print Decision Trees:")
    for i in [1, 3, 5]:
        print("Max Depth = {0}:".format(i))
        # Learn a decision tree with sklearn
        decision_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=i)
        decision_tree = decision_tree.fit(Xtrn, ytrn)

        # Visualize the tree and save it as a PNG image
        dot_str = tree.export_graphviz(decision_tree, out_file=None)
        dt.render_dot_file(dot_str, './img/monks-1/my_learned_tree2_depth{0}'.format(i))

        # Report the Confusion Matrix
        y_pred = decision_tree.predict(Xtst)
        matrixes2[i] = confusion_matrix(ytst, y_pred)

    print("Print Confusion Matrixes:")
    for i in [1, 3, 5]:
        print("Max Depth = {0}:".format(i))
        print(matrixes2[i])

    # part e
    print("part e:")
    print("Problem: Glass Identification")
    # Repeat steps (c) and (d) with your “own” data set and report the confusion matrices. You can use other data sets
    # in the UCI repository.
    # I use Glass Identification Data Set Here
    # Load the training data
    M_ori = np.genfromtxt('./glass.data', missing_values=0, skip_header=0, delimiter=',', dtype=float)

    # simple discretization
    m, n = M_ori.shape
    M = np.array([[int(M_ori[j][-1] + 0.5) for i in range(1, n)] for j in range(m)])
    for i in range(n - 2):
        sum = 0
        for j in range(m):
            sum += M_ori[j][i + 1]
        avg = sum / m
        for j in range(m):
            M[j][i] = 0 if M_ori[j][i + 1] <= avg else 1

    # split data set
    trn_M, tst_M = train_test_split(M, test_size = 0.4, random_state = 42)

    ytrn = trn_M[:, -1]
    Xtrn = trn_M[:, :-1]

    # Load the test data
    ytst = tst_M[:, -1]
    Xtst = tst_M[:, :-1]

    matrixes3 = {}
    print("Print Decision Trees:")
    for i in [1, 3, 5]:
        print("Max Depth = {0}:".format(i))
        # Learn a decision tree with ID3
        decision_tree = dt.id3(Xtrn, ytrn, max_depth=i)

        # Pretty print it to console
        dt.pretty_print(decision_tree)

        # Visualize the tree and save it as a PNG image
        dot_str = dt.to_graphviz(decision_tree)
        dt.render_dot_file(dot_str, './img/glass/my_learned_tree_depth{0}'.format(i))

        # Report the Confusion Matrix
        y_pred = [dt.predict_example(x, decision_tree) for x in Xtst]
        matrixes3[i] = confusion_matrix(ytst, y_pred)

    print("Print Confusion Matrixes:")
    for i in [1, 3, 5]:
        print("Max Depth = {0}:".format(i))
        print(matrixes3[i])

    matrixes4 = {}
    print("Print Decision Trees:")
    for i in [1, 3, 5]:
        print("Max Depth = {0}:".format(i))
        # Learn a decision tree with sklearn
        decision_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=i)
        decision_tree = decision_tree.fit(Xtrn, ytrn)

        # Visualize the tree and save it as a PNG image
        dot_str = tree.export_graphviz(decision_tree, out_file=None)
        dt.render_dot_file(dot_str, './img/glass/my_learned_tree2_depth{0}'.format(i))

        # Report the Confusion Matrix
        y_pred = decision_tree.predict(Xtst)
        matrixes4[i] = confusion_matrix(ytst, y_pred)

    print("Print Confusion Matrixes:")
    for i in [1, 3, 5]:
        print("Max Depth = {0}:".format(i))
        print(matrixes4[i])

    plt.show()