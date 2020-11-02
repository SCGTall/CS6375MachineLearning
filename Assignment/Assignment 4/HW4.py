# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

x_bad = [7, 7]
y_bad = [7, 4]
x_good = [3, 1]
y_good = [4, 4]
x_tst = [3]
y_tst = [7]

theta = np.arange(0, 2 * np.pi, 0.00001)
r = 4.5
x_cir = np.sin(theta) * r + x_tst[0]
y_cir = np.cos(theta) * r + y_tst[0]

plt.figure()
plt.scatter(x_bad, y_bad, marker='o', color='red', s=100)
plt.scatter(x_good, y_good, marker='o', color='blue', s=100)
plt.scatter(x_tst, y_tst, marker='o', color='green', s=100)
plt.scatter(x_cir, y_cir, marker='o', color='black', s=3)
plt.legend(["Bad", "Good", "Testing"], fontsize=16, loc='upper left')
plt.axis([0, 8, 3, 8])
plt.gca().set_aspect('equal', adjustable='box')
plt.show()