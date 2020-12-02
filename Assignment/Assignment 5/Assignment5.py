# -*- coding: utf-8 -*-
from matplotlib import pyplot as io
import numpy as np
from PIL import Image
import time
import random

random.seed(42)


def k_means_clustering(inp, k, outp_dir):
    if (k < 2):
        print("k = %d is too small." % k)
        return
    (m, n, l) = inp.shape
    print("\nk = %d" % k)
    # use k-means++ algorithm to optimize the beginning centers
    centers = find_centers(inp, k)
    print("Start with centers:")
    for c in centers:
        print(c)
    mat = np.zeros([m, n], dtype=int)
    # clustering
    times = 0
    print("Processing: ", end='')
    start_time = time.time()
    flag = True
    while flag:
        flag = alternate(inp, mat, centers)
        times += 1
        print("\rProcessing: %d times " % times + "*" * times, end='')
    end_time = time.time()
    print(" 100%")
    sec_cost = end_time - start_time
    min, sec = divmod(sec_cost, 60)
    print("Time cost: {0:.0f} min {1:.3f} sec,".format(min, sec))
    return regenerate_img(mat, centers, outp_dir)


def find_centers(inp, k):
    (m, n, l) = inp.shape
    # randomly find first center
    mi = random.randint(0, m - 1)
    ni = random.randint(0, n - 1)
    centers = [list(inp[mi, ni])]
    while (len(centers) < k):
        # find next center which has the maximum closest distance from existing centers.
        centers = add_next_center(inp, centers)
    return centers


def add_next_center(inp, centers):
    (m, n, l) = inp.shape
    dist2 = 0
    m_next, n_next = 0, 0
    for i in range(m):
        for j in range(n):
            d2 = 255 * 255 * 3
            for c in centers:
                dc2 = get_distance(inp[i, j], c)
                if (dc2 < d2):
                    d2 = dc2

            if (d2 > dist2):
                dist2 = d2
                m_next, n_next = i, j

    centers.append(list(inp[m_next, n_next]))
    return centers


def get_distance(list_a, list_b):
    # distance ** 2 = (r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2
    if (len(list_a) != len(list_b)): return None
    distance2 = 0
    for i in range(len(list_a)):
        if (list_a[i] >= list_b[i]):  # for uint8, we need to ensure the sign to avoid overflow
            distance2 += (list_a[i] - list_b[i]) ** 2
        else:
            distance2 += (list_b[i] - list_a[i]) ** 2
    return distance2


def alternate(inp, mat, centers):
    flag = assign_data(inp, mat, centers)  # flag -> True : have new assigned data point
    change_center(inp, mat, centers)
    return flag


def assign_data(inp, mat, centers):
    flag = False
    (m, n) = mat.shape
    for i in range(m):
        for j in range(n):
            dist2 = 255 * 255 * 3
            index = -1
            for (ci, c) in enumerate(centers):
                d2 = get_distance(inp[i, j], c)
                if (d2 < dist2):
                    dist2 = d2
                    index = ci

            if (not flag and mat[i, j] != index):
                flag = True
            mat[i, j] = index

    return flag


def change_center(inp, mat, centers):
    (m, n) = mat.shape
    category = [[[], [], []] for _ in centers]
    for i in range(m):
        for j in range(n):
            for t in range(3):
                category[mat[i, j]][t].append(inp[i, j, t])

    for ci in range(len(centers)):
        if (len(category[ci][0]) > 0):
            for t in range(3):
                centers[ci][t] = np.mean(category[ci][t])


def regenerate_img(mat, centers, outp_dir):
    (m, n) = mat.shape
    arr = np.zeros([m, n, 3], dtype=np.uint8)
    for i in range(m):
        for j in range(n):
            arr[i, j] = centers[mat[i, j]]

    outp = Image.fromarray(arr)
    outp.save(outp_dir)
    print("Output: " + outp_dir)
    return outp


# main
K_LIST = [2, 5, 10, 15, 20]
files = ["Koala.jpg", "Penguins.jpg"]
for f in files:
    print("Img: " + f)
    img = io.imread(f)  # image is saved as rows * columns * 3 array
    for k in K_LIST:
        k_means_clustering(img, k, ("k=%d_compressed_" % k) + f)

input("Press and key to continue...")
