# -*- coding: utf-8 -*-
import random
import time

a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
print(a)
b = a[:3]
b[1] = 0
print(a)

def change(a, i):
    a[i] = -2
    return True

change(a, 5)
print(a)

start_time = time.time()
for i in range(100):
    print(random.randint(0, 5))
end_time = time.time()
sec_cost = end_time - start_time
min, sec = divmod(sec_cost, 60)
print("TIme cost: {0:.0f} min {1:.3f} sec,".format(min, sec))