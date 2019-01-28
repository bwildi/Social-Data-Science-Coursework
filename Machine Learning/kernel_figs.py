import numpy as np
import matplotlib.pyplot as plt
np.random.seed(49)

x1 = np.random.normal(3, 1.5, 20)[:,np.newaxis]
x2 = np.random.normal(6, 1.5, 20)[:,np.newaxis]
y = np.sum(np.concatenate((x1, x2), axis=1), axis=1)
y[np.where(y < 9)] = -1
y[np.where(y >= 9)] = 1
y = np.int_(y)[:, np.newaxis]
plt.scatter(x1, x2, c=y, cmap="viridis")
plt.plot([0,4], [9,0])
plt.plot([0.5, 5], [9, 0])
plt.grid()
plt.xlabel("x1")
plt.ylabel("x2")
plt.xlim(0, 9)
plt.ylim(0, 9)
plt.show()

np.randomseed(3)
x5 = np.random.uniform(0, 6, 50)[:,np.newaxis]
x6 = np.random.uniform(0, 6, 50)[:,np.newaxis]
y3 = np.sum(np.concatenate((np.sqrt(x5), np.sqrt(x6)), axis=1), axis=1)
y3[np.where(y3 < 3)] = -1
y3[np.where(y3 >= 3)] = 1
y3 = np.int_(y3)[:, np.newaxis]
plt.scatter(x5, x6, c=y3, cmap="viridis")
plt.grid()
plt.xlabel("x1")
plt.ylabel("x2")
plt.xlim(0, 6)
plt.ylim(0, 6)
plt.show()