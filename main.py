import numpy as np
import pandas as pd
from KMeans import KMeans
import matplotlib.pyplot as plt


df = pd.read_csv("student_clustering.csv")

km = KMeans()
X = df.iloc[:, :].values
y_means = km.fitPredict(X)

# for i in range(km.n_clusters):
# plt.show()

plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], marker='*')
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], marker='*')
plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], marker='*')
plt.scatter(X[y_means == 3, 0], X[y_means == 3, 1], marker='*')
plt.show()