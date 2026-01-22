import gzip
import pickle

import numpy as np
import umap
from scipy.stats import mode
from sklearn.neighbors import NearestNeighbors
import random
import os

seed = 2137
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
with gzip.open('data/mnist.pkl.gz', 'rb') as f:
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = pickle.load(f, encoding='latin1')

X = np.concatenate((X_train, X_test))
y = np.concatenate((y_train, y_test))
X = X.reshape(X.shape[0], -1) / 255.0

idx = np.random.choice(len(X), size=20000, replace=False)
X_sample = X[idx]
y_sample = y[idx]

reducer = umap.UMAP(n_components=10, n_neighbors=15, random_state=seed)
X_umap = reducer.fit_transform(X_sample)


def cluster_accuracy(true_labels, pred_labels):
    clusters = set(pred_labels)
    clusters.discard(-1)
    total_correct = 0
    total_points = 0
    error_points = 0
    for cluster in clusters:
        indices = np.where(pred_labels == cluster)[0]
        if len(indices) == 0:
            continue
        cluster_true = true_labels[indices]
        majority = mode(cluster_true, keepdims=True).mode[0]
        correct = np.sum(cluster_true == majority)
        total_correct += correct
        total_points += len(indices)
        error_points += len(indices) - correct
    accuracy = total_correct / total_points if total_points else 0
    noise = np.sum(pred_labels == -1) / len(pred_labels) * 100
    error_rate = error_points / total_points * 100 if total_points else 0
    return accuracy, noise, error_rate, len(clusters)


def dbscan_manual(X, eps, min_samples):
    n = X.shape[0]
    labels = np.full(n, -1)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    nn = NearestNeighbors(radius=eps, n_jobs=-1)
    nn.fit(X)
    neighbors = nn.radius_neighbors(X, return_distance=False)

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neigh = neighbors[i]
        if len(neigh) < min_samples:
            continue  # punkt jest szumem lub brzegiem

        labels[i] = cluster_id
        seeds = list(neigh)
        seeds.remove(i)
        while seeds:
            j = seeds.pop()
            if not visited[j]:
                visited[j] = True
                neigh_j = neighbors[j]
                if len(neigh_j) >= min_samples:
                    for q in neigh_j:
                        if labels[q] == -1:
                            seeds.append(q)
            if labels[j] == -1:
                labels[j] = cluster_id
        cluster_id += 1

    return labels


eps_values = np.arange(0.2, 0.6, 0.05)
min_samples_values = range(3, 12)

best_acc = 0
best_params = None

print("Szukanie najlepszych parametrów manual DBSCAN...")

for eps in eps_values:
    for min_samples in min_samples_values:
        labels = dbscan_manual(X_umap, eps, min_samples)
        acc, noise, error, n_clusters = cluster_accuracy(y_sample, labels)

        if acc > best_acc and n_clusters >= 6 and n_clusters <= 30:
            best_acc = acc
            best_params = (eps, min_samples, n_clusters, acc, noise, error)
            print(f"Nowy najlepszy wynik: eps={eps:.2f}, min_samples={min_samples} -> "
                  f"acc={acc * 100:.2f}%, clusters={n_clusters}, noise={noise:.2f}%")

if best_params is not None:
    print("\nNajlepsze parametry:")
    print(f"eps = {best_params[0]:.2f}")
    print(f"min_samples = {best_params[1]}")
    print(f"liczba klastrów = {best_params[2]}")
    print(f"dokładność = {best_params[3] * 100:.2f}%")
    print(f"procent szumu = {best_params[4]:.2f}%")
    print(f"procent błędów = {best_params[5]:.2f}%")
else:
    print("Nie znaleziono parametrów spełniających kryteria.")