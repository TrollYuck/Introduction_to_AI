import gzip
import pickle
import csv
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

USE_DOWNSAMPLING = True  # True = 14x14, False = 28x28
NUMBER_OF_CLUSTERS_LIST = [10, 15, 20, 30]


def load_mnist_data(path='data/mnist.pkl.gz'):
    with gzip.open(path, 'rb') as f:
        train_set, _, _ = pickle.load(f, encoding='latin1')
    X, y = train_set
    return X, y


def preprocess(X, downsample=True):
    if downsample:
        new_size = (14, 14)
        images = X.reshape(-1, 28, 28)
        pooled = images[:, ::2, ::2]  # 14x14 Wykonuje podpróbkowanie biorąc co drugi piksel w obu wymiarach (::2), co zmniejsza obrazy do 14x14
        flat = pooled.reshape(len(pooled), -1) # Spłaszcza obrazy do wektorów 196-elementowych
    else:
        new_size = (28, 28)
        flat = X
    return StandardScaler().fit_transform(flat), new_size # Zwraca znormalizowane dane oraz informację o rozmiarze obrazu


def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


def initialize_centroids_kmeans_pp(X, k):
    centroids = []
    centroids.append(X[np.random.randint(len(X))])
    for _ in range(1, k):
        dists = np.min(np.linalg.norm(X[:, None, :] - np.array(centroids)[None, :, :], axis=2) ** 2, axis=1)
        probs = dists / dists.sum()
        centroids.append(X[np.searchsorted(np.cumsum(probs), np.random.rand())])
    return np.array(centroids)


def k_means_cluster(X, k, max_iters=100, tol=1e-6):
    centroids = initialize_centroids_kmeans_pp(X, k)
    for it in range(max_iters):
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            mask = (labels == i)
            if np.any(mask):
                new_centroids[i] = X[mask].mean(axis=0)
            else:
                sse = np.min(dists ** 2, axis=1)
                worst = np.argmax(sse)
                new_centroids[i] = X[worst]
        if np.allclose(new_centroids, centroids, atol=tol):
            print(f"Konwergencja dla k={k} po {it + 1} iteracjach")
            break
        centroids = new_centroids
    return labels, centroids


def calculate_inertia(X, labels, centroids):
    inertia = 0
    for i in range(len(X)):
        inertia += euclidean_distance(X[i], centroids[labels[i]]) ** 2
    return inertia


def compute_cluster_label_matrix(labels, true_labels, k):
    matrix = np.zeros((k, 10), dtype=np.float64)
    counts = np.zeros(k)
    for i in range(len(labels)):
        cluster = labels[i]
        true = true_labels[i]
        matrix[cluster][true] += 1
        counts[cluster] += 1
    for i in range(k):
        if counts[i] > 0:
            matrix[i] = matrix[i] / counts[i] * 100
    return matrix


def plot_cluster_matrix(matrix, k, file_name):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="viridis",
                xticklabels=[str(i) for i in range(10)],
                yticklabels=[f"Cluster {i}" for i in range(k)])
    plt.xlabel("True digit label")
    plt.ylabel("Cluster")
    plt.title(f"Procentowy rozkład cyfr w klastrach (k={k})")
    plt.tight_layout()
    plt.savefig(f"cluster_matrix_{file_name}.png")
    plt.close()


def show_centroid_images(centroids, size, k, file_name):
    plt.figure(figsize=(15, 6))
    cols = min(k, 5)  # Maksymalnie 5 kolumn
    rows = (k + 4) // 5  # Zaokrąglanie w górę
    for i, centroid in enumerate(centroids):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(centroid.reshape(size), cmap='gray')
        plt.title(f'Centroid {i}')
        plt.axis('off')
    plt.suptitle(f"Obrazy centroidów (k={k})")
    plt.tight_layout()
    plt.savefig(f"centroids_{file_name}.png")
    plt.close()

X, y = load_mnist_data()
X_preprocessed, image_size = preprocess(X, downsample=USE_DOWNSAMPLING)

results = {}
for k in tqdm(NUMBER_OF_CLUSTERS_LIST, desc=f"K-means clustering"):
    file_name = f"{k}_no_downsample" if not USE_DOWNSAMPLING else f"{k}"
    print(f"\nKlasteryzacja dla k={k}")

    num_tries = 5
    best_inertia = float('inf')
    best_labels = None
    best_centroids = None

    for try_num in tqdm(range(num_tries), desc=f"K-means attempts (k={k})"):
        labels, centroids = k_means_cluster(X_preprocessed, k)
        inertia = calculate_inertia(X_preprocessed, labels, centroids)
        print(f"Próba {try_num + 1}: Inertia = {inertia:.2f}")
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_centroids = centroids

    print(f"Najlepszy wynik dla k={k}: Inertia = {best_inertia:.2f}")

    matrix = compute_cluster_label_matrix(best_labels, y, k)
    plot_cluster_matrix(matrix, k, file_name)
    show_centroid_images(best_centroids, image_size, k, file_name)
    results[k] = {
        'inertia': best_inertia,
        'labels': best_labels,
        'centroids': best_centroids,
        'matrix': matrix
    }

with open('inertia_results.csv', 'w', newline='') as csvfile:
    fieldnames = ['liczba_klastrow', 'inertia']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for k, result in results.items():
        writer.writerow({
            'liczba_klastrow': k,
            'inertia': result['inertia']
        })

print("Wyniki inercji zostały zapisane do pliku 'inertia_results.csv'")