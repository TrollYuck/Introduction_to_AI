import gzip
import pickle
import numpy as np
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from collections import deque, Counter, defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
import pandas as pd

def load_mnist(path='data/mnist.pkl.gz'):
    with gzip.open(path, 'rb') as f:
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = pickle.load(f, encoding='latin1')
    X = np.concatenate([x_train, x_val, x_test])
    y = np.concatenate([y_train, y_val, y_test])
    return X, y


def preprocess_data(X, n_components=30, method='umap', random_state=25):
    if method == 'pca':
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = PCA(n_components=n_components)
        X_reduced = model.fit_transform(X_scaled)
        print(f"Zredukowana liczba wymiarów (PCA): {X_reduced.shape[1]}")
    elif method == 'umap':
        X_scaled = X
        model = umap.UMAP(n_components=n_components, random_state=random_state)
        X_reduced = model.fit_transform(X_scaled)
        scaler = None
        print(f"Zredukowana liczba wymiarów (UMAP): {X_reduced.shape[1]}")
    else:
        raise ValueError("Nieznana metoda: użyj 'pca' lub 'umap'.")

    return X_reduced, model, scaler

def visualize_2d(X_reduced, y, title="Redukcja do 2D"):
    plt.figure(figsize=(8, 6))
    for digit in tqdm(range(10), desc="Generowanie wykresu"):
        idx = y == digit
        plt.scatter(X_reduced[idx, 0], X_reduced[idx, 1], label=str(digit), alpha=0.5, s=10)
    plt.legend()
    plt.title(title)
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.tight_layout()
    plt.savefig("reduced_2d_plot.png")
    plt.show()

def dbscan(X, eps=1.5, min_samples=5):
    n = X.shape[0]
    labels = np.full(n, -1)  # -1 oznacza szum
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    nbrs = NearestNeighbors(radius=eps, algorithm='kd_tree').fit(X)
    dists, indices = nbrs.radius_neighbors(X)

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        neighbors = indices[i]

        if len(neighbors) < min_samples:
            labels[i] = -1
        else:
            labels[i] = cluster_id
            queue = deque(neighbors)
            while queue:
                j = queue.popleft()
                if not visited[j]:
                    visited[j] = True
                    j_neighbors = indices[j]
                    if len(j_neighbors) >= min_samples:
                        queue.extend([nj for nj in j_neighbors if not visited[nj]])
                    if labels[j] == -1:
                        labels[j] = cluster_id
            cluster_id += 1

    return labels

def plot_dbscan_result(X, labels, title="DBSCAN clustering"):
    plt.figure(figsize=(8, 6))
    unique_labels = set(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    for k, col in tqdm(zip(unique_labels, colors), desc="Generowanie wykresu klastrów", total=len(unique_labels)):
        if k == -1:
            col = 'k'
        class_member_mask = (labels == k)
        plt.scatter(X[class_member_mask, 0], X[class_member_mask, 1],
                    c=[col], label=f'Cluster {k}' if k != -1 else "Noise",
                    s=10, alpha=0.6)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig("dbscan_result.png")
    plt.show()

def analyze_clusters(labels, true_labels):
    cluster_info = defaultdict(list)
    for label, true in zip(labels, true_labels):
        if label != -1:  # pomijamy szum
            cluster_info[label].append(true)

    print(f"\nAnaliza klastrów ({len(cluster_info)} klastrów):")
    homogeneity = []
    for cluster_id, digits in tqdm(sorted(cluster_info.items()), desc="Analiza klastrów"):
        total = len(digits)
        counts = Counter(digits)
        most_common_digit, count = counts.most_common(1)[0]
        percent = 100 * count / total
        homogeneity.append(percent)
        print(f"Klaster {cluster_id:2d}: dominująca cyfra = {most_common_digit}, "
              f"{count}/{total} ({percent:.2f}%)")

    avg_homogeneity = np.mean(homogeneity) if homogeneity else 0
    print(f"\nŚrednia homogeniczność klastrów: {avg_homogeneity:.2f}%")

    correct = 0
    cluster_to_digit = {}
    for cluster_id in cluster_info:
        digits = cluster_info[cluster_id]
        cluster_to_digit[cluster_id] = Counter(digits).most_common(1)[0][0]
    for i, label in enumerate(labels):
        if label != -1 and true_labels[i] == cluster_to_digit[label]:
            correct += 1
    accuracy = (correct / len(labels) * 100) if len(labels) > 0 else 0
    noise_percent = np.sum(labels == -1) / len(labels) * 100
    print(f"Dokładność klasyfikacji: {accuracy:.2f}%")
    print(f"Procent szumu: {noise_percent:.2f}%")
    print(f"Średni procent błędów w klastrach: {100 - avg_homogeneity:.2f}%")

    return homogeneity, accuracy, noise_percent

def grid_search_dbscan(X, y, eps_values, min_samples_values, max_clusters=30, output_csv="dbscan_results.csv"):
    best_setting = None
    best_score = -1

    results = []

    print(f"\nGrid search DBSCAN: {len(eps_values) * len(min_samples_values)} kombinacji\n")

    for eps in tqdm(eps_values, desc="Przeszukiwanie eps"):
        for min_samples in min_samples_values:
            labels = dbscan(X, eps=eps, min_samples=min_samples)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = np.sum(labels == -1)

            # if n_clusters == 0 or n_clusters > max_clusters:
            #     continue

            print(f"Testowane parametry: eps={eps}, min_samples={min_samples} → klastry: {n_clusters}")

            homo_scores, accuracy, noise_percent = analyze_clusters(labels, y)
            avg_homo = np.mean(homo_scores) if homo_scores else 0

            print(f"[eps={eps:.2f}, min_samples={min_samples:2d}] → "
                  f"{n_clusters:2d} klastrów, {n_noise} szumu ({noise_percent:.2f}%), "
                  f"homogeniczność: {avg_homo:.2f}%, dokładność: {accuracy:.2f}%")

            results.append({
                "eps": round(eps, 3),
                "min_samples": min_samples,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "noise_percent": round(noise_percent, 2),
                "homogeneity": round(avg_homo, 2),
                "accuracy": round(accuracy, 2)
            })

            if avg_homo > best_score:
                best_score = avg_homo
                best_setting = (eps, min_samples, n_clusters, n_noise, accuracy)

    with open(output_csv, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    if best_setting:
        eps, min_samples, n_clusters, n_noise, accuracy = best_setting
        print(f"\nNajlepsze ustawienia:")
        print(f"   eps = {eps}, min_samples = {min_samples}")
        print(f"   klastry = {n_clusters}, szum = {n_noise} ({n_noise/len(X)*100:.2f}%), "
              f"homogeniczność = {best_score:.2f}%, dokładność = {accuracy:.2f}%")
    else:
        print("Nie znaleziono dobrych ustawień.")

    return best_setting

def plot_clean_vs_dirty_clusters(X_2d, labels, y_true, purity_threshold=0.9):
    cluster_info = defaultdict(list)
    for i, lbl in enumerate(labels):
        if lbl != -1:
            cluster_info[lbl].append(y_true[i])

    clean_clusters = set()
    dirty_clusters = set()
    for cid, digits in tqdm(cluster_info.items(), desc="Analizowanie czystości klastrów"):
        counts = Counter(digits)
        most_common, count = counts.most_common(1)[0]
        if count / len(digits) >= purity_threshold:
            clean_clusters.add(cid)
        else:
            dirty_clusters.add(cid)

    plt.figure(figsize=(10, 6))
    for i, (x, y) in enumerate(X_2d):
        lbl = labels[i]
        if lbl == -1:
            plt.scatter(x, y, c='black', s=5, alpha=0.3)
        elif lbl in clean_clusters:
            plt.scatter(x, y, c='green', s=5, alpha=0.6)
        elif lbl in dirty_clusters:
            plt.scatter(x, y, c='red', s=5, alpha=0.6)

    plt.title("Czyste (zielone) vs zanieczyszczone (czerwone) klastry | Szum = czarne")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig("clean_vs_dirty_clusters.png")
    plt.show()

def plot_homogeneity_vs_eps(csv_file="dbscan_results.csv"):
    df = pd.read_csv(csv_file)

    plt.figure(figsize=(10, 6))
    min_samples_values = sorted(df['min_samples'].unique())
    for min_samples in tqdm(min_samples_values, desc="Generowanie wykresu homogeniczności"):
        subset = df[df['min_samples'] == min_samples]
        plt.plot(subset['eps'], subset['homogeneity'], marker='o', label=f'min_samples={min_samples}')

    plt.title("Homogeniczność vs eps dla różnych min_samples")
    plt.xlabel("eps")
    plt.ylabel("Homogeniczność (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("homogeneity_vs_eps.png")
    plt.show()

def plot_noise_vs_eps(csv_file="dbscan_results.csv"):
    df = pd.read_csv(csv_file)

    plt.figure(figsize=(10, 6))
    min_samples_values = sorted(df['min_samples'].unique())
    for min_samples in tqdm(min_samples_values, desc="Generowanie wykresu szumu"):
        subset = df[df['min_samples'] == min_samples]
        plt.plot(subset['eps'], subset['noise_percent'], marker='o', label=f'min_samples={min_samples}')

    plt.title("Procent szumu vs eps dla różnych min_samples")
    plt.xlabel("eps")
    plt.ylabel("Procent szumu (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("noise_vs_eps.png")
    plt.show()

def plot_accuracy_vs_eps(csv_file="dbscan_results.csv"):
    df = pd.read_csv(csv_file)

    plt.figure(figsize=(10, 6))
    min_samples_values = sorted(df['min_samples'].unique())
    for min_samples in tqdm(min_samples_values, desc="Generowanie wykresu dokładności"):
        subset = df[df['min_samples'] == min_samples]
        plt.plot(subset['eps'], subset['accuracy'], marker='o', label=f'min_samples={min_samples}')

    plt.title("Dokładność vs eps dla różnych min_samples")
    plt.xlabel("eps")
    plt.ylabel("Dokładność (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("accuracy_vs_eps.png")
    plt.show()


def plot_cluster_averages(X_original, labels, save_path="cluster_averages.png", img_shape=(28, 28)):
    """
    Tworzy wykres ze średnimi obrazami dla każdego klastra.

    Parameters:
        X_original: ndarray (n_samples, 784)
            Oryginalne dane obrazów.
        labels: ndarray (n_samples,)
            Etykiety klastrów (-1 oznacza szum).
        save_path: str
            Ścieżka do zapisu pliku PNG.
        img_shape: tuple
            Rozmiar obrazka (domyślnie MNIST: 28x28).
    """
    from math import ceil, sqrt

    unique_clusters = sorted(set(labels))
    if -1 in unique_clusters:
        unique_clusters.remove(-1)  # pomijamy szum

    n_clusters = len(unique_clusters)
    cols = min(10, n_clusters)
    rows = ceil(n_clusters / cols)

    plt.figure(figsize=(1.5 * cols, 1.5 * rows))

    for idx, cluster_id in enumerate(unique_clusters):
        cluster_images = X_original[labels == cluster_id]
        if len(cluster_images) == 0:
            continue
        avg_image = np.mean(cluster_images, axis=0).reshape(img_shape)

        plt.subplot(rows, cols, idx + 1)
        plt.imshow(avg_image, cmap='gray')
        plt.axis('off')
        plt.title(f'Cluster {cluster_id}')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


# Główny program
method = 'umap'

X, y = load_mnist()

# Redukcja do 50D do klateryzacji
X_reduced, model, scaler = preprocess_data(X, n_components=50, method=method)

# Redukcja do 2D do wizualizacji
X_2d, _, _ = preprocess_data(X, n_components=2, method=method)
visualize_2d(X_2d, y, title=f"{method.upper()} (2D)")

eps_range = np.arange(0.2, 0.6, 0.05)
min_samples_range = [3, 5, 7, 12]

best_params = grid_search_dbscan(X_reduced, y, eps_range, min_samples_range)

if best_params:
    best_eps, best_min_samples, *_ = best_params
    final_labels = dbscan(X_reduced, eps=best_eps, min_samples=best_min_samples)
    X_2d, _, _ = preprocess_data(X, n_components=2)
    plot_dbscan_result(X_2d, final_labels, title=f"DBSCAN: eps={best_eps}, min_samples={best_min_samples}")
    plot_clean_vs_dirty_clusters(X_2d, final_labels, y)
    plot_homogeneity_vs_eps("dbscan_results.csv")
    plot_noise_vs_eps("dbscan_results.csv")
    plot_accuracy_vs_eps("dbscan_results.csv")
    plot_cluster_averages(X, final_labels, save_path="cluster_averages.png")

