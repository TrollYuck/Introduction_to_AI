from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

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

plot_homogeneity_vs_eps("dbscan_results.csv")
plot_noise_vs_eps("dbscan_results.csv")
plot_accuracy_vs_eps("dbscan_results.csv")