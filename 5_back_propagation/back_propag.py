import numpy as np
import matplotlib.pyplot as plt

# --- Funkcje aktywacji i ich pochodne ---

def sigmoid(x):
    """Funkcja aktywacji Sigmoid."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Pochodna funkcji Sigmoid."""
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    """Funkcja aktywacji ReLU."""
    return np.maximum(0, x)

def relu_derivative(x):
    """Pochodna funkcji ReLU."""
    return np.where(x > 0, 1, 0)

# --- Funkcja straty ---

def mse_loss(y_true, y_pred):
    """Mean Squared Error."""
    return np.mean((y_pred - y_true) ** 2)


def generate_data(num_samples=200):
    """Generuje dane dla problemu."""
    X = np.random.uniform(-1, 1, (num_samples, 2))
    # Usuwamy przypadki, gdzie x1 lub x2 jest bliskie zeru, aby uniknąć niejednoznaczności
    X = X[np.all(np.abs(X) > 1e-5, axis=1)]

    # Etykieta y = 1 jeśli znaki są takie same (x1*x2 > 0), wpp. y = 0
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    return X, y.reshape(-1, 1)


def normalize_l1(X):
    """Normalizacja L1.
    Dzieli każdy wektor danych przez sumę jego wartości bezwzględnych.
    Skaluje dane tak, że suma wartości bezwzględnych w każdym wierszu wynosi 1."""
    norm = np.sum(np.abs(X), axis=1, keepdims=True)
    return X / norm


def normalize_l2(X):
    """Normalizacja L2.
    Dzieli każdy wektor danych przez jego długość euklidesową.
    Skaluje dane tak, że każdy wektor staje się wektorem jednostkowym"""
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / norm


class TwoLayerNet:
    """
    Dwuwarstwowa sieć neuronowa z implementacją propagacji wstecznej.
    Architektura: 2 wejścia -> 4 neurony w warstwie ukrytej -> 1 neuron na wyjściu.
    """

    def __init__(self, input_size=2, hidden_size=4, output_size=1, activation='sigmoid'):
        # Inicjalizacja wag i biasów małymi losowymi wartościami
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))

        # Wybór funkcji aktywacji
        if activation == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        else:  # Domyślnie sigmoid
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative

    def forward(self, X):
        """Propagacja w przód."""
        # Warstwa ukryta
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.activation(self.Z1)

        # Warstwa wyjściowa
        self.Z2 = self.A1 @ self.W2 + self.b2
        # Na wyjściu zawsze używamy sigmoidy, aby uzyskać wynik w przedziale [0, 1]
        self.A2 = sigmoid(self.Z2)

        return self.A2

    def backward(self, X, y):
        """Propagacja wsteczna do obliczenia gradientów."""
        num_samples = X.shape[0]

        # Krok 1: Oblicz błąd na warstwie wyjściowej
        # Pochodna MSE: (y_pred - y_true)
        # Pochodna sigmoidy na wyjściu: sigmoid_derivative(Z2)
        delta_A2 = self.A2 - y
        delta_Z2 = delta_A2 * sigmoid_derivative(self.Z2)  # dError/dZ2

        # Krok 2: Oblicz gradienty dla wag i biasu warstwy wyjściowej
        self.dW2 = (self.A1.T @ delta_Z2) / num_samples
        self.db2 = np.sum(delta_Z2, axis=0, keepdims=True) / num_samples

        # Krok 3: Propaguj błąd do warstwy ukrytej
        delta_A1 = delta_Z2 @ self.W2.T  # dError/dA1
        delta_Z1 = delta_A1 * self.activation_derivative(self.Z1)  # dError/dZ1

        # Krok 4: Oblicz gradienty dla wag i biasu warstwy ukrytej
        self.dW1 = (X.T @ delta_Z1) / num_samples
        self.db1 = np.sum(delta_Z1, axis=0, keepdims=True) / num_samples

    def update_weights(self, learning_rate):
        """Aktualizacja wag i biasów."""
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1
        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.db2

    def train(self, X, y, epochs, learning_rate):
        """Pętla treningowa."""
        loss_history = []
        for epoch in range(epochs):
            # Propagacja w przód
            y_pred = self.forward(X)

            # Obliczanie straty
            loss = mse_loss(y, y_pred)
            loss_history.append(loss)

            # Propagacja wsteczna
            self.backward(X, y)

            # Aktualizacja wag
            self.update_weights(learning_rate)

            if (epoch + 1) % (epochs // 10) == 0:
                print(f"Epoka {epoch + 1}/{epochs}, Strata: {loss:.6f}")

        return loss_history

    def predict(self, X):
        """Predykcja dla nowych danych."""
        y_pred_proba = self.forward(X)
        return (y_pred_proba > 0.5).astype(int)

# Parametry
EPOCHS = 20000
LEARNING_RATES = [0.01, 0.1, 0.5, 0.75, 1.0, 2.5, 5.0] # Współczynnik uczenia

# Generowanie i przygotowanie danych
X_raw, y = generate_data(500)
X_l1 = normalize_l1(X_raw)
X_l2 = normalize_l2(X_raw)

# Słownik z danymi do testów
datasets = {
    "Nieznormalizowane": X_raw,
    "Znormalizowane L1": X_l1,
    "Znormalizowane L2": X_l2
}

# Słownik z funkcjami aktywacji
activations = ["sigmoid", "relu"]

# Przechowywanie wyników dla różnych współczynników uczenia
all_results = {}

# Przechowywanie wyników
for learning_rate in LEARNING_RATES:
    print(f"\n===== WSPÓŁCZYNNIK UCZENIA: {learning_rate} =====\n")

    # Słownik dla wyników z danym współczynnikiem uczenia
    results = {}

    # Utworzenie nowego wykresu dla danego współczynnika uczenia
    plt.figure(figsize=(14, 8))

    for norm_name, X_data in datasets.items():
        for act_name in activations:
            print(f"--- Trenowanie dla: {norm_name}, Aktywacja: {act_name.upper()} ---")

            # Inicjalizacja i trening sieci
            net = TwoLayerNet(activation=act_name)
            loss_history = net.train(X_data, y, epochs=EPOCHS, learning_rate=learning_rate)

            # Zapisanie wyników
            key = f"{norm_name} + {act_name.upper()}"
            results[key] = loss_history

            # Wykres
            plt.plot(loss_history, label=key)

            # Testowanie na kilku przykładach
            test_samples = X_data[:5]
            true_labels = y[:5].flatten()
            predictions = net.predict(test_samples).flatten()
            print(f"Przykładowe predykcje: {predictions}")
            print(f"Prawdziwe etykiety:    {true_labels}")
            print("-" * 50)

    # Zapisanie wyników dla danego współczynnika uczenia
    all_results[learning_rate] = results

    # Ustawienia wykresu dla danego współczynnika uczenia
    plt.title(f"Zbieżność straty dla różnych konfiguracji, LR = {learning_rate}")
    plt.xlabel("Epoka")
    plt.ylabel("Strata (MSE)")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 0.3)
    plt.savefig(f"loss_convergence_lr_{learning_rate}.png")
    plt.tight_layout()
    plt.show()

# Dodatkowy wykres porównujący najlepsze wyniki dla różnych współczynników uczenia
plt.figure(figsize=(14, 8))
for lr, results in all_results.items():
    # Dla uproszczenia, wybieramy tylko jedną konfigurację do porównania (np. Znormalizowane L2 + SIGMOID)
    key = "Znormalizowane L2 + SIGMOID"
    if key in results:
        plt.plot(results[key], label=f"LR = {lr}")

plt.title("Porównanie zbieżności dla różnych współczynników uczenia (L2 + SIGMOID)")
plt.xlabel("Epoka")
plt.ylabel("Strata (MSE)")
plt.legend()
plt.grid(True)
plt.ylim(0, 0.3)
plt.savefig("learning_rate_comparison.png")
plt.show()

