import csv
import json
import random

def load_data(filename='data.csv'):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # sauter l'en-tête
        for row in reader:
            try:
                km = float(row[0])
                price = float(row[1])
                data.append((km, price))
            except:
                pass
    return data

def normalize(x):
    x_min = min(x)
    x_max = max(x)
    x_norm = [(xi - x_min) / (x_max - x_min) for xi in x]
    return x_norm, x_min, x_max

def denormalize(x_norm, x_min, x_max):
    return [xi * (x_max - x_min) + x_min for xi in x_norm]

def split_data(data, train_ratio=0.8):
    random.shuffle(data)
    train_size = int(len(data) * train_ratio)
    train = data[:train_size]
    test = data[train_size:]
    return train, test

def train_model(x, y, learning_rate=0.001, iterations=10000):
    theta0 = 0
    theta1 = 0
    m = len(x)

    for _ in range(iterations):
        sum_errors_0 = sum((theta0 + theta1 * x[i] - y[i]) for i in range(m))
        sum_errors_1 = sum((theta0 + theta1 * x[i] - y[i]) * x[i] for i in range(m))

        theta0 -= learning_rate * (1/m) * sum_errors_0
        theta1 -= learning_rate * (1/m) * sum_errors_1

    return theta0, theta1

def predict(theta0, theta1, x):
    return [theta0 + theta1 * xi for xi in x]

def mean_squared_error(y_true, y_pred):
    n = len(y_true)
    return sum((y_true[i] - y_pred[i])**2 for i in range(n)) / n

def mean_absolute_error(y_true, y_pred):
    n = len(y_true)
    return sum(abs(y_true[i] - y_pred[i]) for i in range(n)) / n

def r2_score(y_true, y_pred):
    mean_y = sum(y_true) / len(y_true)
    ss_res = sum((y_true[i] - y_pred[i])**2 for i in range(len(y_true)))
    ss_tot = sum((y_true[i] - mean_y)**2 for i in range(len(y_true)))
    return 1 - (ss_res / ss_tot)

if __name__ == "__main__":
    data = load_data()
    train_data, test_data = split_data(data, train_ratio=0.8)

    x_train_raw = [point[0] for point in train_data]
    y_train = [point[1] for point in train_data]

    x_test_raw = [point[0] for point in test_data]
    y_test = [point[1] for point in test_data]

    # Normalisation des données d'entrée (km)
    x_train, x_min, x_max = normalize(x_train_raw)
    # Normaliser aussi le test avec les mêmes paramètres
    x_test = [(xi - x_min) / (x_max - x_min) for xi in x_test_raw]

    theta0, theta1 = train_model(x_train, y_train, learning_rate=0.001, iterations=10000)

    print(f"Modèle entraîné : θ0 = {theta0:.4f}, θ1 = {theta1:.6f}")

    # Évaluer sur jeu d'entraînement normalisé
    y_train_pred = predict(theta0, theta1, x_train)
    mse_train = mean_squared_error(y_train, y_train_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)

    print("\nSur le jeu d'entraînement :")
    print(f"  MSE = {mse_train:.2f}")
    print(f"  MAE = {mae_train:.2f}")
    print(f"  R² = {r2_train:.4f}")

    # Évaluer sur jeu de test normalisé
    y_test_pred = predict(theta0, theta1, x_test)
    mse_test = mean_squared_error(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)

    print("\nSur le jeu de test :")
    print(f"  MSE = {mse_test:.2f}")
    print(f"  MAE = {mae_test:.2f}")
    print(f"  R² = {r2_test:.4f}")

    # Sauvegarder theta0, theta1 et les paramètres de normalisation
    with open('thetas.json', 'w') as file:
        json.dump({
            "theta0": theta0,
            "theta1": theta1,
            "x_min": x_min,
            "x_max": x_max
        }, file)
