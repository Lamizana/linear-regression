import csv
import json

# Charger les données CSV
def load_data(filename):
    x = []
    y = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Ignore l'en-tête
        for row in reader:
            try:
                x.append(float(row[0]))
                y.append(float(row[1]))
            except ValueError:
                continue  # Ignore les lignes invalides
    return x, y

# Normaliser x (kilométrage)
def normalize(x):
    min_x = min(x)
    max_x = max(x)
    x_norm = [(xi - min_x) / (max_x - min_x) for xi in x]
    return x_norm, min_x, max_x

# Descente de gradient
def train(x, y, learning_rate=0.01, iterations=5000):
    x_norm, min_x, max_x = normalize(x)
    theta0 = 0
    theta1 = 0
    m = len(x)

    for _ in range(iterations):
        sum_errors_0 = sum((theta0 + theta1 * x_norm[i] - y[i]) for i in range(m))
        sum_errors_1 = sum((theta0 + theta1 * x_norm[i] - y[i]) * x_norm[i] for i in range(m))

        theta0 -= learning_rate * (1 / m) * sum_errors_0
        theta1 -= learning_rate * (1 / m) * sum_errors_1

    # Reconvertir dans l’échelle réelle (non normalisée)
    scale = max_x - min_x
    theta1_real = theta1 / scale
    theta0_real = theta0 - (theta1 * min_x / scale)

    return theta0_real, theta1_real, min_x, max_x

# Sauvegarder dans thetas.json
def save_thetas(theta0, theta1, min_x, max_x, filename='thetas.json'):
    data = {
        'theta0': theta0,
        'theta1': theta1,
        'min_x': min_x,
        'max_x': max_x
    }
    with open(filename, 'w') as file:
        json.dump(data, file)

# Lancement du programme
if __name__ == "__main__":
    x, y = load_data('data.csv')
    theta0, theta1, min_x, max_x = train(x, y)
    save_thetas(theta0, theta1, min_x, max_x)
    print(f"Modèle entraîné : θ0 = {theta0:.4f}, θ1 = {theta1:.6f}")
