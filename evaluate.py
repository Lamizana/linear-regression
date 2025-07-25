import csv
import json

# Charger les données (km, price)
def load_data(filename='data.csv'):
    x = []
    y = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            try:
                x.append(float(row[0]))
                y.append(float(row[1]))
            except:
                pass
    return x, y

# Charger θ0, θ1
def load_thetas(filename='thetas.json'):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data['theta0'], data['theta1']

# Prédire avec le modèle
def predict(theta0, theta1, x):
    return [theta0 + theta1 * xi for xi in x]

# Calcul MSE
def mean_squared_error(y_true, y_pred):
    n = len(y_true)
    return sum((y_true[i] - y_pred[i])**2 for i in range(n)) / n

# Calcul MAE
def mean_absolute_error(y_true, y_pred):
    n = len(y_true)
    return sum(abs(y_true[i] - y_pred[i]) for i in range(n)) / n

# Calcul R²
def r2_score(y_true, y_pred):
    mean_y = sum(y_true) / len(y_true)
    ss_res = sum((y_true[i] - y_pred[i])**2 for i in range(len(y_true)))
    ss_tot = sum((y_true[i] - mean_y)**2 for i in range(len(y_true)))
    return 1 - (ss_res / ss_tot)

if __name__ == "__main__":
    x, y_true = load_data()
    theta0, theta1 = load_thetas()
    y_pred = predict(theta0, theta1, x)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"MSE (Mean Squared Error) : {mse:.2f}")
    print(f"MAE (Mean Absolute Error) : {mae:.2f}")
    print(f"R² score : {r2:.4f}")
