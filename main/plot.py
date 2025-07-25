import csv
import json
import matplotlib.pyplot as plt

# Charger les données d'entraînement
def load_data(filename='data.csv'):
    x = []
    y = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            try:
                x.append(float(row[0]))
                y.append(float(row[1]))
            except ValueError:
                continue
    return x, y

# Charger les paramètres appris
def load_thetas(filename='thetas.json'):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data['theta0'], data['theta1']

# Tracer les données + la droite
def plot_regression(x, y, theta0, theta1):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='Données réelles')
    
    # Ligne de régression
    x_line = list(range(int(min(x)), int(max(x)) + 1, 1000))
    y_line = [theta0 + theta1 * xi for xi in x_line]
    plt.plot(x_line, y_line, color='red', label='Régression linéaire')

    # Mise en forme
    plt.title('Régression linéaire - Prix vs Kilométrage')
    plt.xlabel('Kilométrage (km)')
    plt.ylabel('Prix (€)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Programme principal
if __name__ == "__main__":
    x, y = load_data()
    theta0, theta1 = load_thetas()
    plot_regression(x, y, theta0, theta1)
