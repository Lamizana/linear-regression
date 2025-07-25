import json

# Charger les valeurs de theta, min_x et max_x
def load_thetas(filename='thetas.json'):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data['theta0'], data['theta1'], data['min_x'], data['max_x']

# Prédire le prix
def predict_price(km, theta0, theta1):
    return theta0 + theta1 * km

# Lancer la prédiction
if __name__ == "__main__":
    try:
        km = float(input("Entrez le kilométrage de la voiture : "))
        theta0, theta1, min_x, max_x = load_thetas()
        price = predict_price(km, theta0, theta1)
        print(f"Prix estimé pour {km:.0f} km : {price:.2f} €")
    except ValueError:
        print("Veuillez entrer un nombre valide.")
    except FileNotFoundError:
        print("Fichier 'thetas.json' introuvable. Lancez d'abord train.py.")
