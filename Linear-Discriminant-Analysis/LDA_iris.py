# Importation des bibliothèques nécessaires
import numpy as np  # Bibliothèque pour le calcul scientifique
import matplotlib.pyplot as plt  # Bibliothèque pour la visualisation des données
import pandas as pd  # Bibliothèque pour la manipulation et l'analyse des données
from sklearn.preprocessing import StandardScaler  # Module pour la standardisation des données
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA  # Module pour l'Analyse Discriminante Linéaire

# Chargement du jeu de données Iris dans un DataFrame pandas
df = pd.read_csv("/home/kyriel/Downloads/Linear-Discriminant-Analysis-Using-Python-master/iris.csv")

# Séparation des caractéristiques et de la cible
features = ['sepal length', 'sepal width', 'petal length', 'petal width']  # Liste des noms des colonnes des caractéristiques
x = df.loc[:, features].values  # Extraction des valeurs des caractéristiques
y = df.loc[:,['target']].values.ravel()  # Extraction des valeurs de la cible et transformation en un tableau 1D

# Standardisation des caractéristiques
x = StandardScaler().fit_transform(x)  # Standardisation pour avoir une moyenne de 0 et un écart-type de 1

# Création du modèle LDA avec deux composantes
lda = LDA(n_components=2)

# Entraînement du modèle LDA et transformation des données
reduced_data = lda.fit(x, y).transform(x)  # Entraînement du modèle sur les données et transformation en deux dimensions

# Création d'un DataFrame pour les données réduites
principalDf = pd.DataFrame(data = reduced_data, columns = ['PC-1', 'PC-2'])  # Création d'un DataFrame avec les données réduites

# Ajout des étiquettes au DataFrame
finalDf = pd.concat([principalDf, df[['target']]], axis = 1)  # Concaténation des données réduites et des étiquettes dans un même DataFrame

# Création du graphique de dispersion pour les deux premières composantes principales
plt.figure(figsize = (8,8))  # Création d'une nouvelle figure avec une taille spécifiée
for target, color in zip(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], ['g', 'r', 'b']):  # Boucle sur chaque espèce et couleur correspondante
    indicesToKeep = finalDf['target'] == target  # Sélection des indices correspondant à l'espèce actuelle
    plt.scatter(finalDf.loc[indicesToKeep, 'PC-1'], finalDf.loc[indicesToKeep, 'PC-2'], c = color, s = 30)  # Ajout des points au graphique
plt.xlabel('PC-1')  # Ajout de l'étiquette de l'axe x
plt.ylabel('PC-2')  # Ajout de l'étiquette de l'axe y
plt.title('LDA sur le jeu de données Iris')  # Ajout du titre du graphique
plt.legend(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])  # Ajout de la légende
plt.grid()  # Ajout de la grille
plt.show()  # Affichage du graphique

# Enregistrement des données réduites dans un fichier CSV
finalDf.to_csv('iris_after_LDA.csv')  # Enregistrement du DataFrame dans un fichier CSV
