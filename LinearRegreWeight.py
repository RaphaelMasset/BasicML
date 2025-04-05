import pandas as pd
df = pd.read_csv("age_vs_poids_vs_taille_vs_sexe.csv")

# sub dataframe containing predictiv variables
#By convention, we denote the predictor matrix X
X = df[['age', 'taille', 'sexe']]

# target variable, weight
y = df.poids

# on choisit un modèle de régression linéaire
from sklearn.linear_model import LinearRegression

#instance of the Linear Regression model
reg = LinearRegression()

# on entraîne ce modèle sur les données avec la méthode fit
reg.fit(X, y)

# et on obtient directement un score
print(reg.score(X, y))

# ainsi que les coefficients a,b,c de la régression linéaire
print(reg.coef_)