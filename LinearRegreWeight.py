import pandas as pd
df = pd.read_csv("age_vs_poids_vs_taille_vs_sexe.csv")
df.describe()
# sub dataframe containing predictiv variables
#By convention, we denote the predictor matrix X
X = df[['sexe','age', 'taille']]

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

#coefficients a,b,c of liear regression
print(reg.coef_)

#the constant term
print(reg.intercept_)

y_pred = reg.predict(df[['sexe', 'age', 'taille']])

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
print("1er Model: weight ~ gender + age")
print(f"\tmean_squared_error(y, y_pred): {mean_squared_error(y, y_pred)}")
print(f"\tmean_absolute_error(y, y_pred): {mean_absolute_error(y, y_pred)}")
print(f"\tmean_absolute_percentage_error(y, y_pred): {mean_absolute_percentage_error(y, y_pred)}")
print()



# New data for prediction (example: 30 years old, 180 cm, male)
new_data = pd.DataFrame({'sexe': [0], 'age': [29*12], 'taille': [167] })
new_data2 = pd.DataFrame({'sexe': [1], 'age': [29*12], 'taille': [188] })

# Predict with the trained model
predicted_weight = reg.predict(new_data)
predicted_weight2 = reg.predict(new_data2)

print(predicted_weight)
print(predicted_weight2)


'''

weight ≈ a × age + b × height + c × gender + constant/noise


model has moderate predictive power (63% explained)

The values [0.15299379 0.10803477 0.55435737] here are:

age: +0.108 → every year of age increases predicted weight by ~0.11 kg, all else equal.

taille: +0.554 → every extra cm of height adds ~0.55 kg to predicted weight.

sexe: +0.153 → if sexe is encoded as 0/1 (e.g., female=0, male=1), being male adds ~0.15 kg.

 try polynomial regression, decision trees?
 
'''
