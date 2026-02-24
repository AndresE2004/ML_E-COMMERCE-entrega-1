
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder




df = pd.read_csv("e_commerce_shopper_behaviour_and_lifestyle.csv")

print("Primeras filas:")
print(df.head())

print("\nInformación general:")
print(df.info())

print("\nDimensiones:")
print(df.shape)

print("\nEstadística descriptiva:")
print(df.describe())


#Limpieza

print("\nValores nulos antes:")
print(df.isnull().sum())

df.replace("?", np.nan, inplace=True)

# Mediana
df = df.fillna(df.median(numeric_only=True))

# Moda
for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("\nValores nulos después:")
print(df.isnull().sum())


#Variables categóricas

for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])


#Predecir income


X = df[['age', 'weekly_purchases', 'monthly_spend']]
y = df['income_level']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

print("\n===== REGRESIÓN LINEAL =====")
print("Coeficientes:", modelo.coef_)
print("Intercepto:", modelo.intercept_)
print("Error Cuadrático Medio:", mean_squared_error(y_test, y_pred))


#Grafica visual age

plt.scatter(df['age'], df['income_level'])


age_sorted = np.sort(df['age'])
pred_line = modelo.predict(
    pd.DataFrame({
        'age': age_sorted,
        'weekly_purchases': df['weekly_purchases'].mean(),
        'monthly_spend': df['monthly_spend'].mean()
    })
)

plt.plot(age_sorted, pred_line)
plt.xlabel("Age")
plt.ylabel("Income Level")
plt.title("Linear Regression")
plt.show()


#Graffica de residuos

residuos = y_test - y_pred

plt.scatter(y_pred, residuos)
plt.axhline(y=0)
plt.xlabel("Valores Predichos")
plt.ylabel("Residuos")
plt.title("Gráfico de Residuos")
plt.show()


#Por medio de la mediana vemos si el income es alto o bajo

df['Income_High'] = (df['income_level'] > df['income_level'].median()).astype(int)

X_log = df[['age', 'weekly_purchases', 'monthly_spend']]
y_log = df['Income_High']

X_train, X_test, y_train, y_test = train_test_split(
    X_log, y_log, test_size=0.2, random_state=42
)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

y_pred_log = log_reg.predict(X_test)

print("\n===== REGRESIÓN LOGÍSTICA =====")
print("Accuracy:", accuracy_score(y_test, y_pred_log))

print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred_log))

print("\nProbabilidades:")
print(log_reg.predict_proba(X_test))




print("\nIngreso promedio por edad:")
print(df.groupby('age')['income_level'].mean())


#Normalizamos

X_normalizado = (X - X.mean()) / X.std()

print("\nMedia después de normalizar:")
print(X_normalizado.mean())

print("\nDesviación estándar después de normalizar:")
print(X_normalizado.std())