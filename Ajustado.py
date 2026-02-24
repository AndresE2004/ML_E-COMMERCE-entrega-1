import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


# Cargar datos
df = pd.read_csv("e_commerce_shopper_behaviour_and_lifestyle.csv")

# Limpieza
df.replace("?", np.nan, inplace=True)

# Rellenar numéricos con mediana
df = df.fillna(df.median(numeric_only=True))

# Rellenar categóricos con moda
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

#  Crear variable coherente
df['realistic_monthly_spend'] = (
    df['weekly_purchases'] * 4 *
    df['average_order_value'] *
    (df['purchase_conversion_rate'] / 100)
)

# Evitar problemas con log(0)
df = df[df['realistic_monthly_spend'] > 0]
df = df[df['weekly_purchases'] >= 0]
df = df[df['purchase_conversion_rate'] >= 0]

# Transformación logarítmica
df['log_spend'] = np.log(df['realistic_monthly_spend'])
df['log_weekly'] = np.log(df['weekly_purchases'] + 1)
df['log_avg'] = np.log(df['average_order_value'])
df['log_conversion'] = np.log(df['purchase_conversion_rate'] + 1)


#  Definir variables
X = df[['log_weekly', 'log_avg', 'log_conversion']]
y = df['log_spend']

#  División de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

#  Métricas
print("===== RESULTADOS =====")
print("R2:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))


# Coeficientes
coef_df = pd.DataFrame({
    "Variable": X.columns,
    "Coeficiente": modelo.coef_
})

print("\nCoeficientes:")
print(coef_df)

print("\nIntercepto:", modelo.intercept_)

# Nuevo gráfico de residuos
residuos = y_test - y_pred

plt.scatter(y_pred, residuos)
plt.axhline(0)
plt.xlabel("Valores Predichos (log)")
plt.ylabel("Residuos")
plt.title("Gráfico de Residuos")

plt.show()
