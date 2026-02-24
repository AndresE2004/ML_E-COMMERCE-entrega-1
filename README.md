# Entrega 1 - Análitica de Datos

Esta es la entrega número 1 de la clase analitica de datos.

## Integrantes

**Andres Escobar Hincapie**  
Código de estudiante: 30000090946

**Juan Esteban Murillo**  
Código de estudiante: 30000091189

## Descripción del Proyecto

Este proyecto realiza análisis de Machine Learning sobre un dataset de comportamiento de compras en e-commerce, incluyendo:

- Regresión lineal para predecir niveles de ingresos
- Regresión logística para clasificación de ingresos
- Análisis estadístico y visualización de datos
- Limpieza y procesamiento de datos

## Cómo ejecutar el proyecto

1. Asegúrate de tener Python instalado
2. Instala las dependencias necesarias:
   ```bash
   pip install pandas numpy matplotlib scikit-learn
   ```
3. Ejecuta el script principal:
   ```bash
   python run.py
   ```

## Dataset

Se utiliza el archivo `e_commerce_shopper_behaviour_and_lifestyle.csv` que contiene datos de comportamiento de compradores en línea con 60 columnas y 1,000,000 de registros.

debido a que el dataset es muy grande, no permite ser adjuntado en el repositorio, por lo que se debe descargar manualmente desde el siguiente enlace:
https://www.kaggle.com/datasets/andrewmvd/ecommerce-customers-behavior-and-lifestyle


# E-Commerce Shopper Behavior – Modelado Predictivo

##  Descripción del Proyecto

Este proyecto analiza el comportamiento de compra en un entorno de e-commerce utilizando modelos de regresión.  
El objetivo inicial fue predecir el nivel de ingresos (`income_level`) a partir de variables demográficas y de consumo.

Durante el desarrollo se detectó un problema de especificación del modelo, lo que llevó a reformular el enfoque para construir un modelo con fundamento matemático y económico sólido.

Este repositorio muestra tanto el intento inicial como la versión corregida y optimizada.

---

# Modelo Inicial

## Objetivo

Predecir `income_level` usando:

- `age`
- `weekly_purchases`
- `monthly_spend`

##  Procedimiento

- Limpieza de datos (mediana para numéricos, moda para categóricos)
- Codificación con `LabelEncoder`
- División entrenamiento/prueba (80/20)
- Regresión Lineal
- Evaluación con R² y RMSE
- Visualización de residuos
- Regresión Logística adicional para clasificación binaria

## Resultados

- R² ≈ 0
- RMSE elevado
- Baja capacidad predictiva

## Problema Identificado

El modelo no funcionó por las siguientes razones:

1. No existía una relación lineal clara entre las variables predictoras y `income_level`.
2. Se intentó modelar una variable ordinal como si fuera continua.
3. No había una hipótesis económica detrás de la selección de variables.
4. No existía una estructura matemática que justificara el modelo.

---

# Reformulación del Modelo en Ajustado.py

## nueva variable

En lugar de forzar una relación débil, se construyó una variable con sentido económico:

```
monthly_spend ≈ weekly_purchases × average_order_value × conversion_rate
```

Esto representa el comportamiento real de un cliente en e-commerce:

> Gasto = Frecuencia × Ticket Promedio × Tasa de Conversión

Aquí sí existe una relación estructural entre variables.

---

# Transformación Log-Log

Para modelar una relación multiplicativa se aplicó transformación logarítmica:

```
log(spend) = β0 
             + β1 log(weekly_purchases)
             + β2 log(average_order_value)
             + β3 log(conversion_rate)
```


# Resultados del Modelo Reformulado

R²: 0.99688  
RMSE: 0.082  

Coeficientes:

| Variable | Coeficiente |
|-----------|-------------|
| log_weekly | 1.21 |
| log_avg | 0.99 |
| log_conversion | 1.08 |

---
