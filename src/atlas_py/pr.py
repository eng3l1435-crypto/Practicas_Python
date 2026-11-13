import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# datos = {
#     "horas_estudio": [
#         1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
#         1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
#         2, 4, 6, 8, 10, 3, 5, 7, 9, 6
#     ],

#     "asistencia": [
#         40, 50, 55, 60, 65, 70, 75, 80, 85, 90,
#         45, 55, 60, 65, 70, 75, 80, 85, 90, 95,
#         50, 60, 70, 80, 95, 55, 75, 85, 90, 68
#     ],

#     "actividades_entregadas": [
#         30, 40, 45, 55, 60, 65, 70, 80, 90, 95,
#         20, 35, 50, 60, 65, 70, 75, 85, 95, 100,
#         40, 55, 75, 85, 100, 45, 70, 80, 90, 60
#     ],

#     "aprueba": [
#         0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
#         0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
#         0, 0, 1, 1, 1, 0, 1, 1, 1, 0
#     ],

#     "calificacion_final": [
#     30, 40, 45, 55, 60, 65, 70, 80, 90, 95,
#     20, 35, 50, 60, 65, 70, 75, 85, 95, 100,
#     40, 55, 75, 85, 100, 45, 70, 80, 90, 60
#     ],

#     "participacion": [
#     1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
#     1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
#     2, 4, 6, 8, 10, 3, 5, 7, 9, 6
#     ]
# }

# 60 entrenamiento - Aprender patrones
# 20 validacion - Ajusta los parametros y elige modelos
# 20 prueba - evalua el desempeño final con datos no utilizados, anteriormente

#Cantidad de estudiantes 500 y usar random para crear los datos
#Horas de estudio entre 0 y 15
#Asistencia entre 30 y 100
#actividades entre 10 y 100
#Participacion entre 0 y 10
#calificacion de 0 a 100
#participacion de 0 a 10


# 1. Generación de datos aleatorios para 500 estudiantes
np.random.seed(42)
n_estudiantes = 500

datos = pd.DataFrame({
    "horas_estudio": np.random.randint(0, 16, n_estudiantes),            # 0 a 15
    "asistencia": np.random.randint(30, 101, n_estudiantes),             # 30 a 100
    "actividades_entregadas": np.random.randint(10, 101, n_estudiantes), # 10 a 100
    "participacion": np.random.randint(0, 11, n_estudiantes),            # 0 a 10
    "calificacion_final": np.random.randint(0, 101, n_estudiantes)       # 0 a 100
})

# Creamos la variable objetivo (target)
datos["aprueba"] = (datos["calificacion_final"] >= 60).astype(int)

# 2. Definir variables predictoras (X) y objetivo (y)
# Quitamos 'calificacion_final' y 'aprueba' para que el modelo aprenda de los otros hábitos
X = datos.drop(columns=["aprueba", "calificacion_final"])
y = datos["aprueba"]

# 3. División de los datos (60% train, 20% val, 20% test)
# Primero 60% train y 40% temporal
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=42)

# Luego dividimos el 40% temporal a la mitad (20% val, 20% test)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

print(f"Tamaños - Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}\n")

# 4. Crear el modelo de Clasificación (Árbol de Decisiones)
modelo_arbol = DecisionTreeClassifier(max_depth=4, random_state=42)

# 5. Validación Cruzada (en el set de entrenamiento)
# Evalúa el modelo en 5 pliegues diferentes para asegurar que sea estable
scores_cv = cross_val_score(modelo_arbol, X_train, y_train, cv=5, scoring='accuracy')
print("--- Validación Cruzada ---")
print(f"Puntuaciones de cada pliegue (CV=5): {scores_cv}")
print(f"Precisión media (Cross-Validation): {scores_cv.mean():.4f}\n")

# 6. Entrenar el modelo
modelo_arbol.fit(X_train, y_train)

# 7. Evaluar en el set de Validación (Para ajustar hiperparámetros si se deseara)
y_pred_val = modelo_arbol.predict(X_val)
acc_val = accuracy_score(y_val, y_pred_val)
print("--- Set de Validación ---")
print(f"Precisión en validación: {acc_val:.4f}\n")

# 8. Evaluar el desempeño final en el set de Prueba (Datos nunca vistos)
y_pred_test = modelo_arbol.predict(X_test)

print("--- Set de Prueba (Evaluación Final) ---")
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred_test))

print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred_test))