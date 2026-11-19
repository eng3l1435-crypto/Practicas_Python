
# ============================================================
# PROYECTO: SMARTACADEMY
# ETAPA 17: COMPARACION DE MODELOS DE IA
#
# Modelos:
# 1. CNN 1D - PyTorch
# 2. Random Forest - Scikit-learn
#
# Objetivo:
# Comparar ambos modelos utilizando exactamente los mismos
# conjuntos de entrenamiento, validacion y prueba.
# ============================================================

from pathlib import Path
import json
import time
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# 1. CONFIGURACION DE RUTAS
# ============================================================

RUTA_PROYECTO = Path(__file__).resolve().parents[1]

RUTA_DATOS = RUTA_PROYECTO / "resultados_cnn"

RUTA_SALIDA = RUTA_PROYECTO / "resultados_comparacion"

RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

ARCHIVO_DATOS = RUTA_DATOS / "smartacademy_datos_cnn.npz"

ARCHIVO_METRICAS_CNN = RUTA_DATOS / "metricas_cnn.json"

ARCHIVO_MODELO_RF = RUTA_SALIDA / "modelo_random_forest.joblib"

# ============================================================
# 2. CONFIGURACION GENERAL
# ============================================================

NOMBRE_CLASES = [
    "Descendente",
    "Estable",
    "Favorable"
]

RANDOM_STATE = 42

# ============================================================
# 3. CARGAR LOS DATOS PREPARADOS
# ============================================================

print("=" * 65)
print("SMARTACADEMY")
print("ETAPA 17 - COMPARACION CNN 1D VS RANDOM FOREST")
print("=" * 65)

print("\nCARGANDO LOS DATOS DE ENTRENAMIENTO...")

if not ARCHIVO_DATOS.exists():
    raise FileNotFoundError(
        f"No se encontro el archivo: {ARCHIVO_DATOS}"
    )

# El archivo NPZ contiene los tensores y las divisiones
# creadas durante la etapa 15.
datos = np.load(ARCHIVO_DATOS)

# Recuperamos los mismos conjuntos utilizados por la CNN.
X_train = datos["X_train"]
X_val = datos["X_val"]
X_test = datos["X_test"]

y_train = datos["y_train"]
y_val = datos["y_val"]
y_test = datos["y_test"]

print("\nDATOS CARGADOS CORRECTAMENTE")

print(f"Entrenamiento: {X_train.shape}")
print(f"Validacion:    {X_val.shape}")
print(f"Prueba:        {X_test.shape}")

# ============================================================
# 4. TRANSFORMAR LOS DATOS PARA RANDOM FOREST
# ============================================================

# La CNN trabaja con dimensiones:
# (registros, canales, parciales)
#
# Ejemplo:
# (35000, 2, 3)
#
# Random Forest necesita una matriz bidimensional:
# (registros, caracteristicas)
#
# Por eso transformamos los datos de:
# (N, 2, 3)
# a:
# (N, 6)

print("\nTRANSFORMANDO LOS DATOS PARA RANDOM FOREST...")

X_train_rf = X_train.reshape(X_train.shape[0], -1)
X_val_rf = X_val.reshape(X_val.shape[0], -1)
X_test_rf = X_test.reshape(X_test.shape[0], -1)

print("Transformacion completada.")

print("Caracteristicas utilizadas:")
print("1. Calificacion parcial 1")
print("2. Calificacion parcial 2")
print("3. Calificacion parcial 3")
print("4. Asistencia parcial 1")
print("5. Asistencia parcial 2")
print("6. Asistencia parcial 3")

print(f"\nForma de entrenamiento: {X_train_rf.shape}")

# ============================================================
# 5. VALIDAR LOS CONJUNTOS
# ============================================================

print("\nVALIDANDO LOS DATOS...")

assert X_train_rf.shape[1] == 6
assert X_val_rf.shape[1] == 6
assert X_test_rf.shape[1] == 6

assert len(X_train_rf) == len(y_train)
assert len(X_val_rf) == len(y_val)
assert len(X_test_rf) == len(y_test)

assert set(np.unique(y_train)) == {0, 1, 2}
assert set(np.unique(y_val)) == {0, 1, 2}
assert set(np.unique(y_test)) == {0, 1, 2}

print("Validaciones correctas.")

# ============================================================
# 6. CREAR EL MODELO RANDOM FOREST
# ============================================================

print("\nCONFIGURANDO RANDOM FOREST...")

modelo_rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

print("Modelo configurado correctamente.")

print("Numero de arboles: 300")
print("Profundidad maxima: sin limite")
print("Semilla aleatoria: 42")

# ============================================================
# 7. ENTRENAR RANDOM FOREST
# ============================================================

print("\nINICIANDO ENTRENAMIENTO...")

inicio_entrenamiento = time.perf_counter()

modelo_rf.fit(X_train_rf, y_train)

tiempo_entrenamiento_rf = (
    time.perf_counter() - inicio_entrenamiento
)

print("\nENTRENAMIENTO FINALIZADO.")

print(
    f"Tiempo de entrenamiento: "
    f"{tiempo_entrenamiento_rf:.4f} segundos"
)

# ============================================================
# 8. VALIDACION DEL MODELO
# ============================================================

print("\nEVALUANDO CONJUNTO DE VALIDACION...")

predicciones_val = modelo_rf.predict(X_val_rf)

accuracy_val = accuracy_score(
    y_val,
    predicciones_val
)

f1_val = f1_score(
    y_val,
    predicciones_val,
    average="macro"
)

print(f"Accuracy validacion: {accuracy_val:.4f}")
print(f"F1 Macro validacion: {f1_val:.4f}")

# ============================================================
# 9. EVALUACION FINAL CON DATOS DE PRUEBA
# ============================================================

print("\nEVALUANDO CONJUNTO DE PRUEBA...")

inicio_prediccion = time.perf_counter()

predicciones_test = modelo_rf.predict(X_test_rf)

tiempo_prediccion_rf = (
    time.perf_counter() - inicio_prediccion
)

# Calculamos las metricas principales.
accuracy_rf = accuracy_score(
    y_test,
    predicciones_test
)

precision_rf = precision_score(
    y_test,
    predicciones_test,
    average="macro",
    zero_division=0
)

recall_rf = recall_score(
    y_test,
    predicciones_test,
    average="macro",
    zero_division=0
)

f1_rf = f1_score(
    y_test,
    predicciones_test,
    average="macro",
    zero_division=0
)

matriz_rf = confusion_matrix(
    y_test,
    predicciones_test,
    labels=[0, 1, 2]
)

# ============================================================
# 10. MOSTRAR RESULTADOS RANDOM FOREST
# ============================================================

print("\nRESULTADOS RANDOM FOREST")

print(f"Accuracy:  {accuracy_rf:.4f}")
print(f"Precision: {precision_rf:.4f}")
print(f"Recall:    {recall_rf:.4f}")
print(f"F1 Score:  {f1_rf:.4f}")

print("\nMATRIZ DE CONFUSION")

print(matriz_rf)

print("\nREPORTE DE CLASIFICACION")

reporte_rf = classification_report(
    y_test,
    predicciones_test,
    labels=[0, 1, 2],
    target_names=NOMBRE_CLASES,
    zero_division=0
)

print(reporte_rf)

# ============================================================
# 11. RECUPERAR RESULTADOS DE LA CNN
# ============================================================

print("\nRECUPERANDO RESULTADOS DE LA CNN...")

if not ARCHIVO_METRICAS_CNN.exists():
    raise FileNotFoundError(
        "No se encontro metricas_cnn.json. "
        "Ejecuta primero la etapa 16."
    )

with open(
    ARCHIVO_METRICAS_CNN,
    "r",
    encoding="utf-8"
) as archivo:

    metricas_cnn = json.load(archivo)

# Revisaremos las claves disponibles antes de utilizarlas.
print("\nClaves disponibles en el archivo de metricas:")

for clave in metricas_cnn.keys():
    print(f"- {clave}")


# ============================================================
# 12. NORMALIZAR LA LECTURA DE METRICAS CNN
# ============================================================

# En nuestro archivo JSON, las metricas de evaluacion
# estan almacenadas dentro del diccionario:
#
# "metricas_prueba"
#
# Primero recuperamos ese diccionario.

metricas_prueba_cnn = metricas_cnn["metricas_prueba"]

# Ahora extraemos cada metrica individualmente.

accuracy_cnn = float(
    metricas_prueba_cnn["accuracy"]
)

precision_cnn = float(
    metricas_prueba_cnn["precision_macro"]
)

recall_cnn = float(
    metricas_prueba_cnn["recall_macro"]
)

f1_cnn = float(
    metricas_prueba_cnn["f1_macro"]
)

print("\nMETRICAS CNN RECUPERADAS CORRECTAMENTE")

print(f"Accuracy:  {accuracy_cnn:.6f}")
print(f"Precision: {precision_cnn:.6f}")
print(f"Recall:    {recall_cnn:.6f}")
print(f"F1 Score:  {f1_cnn:.6f}")


# ============================================================
# 13. CONSTRUIR TABLA COMPARATIVA
# ============================================================

print("\nCONSTRUYENDO TABLA COMPARATIVA...")

comparacion = pd.DataFrame({

    "Modelo": [
        "CNN 1D",
        "Random Forest"
    ],

    "Accuracy": [
        accuracy_cnn,
        accuracy_rf
    ],

    "Precision Macro": [
        precision_cnn,
        precision_rf
    ],

    "Recall Macro": [
        recall_cnn,
        recall_rf
    ],

    "F1 Macro": [
        f1_cnn,
        f1_rf
    ]

})

print("\nRESULTADOS COMPARATIVOS")

print(
    comparacion.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# ============================================================
# 14. GUARDAR TABLA COMPARATIVA
# ============================================================

archivo_csv = RUTA_SALIDA / "comparacion_cnn_randomforest.csv"

comparacion.to_csv(
    archivo_csv,
    index=False,
    encoding="utf-8"
)

print(f"\nTabla guardada: {archivo_csv}")

# ============================================================
# 15. GUARDAR METRICAS EN JSON
# ============================================================

resultados_json = {

    "proyecto": "SmartAcademy",

    "etapa": 17,

    "modelos": {

        "CNN_1D": {

            "accuracy": accuracy_cnn,
            "precision_macro": precision_cnn,
            "recall_macro": recall_cnn,
            "f1_macro": f1_cnn

        },

        "Random_Forest": {

            "accuracy": accuracy_rf,
            "precision_macro": precision_rf,
            "recall_macro": recall_rf,
            "f1_macro": f1_rf,
            "tiempo_entrenamiento_segundos":
                tiempo_entrenamiento_rf,
            "tiempo_prediccion_segundos":
                tiempo_prediccion_rf

        }

    },

    "observacion": (
        "Ambos modelos utilizan los mismos conjuntos de datos. "
        "La CNN y Random Forest clasifican una tendencia derivada "
        "de las calificaciones observadas."
    )

}

archivo_json = RUTA_SALIDA / "metricas_comparacion.json"

with open(
    archivo_json,
    "w",
    encoding="utf-8"
) as archivo:

    json.dump(
        resultados_json,
        archivo,
        indent=4,
        ensure_ascii=False
    )

print(f"Metricas guardadas: {archivo_json}")

# ============================================================
# 16. GRAFICAR COMPARACION DE MODELOS
# ============================================================

print("\nGENERANDO GRAFICA COMPARATIVA...")

metricas = [
    "Accuracy",
    "Precision Macro",
    "Recall Macro",
    "F1 Macro"
]

posiciones = np.arange(len(metricas))

ancho = 0.35

fig, ax = plt.subplots(figsize=(11, 6))

ax.bar(
    posiciones - ancho / 2,
    comparacion.iloc[0, 1:].values,
    ancho,
    label="CNN 1D"
)

ax.bar(
    posiciones + ancho / 2,
    comparacion.iloc[1, 1:].values,
    ancho,
    label="Random Forest"
)

ax.set_title(
    "SmartAcademy: CNN 1D vs Random Forest"
)

ax.set_ylabel("Valor de la metrica")

ax.set_xlabel("Metricas de evaluacion")

ax.set_xticks(posiciones)

ax.set_xticklabels(metricas)

ax.set_ylim(0, 1.05)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

archivo_grafica = RUTA_SALIDA / "comparacion_modelos.png"

plt.savefig(
    archivo_grafica,
    dpi=300
)

plt.close()

print(f"Grafica guardada: {archivo_grafica}")

# ============================================================
# 17. GUARDAR MODELO RANDOM FOREST
# ============================================================

print("\nGUARDANDO MODELO RANDOM FOREST...")

joblib.dump(
    modelo_rf,
    ARCHIVO_MODELO_RF
)

print(f"Modelo guardado: {ARCHIVO_MODELO_RF}")

# ============================================================
# 18. CONCLUSION AUTOMATICA
# ============================================================

print("\n" + "=" * 65)

print("CONCLUSION DE LA COMPARACION")

print("=" * 65)

if accuracy_cnn > accuracy_rf:

    print("CNN 1D obtuvo mayor accuracy.")

elif accuracy_rf > accuracy_cnn:

    print("Random Forest obtuvo mayor accuracy.")

else:

    print("Ambos modelos obtuvieron el mismo accuracy.")

if f1_cnn > f1_rf:

    print("CNN 1D obtuvo mayor F1 Macro.")

elif f1_rf > f1_cnn:

    print("Random Forest obtuvo mayor F1 Macro.")

else:

    print("Ambos modelos obtuvieron el mismo F1 Macro.")

print("\nIMPORTANTE:")

print(
    "Una metrica elevada no demuestra por si sola "
    "capacidad de predecir el rendimiento futuro."
)

print(
    "La etiqueta de tendencia se deriva de las "
    "calificaciones utilizadas como entradas."
)

print("\nETAPA 17 FINALIZADA")

print("=" * 65)
