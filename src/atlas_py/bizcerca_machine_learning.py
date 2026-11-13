import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.base import clone

os.makedirs("resultados_bizcerca", exist_ok=True)

# ------------------------------------------------------------
# 1. GENERACION Y ENSUCIADO DE DATOS 
# ------------------------------------------------------------
np.random.seed(42)
cantidad = 500
df = pd.DataFrame({
    "similitud_nlp": np.random.uniform(0.1, 1.0, cantidad),
    "distancia_km": np.random.uniform(0.5, 25.0, cantidad),
    "dias_inactividad": np.random.uniform(0, 45, cantidad),
    "reportes": np.random.uniform(0, 5, cantidad)
})

puntaje = (df["similitud_nlp"] * 0.4) + ((25 - df["distancia_km"])/25 * 0.3) + ((45 - df["dias_inactividad"])/45 * 0.2) - (df["reportes"] * 0.1)
df["match_perfecto"] = ((puntaje + np.random.normal(0, 0.1, cantidad)) >= 0.55).astype(int)

# Inyectando ruido
df.loc[5:15, "distancia_km"] = np.nan
df.loc[250, "distancia_km"] = -50.0  
df.loc[300, "reportes"] = 5000       
df = pd.concat([df, df.iloc[0:15]], ignore_index=True)

print("\n==========================================")
print("1. DATOS ORIGINALES BIZCERCA")
print("==========================================")
print(f"Registros iniciales: {len(df)}")
print("Nulos detectados:\n", df.isnull().sum())

# ------------------------------------------------------------
# 2. LIMPIEZA DE DATOS (DATA CLEANING)
# ------------------------------------------------------------
# A. Duplicados
df = df.drop_duplicates().reset_index(drop=True)
# B. Nulos
df = df.dropna().reset_index(drop=True)
# C. Atipicos / Logica de Negocio
df = df[
    (df["distancia_km"] >= 0) & 
    (df["reportes"] >= 0) & (df["reportes"] <= 50) &
    (df["dias_inactividad"] >= 0)
].reset_index(drop=True)

print("\n==========================================")
print("2. RESULTADO DE LIMPIEZA")
print("==========================================")
print(f"Registros despues de limpieza: {len(df)}")

# ------------------------------------------------------------
# 3. DIVISION DE DATOS (60/20/20)
# ------------------------------------------------------------
X = df.drop("match_perfecto", axis=1)
y = df["match_perfecto"]

X_ent_val, X_prueba, y_ent_val, y_prueba = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
X_entrenamiento, X_validacion, y_entrenamiento, y_validacion = train_test_split(X_ent_val, y_ent_val, test_size=0.25, random_state=42, stratify=y_ent_val)

# ------------------------------------------------------------
# 4. ALGORITMOS A COMPARAR
# ------------------------------------------------------------
modelos = {
    "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=4, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    "KNN": Pipeline([
        ("escalador", StandardScaler()),
        ("clasificador", KNeighborsClassifier(n_neighbors=7, weights="distance"))
    ])
}

print("\n==========================================")
print("3. COMPARACION DE ALGORITMOS DE SMART MATCH")
print("==========================================")
resultados = []
modelos_entrenados = {}

for nombre, modelo in modelos.items():
    print(f"\nEntrenando: {nombre}...")
    modelo.fit(X_entrenamiento, y_entrenamiento)
    modelos_entrenados[nombre] = modelo
    
    preds = modelo.predict(X_validacion)
    resultados.append({
        "algoritmo": nombre,
        "exactitud": accuracy_score(y_validacion, preds),
        "precision": precision_score(y_validacion, preds, zero_division=0),
        "recall": recall_score(y_validacion, preds, zero_division=0),
        "f1_score": f1_score(y_validacion, preds, zero_division=0)
    })

df_resultados = pd.DataFrame(resultados).sort_values(by="f1_score", ascending=False).reset_index(drop=True)
mejor_nombre = df_resultados.iloc[0]["algoritmo"]

print("\n==========================================")
print("4. TABLA COMPARATIVA")
print("==========================================")
print(df_resultados.to_string(index=False, formatters={"exactitud": "{:.2%}".format, "precision": "{:.2%}".format, "recall": "{:.2%}".format, "f1_score": "{:.2%}".format}))

print("\n==========================================")
print("5. MODELO SELECCIONADO")
print("==========================================")
print(f"Algoritmo Ganador: {mejor_nombre}")

# ------------------------------------------------------------
# 5. REENTRENAMIENTO Y EVALUACION FINAL
# ------------------------------------------------------------
modelo_final = clone(modelos[mejor_nombre])
modelo_final.fit(X_ent_val, y_ent_val)
preds_finales = modelo_final.predict(X_prueba)

f1_final = f1_score(y_prueba, preds_finales, zero_division=0)

print("\n==========================================")
print("6. EVALUACION FINAL EN PRUEBA")
print("==========================================")
print(f"Exactitud: {accuracy_score(y_prueba, preds_finales):.2%}")
print(f"Precision: {precision_score(y_prueba, preds_finales, zero_division=0):.2%}")
print(f"Recall: {recall_score(y_prueba, preds_finales, zero_division=0):.2%}")
print(f"F1-score: {f1_final:.2%}")

print("\n==========================================")
print("7. REPORTE Y MATRIZ DE CONFUSION")
print("==========================================")
print(classification_report(y_prueba, preds_finales, target_names=["Descartado", "Match Perfecto"]))
matriz_final = confusion_matrix(y_prueba, preds_finales)
print(matriz_final)

# ------------------------------------------------------------
# GRAFICAS DE COMPARACION
# ------------------------------------------------------------
plt.figure(figsize=(12, 6))
ancho = 0.22
posiciones = np.arange(len(df_resultados))
metricas = ["exactitud", "precision", "recall", "f1_score"]

for i, metrica in enumerate(metricas):
    desplazamiento = (i - 1.5) * ancho
    plt.bar(posiciones + desplazamiento, df_resultados[metrica], width=ancho, label=metrica.replace("_", " ").title())

plt.xticks(posiciones, df_resultados["algoritmo"])
plt.ylim(0, 1.05)
plt.ylabel("Resultado")
plt.xlabel("Algoritmo")
plt.title("Comparacion de algoritmos de clasificacion (BizCerca)")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("resultados_bizcerca/p05_comparacion_algoritmos.png", dpi=300)
plt.show()

visualizacion_final = ConfusionMatrixDisplay(confusion_matrix=matriz_final, display_labels=["Descartado", "Match"])
visualizacion_final.plot(cmap="Blues", values_format="d")
plt.title("Matriz de confusion - " + mejor_nombre)
plt.tight_layout()
plt.savefig("resultados_bizcerca/p05_matriz_confusion.png", dpi=300)
plt.show()

print("\n==========================================")
print("8. VALIDACION CRUZADA")
print("==========================================")
resultados_cv = cross_val_score(clone(modelos[mejor_nombre]), X_ent_val, y_ent_val, cv=5, scoring="f1", n_jobs=-1)
print(f"Promedio F1: {resultados_cv.mean():.2%} (Desviacion: {resultados_cv.std():.2%})")

print("\nPROCESO FINALIZADO.")
