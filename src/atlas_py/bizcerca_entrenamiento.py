import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

os.makedirs("resultados_bizcerca", exist_ok=True)

# ------------------------------------------------------------
# 1. GENERACION DE DATOS SINTETICOS
# ------------------------------------------------------------
np.random.seed(42)
cantidad_oferentes = 500

similitud_nlp = np.random.uniform(0.1, 1.0, cantidad_oferentes)
distancia_km = np.random.uniform(0.5, 25.0, cantidad_oferentes)
dias_inactividad = np.random.uniform(0, 45, cantidad_oferentes)
reportes = np.random.uniform(0, 5, cantidad_oferentes)

ruido = np.random.normal(0, 0.1, cantidad_oferentes)
puntaje = (similitud_nlp * 0.4) + ((25 - distancia_km)/25 * 0.3) + ((45 - dias_inactividad)/45 * 0.2) - (reportes * 0.1) + ruido
match_perfecto = (puntaje >= 0.55).astype(int)

df = pd.DataFrame({
    "similitud_nlp": similitud_nlp,
    "distancia_km": distancia_km,
    "dias_inactividad": dias_inactividad,
    "reportes": reportes,
    "match_perfecto": match_perfecto
})

# Introducir errores intencionales (Simulacion de mundo real)
df.loc[10:25, "distancia_km"] = np.nan        # Fallo de GPS (Nulos)
df.loc[40:50, "similitud_nlp"] = np.nan       # Error de NLP (Nulos)
df.loc[100, "distancia_km"] = -15.0           # Distancia negativa (Outlier)
df.loc[101, "reportes"] = 999                 # Ataque de bots (Outlier)
df.loc[102, "dias_inactividad"] = -5          # Error de fecha (Outlier)
df = pd.concat([df, df.iloc[0:10]], ignore_index=True) # Registros duplicados

print("\n==========================================")
print("1. DATOS DE BIZCERCA (CON ERRORES)")
print("==========================================")
print(f"Registros iniciales: {len(df)}")
print("Valores nulos detectados:\n", df.isnull().sum())

# ------------------------------------------------------------
# 2. LIMPIEZA DE DATOS (DATA CLEANING)
# ------------------------------------------------------------
print("\n==========================================")
print("2. PROCESO DE LIMPIEZA DE DATOS")
print("==========================================")

# A. Eliminar duplicados exactos
df = df.drop_duplicates().reset_index(drop=True)

# B. Tratar valores nulos (Eliminacion de filas incompletas)
df = df.dropna().reset_index(drop=True)

# C. Filtrar valores ilogicos o atipicos (Outliers)
df = df[
    (df["distancia_km"] >= 0) & 
    (df["reportes"] >= 0) & (df["reportes"] <= 50) &
    (df["dias_inactividad"] >= 0)
].reset_index(drop=True)

print(f"Registros despues de limpieza: {len(df)}")
print("Valores nulos restantes:\n", df.isnull().sum())
print("\nDistribucion limpia de las clases:")
print(df["match_perfecto"].value_counts())

# ------------------------------------------------------------
# 3. SEPARAR CONJUNTOS (60% Ent, 20% Val, 20% Prueba)
# ------------------------------------------------------------
X = df[["similitud_nlp", "distancia_km", "dias_inactividad", "reportes"]]
y = df["match_perfecto"]

X_entrenamiento, X_temporal, y_entrenamiento, y_temporal = train_test_split(X, y, test_size=0.40, random_state=42, stratify=y)
X_validacion, X_prueba, y_validacion, y_prueba = train_test_split(X_temporal, y_temporal, test_size=0.50, random_state=42, stratify=y_temporal)

print("\n==========================================")
print("3. DISTRIBUCION DE LOS CONJUNTOS")
print("==========================================")
print("Entrenamiento:", len(X_entrenamiento))
print("Validacion:", len(X_validacion))
print("Prueba:", len(X_prueba))

# ------------------------------------------------------------
# 4. ENTRENAR MODELOS Y ANALIZAR AJUSTE
# ------------------------------------------------------------
profundidades = [1, 2, 3, 5, 10, None]
resultados = []
modelos = {}

for profundidad in profundidades:
    print(f"\n--------------------------------------\nEntrenando modelo: {profundidad}\n--------------------------------------")
    modelo = DecisionTreeClassifier(max_depth=profundidad, random_state=42)
    modelo.fit(X_entrenamiento, y_entrenamiento)
    
    acc_ent = accuracy_score(y_entrenamiento, modelo.predict(X_entrenamiento))
    acc_val = accuracy_score(y_validacion, modelo.predict(X_validacion))
    
    modelos[profundidad] = modelo
    resultados.append({
        "profundidad": str(profundidad) if profundidad else "Sin limite",
        "profundidad_grafica": 12 if profundidad is None else profundidad,
        "entrenamiento": acc_ent,
        "validacion": acc_val
    })

df_resultados = pd.DataFrame(resultados)
print("\n==========================================")
print("4. COMPARACION DE MODELOS")
print("==========================================")
print(df_resultados[["profundidad", "entrenamiento", "validacion"]].to_string(index=False))

print("\n==========================================")
print("5. ANALISIS DE AJUSTE")
print("==========================================")
for _, fila in df_resultados.iterrows():
    diferencia = fila["entrenamiento"] - fila["validacion"]
    print(f"\nModelo: {fila['profundidad']}")
    print(f"Exactitud entrenamiento: {fila['entrenamiento']:.2%}")
    print(f"Exactitud validacion: {fila['validacion']:.2%}")
    print(f"Diferencia: {diferencia:.2%}")
    
    if fila["entrenamiento"] < 0.70 and fila["validacion"] < 0.70:
        print("Posible subajuste.")
    elif diferencia > 0.10:
        print("Posible sobreajuste.")
    else:
        print("No se observa una brecha grande.")

# ------------------------------------------------------------
# 5. SELECCION Y EVALUACION FINAL
# ------------------------------------------------------------
indice_mejor = df_resultados["validacion"].idxmax()
profundidad_optima = profundidades[indice_mejor]
mejor_modelo = modelos[profundidad_optima]

print("\n==========================================")
print("6. MODELO SELECCIONADO")
print("==========================================")
print(f"Profundidad seleccionada: {profundidad_optima}")
print(f"Exactitud de validacion: {df_resultados.loc[indice_mejor, 'validacion']:.2%}")

pred_prueba = mejor_modelo.predict(X_prueba)
print("\n==========================================")
print("7. EVALUACION FINAL")
print("==========================================")
print(f"Exactitud: {accuracy_score(y_prueba, pred_prueba):.2%}")
print(f"Precision: {precision_score(y_prueba, pred_prueba, zero_division=0):.2%}")
print(f"Recall: {recall_score(y_prueba, pred_prueba, zero_division=0):.2%}")
print(f"F1-score: {f1_score(y_prueba, pred_prueba, zero_division=0):.2%}")

print("\n==========================================")
print("8. REPORTE DE CLASIFICACION")
print("==========================================")
print(classification_report(y_prueba, pred_prueba, target_names=["Descartado", "Match Perfecto"]))

print("\n==========================================")
print("9. MATRIZ DE CONFUSION")
print("==========================================")
matriz = confusion_matrix(y_prueba, pred_prueba)
print(matriz)

# ------------------------------------------------------------
# GRAFICAS DE ENTRENAMIENTO
# ------------------------------------------------------------
plt.figure(figsize=(11, 6))
x = df_resultados["profundidad_grafica"]
plt.plot(x, df_resultados["entrenamiento"], marker="o", linewidth=2, label="Entrenamiento")
plt.plot(x, df_resultados["validacion"], marker="s", linewidth=2, label="Validacion")
plt.xticks(x, df_resultados["profundidad"])
plt.title("Comparacion de Modelos (Smart Match BizCerca)")
plt.xlabel("Profundidad del arbol")
plt.ylabel("Exactitud")
plt.ylim(0, 1.05)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("resultados_bizcerca/p04_comparacion_modelos.png", dpi=300)
plt.show()

visualizacion = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=["Descartado", "Match"])
visualizacion.plot(cmap="Blues", values_format="d")
plt.title(f"Matriz de confusion (Profundidad {profundidad_optima})")
plt.tight_layout()
plt.savefig("resultados_bizcerca/p04_matriz_confusion.png", dpi=300)
plt.show()

print("\n==========================================")
print("10. VALIDACION CRUZADA")
print("==========================================")
modelo_cv = DecisionTreeClassifier(max_depth=profundidad_optima, random_state=42)
puntuaciones_cv = cross_val_score(modelo_cv, X_entrenamiento, y_entrenamiento, cv=5, scoring="accuracy")
print("Resultados de cada particion:")
print(puntuaciones_cv)
print(f"Promedio: {puntuaciones_cv.mean():.2%}")
print(f"Desviacion estandar: {puntuaciones_cv.std():.2%}")

print("\n==========================================")
print("11. CONCLUSIONES")
print("==========================================")
print("El entrenamiento permite aprender patrones a partir de ejemplos limpios.")
print("Una correcta limpieza de datos previene sesgos provocados por outliers.")
print("La prueba permite estimar el comportamiento del modelo en registros no utilizados anteriormente.")
print("\nPRACTICA 04 FINALIZADA.")
