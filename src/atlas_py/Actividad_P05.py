# ================================================================
# PROYECTO: SMARTACADEMY
# ACTIVIDAD PRACTICA 05: EXPERIMENTOS DE HIPERPARAMETROS
#
# Archivo:
# Actividad_P05.py
#
# Objetivo:
# Realizar tres experimentos modificando hiperparametros clave
# en los modelos de clasificacion, registrar los resultados
# y analizar como cambia el rendimiento:
#
# - Experimento A: modificar n_neighbors de KNN a 3, 5, 7, 9 y 15.
# - Experimento B: modificar n_estimators de Random Forest a 50, 100, 200 y 500.
# - Experimento C: modificar max_depth de Decision Tree a 2, 3, 5, 8 y None.
#
# IMPORTANTE:
# Los datos de esta practica son sinteticos y se utilizan
# exclusivamente con fines educativos.
# ================================================================


# ================================================================
# 1. IMPORTAR LAS BIBLIOTECAS
# ================================================================

# NumPy permite realizar operaciones matematicas y generar datos.
import numpy as np

# Pandas permite organizar los datos en tablas.
import pandas as pd

# Matplotlib permite crear graficas.
import matplotlib.pyplot as plt

# train_test_split divide los datos en entrenamiento,
# validacion y prueba.
from sklearn.model_selection import train_test_split

# DecisionTreeClassifier implementa arboles de decision.
from sklearn.tree import DecisionTreeClassifier

# RandomForestClassifier implementa bosques aleatorios.
from sklearn.ensemble import RandomForestClassifier

# KNeighborsClassifier implementa el algoritmo KNN.
from sklearn.neighbors import KNeighborsClassifier

# StandardScaler normaliza las variables numericas.
from sklearn.preprocessing import StandardScaler

# Pipeline permite unir el escalamiento y el algoritmo.
from sklearn.pipeline import Pipeline

# Metricas para evaluar los modelos.
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ================================================================
# 2. CONFIGURACION GENERAL
# ================================================================

# Fijamos la misma semilla de la Practica 05 para mantener
# exactamente el mismo conjunto de datos y particiones.
SEMILLA = 42

# Generador de numeros aleatorios.
rng = np.random.default_rng(SEMILLA)

# Cantidad de estudiantes simulados.
CANTIDAD_ESTUDIANTES = 500

print("\n==========================================")
print("SMARTACADEMY")
print("ACTIVIDAD P05: EXPERIMENTOS DE AJUSTE")
print("==========================================")


# ================================================================
# 3. GENERACION DEL CONJUNTO DE DATOS
# ================================================================

# Generamos las horas de estudio (0 a 20 horas).
horas_estudio = rng.uniform(
    0,
    20,
    CANTIDAD_ESTUDIANTES
)

# Generamos el porcentaje de asistencia (40% a 100%).
asistencia = rng.uniform(
    40,
    100,
    CANTIDAD_ESTUDIANTES
)

# Generamos el porcentaje de actividades entregadas (0% a 100%).
actividades = rng.uniform(
    0,
    100,
    CANTIDAD_ESTUDIANTES
)

# Generamos el nivel de participacion (0 a 10).
participacion = rng.uniform(
    0,
    10,
    CANTIDAD_ESTUDIANTES
)

# Construimos el indicador sintetico de rendimiento.
puntaje = (
    (horas_estudio / 20) * 0.30
    +
    (asistencia / 100) * 0.25
    +
    (actividades / 100) * 0.30
    +
    (participacion / 10) * 0.15
)

# Agregamos ruido aleatorio.
ruido = rng.normal(
    loc=0,
    scale=0.12,
    size=CANTIDAD_ESTUDIANTES
)

# Calculamos el puntaje final y la variable binaria aprueba.
puntaje_final = puntaje + ruido
aprueba = (puntaje_final >= 0.58).astype(int)


# ================================================================
# 4. CONSTRUIR EL DATAFRAME Y SEPARAR VARIABLES
# ================================================================

df = pd.DataFrame({
    "horas_estudio": horas_estudio,
    "asistencia": asistencia,
    "actividades": actividades,
    "participacion": participacion,
    "aprueba": aprueba
})

X = df[
    [
        "horas_estudio",
        "asistencia",
        "actividades",
        "participacion"
    ]
]

y = df["aprueba"]


# ================================================================
# 5. DIVIDIR LOS DATOS (60% ENTRENAMIENTO, 20% VALIDACION, 20% PRUEBA)
# ================================================================

X_entrenamiento_validacion, X_prueba, \
y_entrenamiento_validacion, y_prueba = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEMILLA,
    stratify=y
)

X_entrenamiento, X_validacion, \
y_entrenamiento, y_validacion = train_test_split(
    X_entrenamiento_validacion,
    y_entrenamiento_validacion,
    test_size=0.25,
    random_state=SEMILLA,
    stratify=y_entrenamiento_validacion
)

print("\nDistribucion de los conjuntos:")
print("Entrenamiento:", len(X_entrenamiento))
print("Validacion:", len(X_validacion))
print("Prueba:", len(X_prueba))


# ================================================================
# FUNCION AUXILIAR PARA EVALUAR CADA CONFIGURACION
# ================================================================

def evaluar_configuracion(modelo, nombre_parametro, valor_parametro):
    """
    Entrena un modelo con el conjunto de entrenamiento y calcula
    las metricas de desempeno tanto en entrenamiento como en validacion.
    """
    modelo.fit(X_entrenamiento, y_entrenamiento)

    pred_entrenamiento = modelo.predict(X_entrenamiento)
    pred_validacion = modelo.predict(X_validacion)

    exactitud_ent = accuracy_score(y_entrenamiento, pred_entrenamiento)
    exactitud_val = accuracy_score(y_validacion, pred_validacion)
    precision_val = precision_score(y_validacion, pred_validacion, zero_division=0)
    recall_val = recall_score(y_validacion, pred_validacion, zero_division=0)
    f1_val = f1_score(y_validacion, pred_validacion, zero_division=0)

    return {
        nombre_parametro: str(valor_parametro),
        "exactitud_entrenamiento": exactitud_ent,
        "exactitud_validacion": exactitud_val,
        "precision": precision_val,
        "recall": recall_val,
        "f1_score": f1_val
    }


def mostrar_tabla_resultados(df_exp, titulo):
    """
    Imprime en consola la tabla formateada en porcentajes.
    """
    print("\n------------------------------------------")
    print(titulo)
    print("------------------------------------------")
    print(
        df_exp.to_string(
            index=False,
            formatters={
                "exactitud_entrenamiento": "{:.2%}".format,
                "exactitud_validacion": "{:.2%}".format,
                "precision": "{:.2%}".format,
                "recall": "{:.2%}".format,
                "f1_score": "{:.2%}".format
            }
        )
    )


# ================================================================
# 6. EXPERIMENTO A: MODIFICAR n_neighbors EN KNN
# ================================================================

print("\n==========================================")
print("EXPERIMENTO A: K-NEAREST NEIGHBORS (KNN)")
print("Valores de n_neighbors: [3, 5, 7, 9, 15]")
print("==========================================")

valores_k = [3, 5, 7, 9, 15]
resultados_exp_a = []

for k in valores_k:

    # Mantenemos el Pipeline con StandardScaler igual que en P05.
    modelo_knn = Pipeline([
        ("escalador", StandardScaler()),
        ("clasificador", KNeighborsClassifier(
            n_neighbors=k,
            weights="distance"
        ))
    ])

    registro = evaluar_configuracion(
        modelo_knn,
        "n_neighbors",
        k
    )
    resultados_exp_a.append(registro)

df_exp_a = pd.DataFrame(resultados_exp_a)
mostrar_tabla_resultados(
    df_exp_a,
    "RESULTADOS EXPERIMENTO A (KNN)"
)


# ================================================================
# 7. EXPERIMENTO B: MODIFICAR n_estimators EN RANDOM FOREST
# ================================================================

print("\n==========================================")
print("EXPERIMENTO B: RANDOM FOREST")
print("Valores de n_estimators: [50, 100, 200, 500]")
print("==========================================")

valores_estimadores = [50, 100, 200, 500]
resultados_exp_b = []

for n_est in valores_estimadores:

    modelo_rf = RandomForestClassifier(
        n_estimators=n_est,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=SEMILLA
    )

    registro = evaluar_configuracion(
        modelo_rf,
        "n_estimators",
        n_est
    )
    resultados_exp_b.append(registro)

df_exp_b = pd.DataFrame(resultados_exp_b)
mostrar_tabla_resultados(
    df_exp_b,
    "RESULTADOS EXPERIMENTO B (RANDOM FOREST)"
)


# ================================================================
# 8. EXPERIMENTO C: MODIFICAR max_depth EN DECISION TREE
# ================================================================

print("\n==========================================")
print("EXPERIMENTO C: DECISION TREE")
print("Valores de max_depth: [2, 3, 5, 8, None]")
print("==========================================")

valores_profundidad = [2, 3, 5, 8, None]
resultados_exp_c = []

for profundidad in valores_profundidad:

    modelo_dt = DecisionTreeClassifier(
        max_depth=profundidad,
        min_samples_leaf=4,
        random_state=SEMILLA
    )

    etiqueta_prof = "None" if profundidad is None else profundidad

    registro = evaluar_configuracion(
        modelo_dt,
        "max_depth",
        etiqueta_prof
    )
    resultados_exp_c.append(registro)

df_exp_c = pd.DataFrame(resultados_exp_c)
mostrar_tabla_resultados(
    df_exp_c,
    "RESULTADOS EXPERIMENTO C (DECISION TREE)"
)


# ================================================================
# 9. EXPORTAR TABLAS DE RESULTADOS A CSV
# ================================================================

df_exp_a.to_csv("actividad_p05_expA_knn.csv", index=False, encoding="utf-8")
df_exp_b.to_csv("actividad_p05_expB_rf.csv", index=False, encoding="utf-8")
df_exp_c.to_csv("actividad_p05_expC_dt.csv", index=False, encoding="utf-8")


# ================================================================
# 10. GRAFICA COMPARATIVA DE LOS TRES EXPERIMENTOS
# ================================================================

fig, ejes = plt.subplots(1, 3, figsize=(16, 5))

# --- Subgrafica A: KNN ---
ejes[0].plot(
    df_exp_a["n_neighbors"],
    df_exp_a["exactitud_validacion"],
    marker="o",
    linewidth=2,
    label="Exactitud (Val)"
)
ejes[0].plot(
    df_exp_a["n_neighbors"],
    df_exp_a["f1_score"],
    marker="s",
    linewidth=2,
    label="F1-score (Val)"
)
ejes[0].set_title("Experimento A: KNN (n_neighbors)")
ejes[0].set_xlabel("n_neighbors")
ejes[0].set_ylabel("Rendimiento")
ejes[0].set_ylim(0.5, 1.05)
ejes[0].grid(True, linestyle="--", alpha=0.4)
ejes[0].legend()

# --- Subgrafica B: Random Forest ---
ejes[1].plot(
    df_exp_b["n_estimators"],
    df_exp_b["exactitud_validacion"],
    marker="o",
    linewidth=2,
    label="Exactitud (Val)"
)
ejes[1].plot(
    df_exp_b["n_estimators"],
    df_exp_b["f1_score"],
    marker="s",
    linewidth=2,
    label="F1-score (Val)"
)
ejes[1].set_title("Experimento B: Random Forest (n_estimators)")
ejes[1].set_xlabel("n_estimators")
ejes[1].set_ylim(0.5, 1.05)
ejes[1].grid(True, linestyle="--", alpha=0.4)
ejes[1].legend()

# --- Subgrafica C: Decision Tree ---
ejes[2].plot(
    df_exp_c["max_depth"],
    df_exp_c["exactitud_entrenamiento"],
    marker="^",
    linestyle="--",
    linewidth=2,
    label="Exactitud (Entrenamiento)"
)
ejes[2].plot(
    df_exp_c["max_depth"],
    df_exp_c["exactitud_validacion"],
    marker="o",
    linewidth=2,
    label="Exactitud (Validacion)"
)
ejes[2].plot(
    df_exp_c["max_depth"],
    df_exp_c["f1_score"],
    marker="s",
    linewidth=2,
    label="F1-score (Validacion)"
)
ejes[2].set_title("Experimento C: Decision Tree (max_depth)")
ejes[2].set_xlabel("max_depth")
ejes[2].set_ylim(0.5, 1.05)
ejes[2].grid(True, linestyle="--", alpha=0.4)
ejes[2].legend()

plt.tight_layout()
plt.savefig("actividad_p05_grafica_experimentos.png", dpi=300)
plt.show()


# ================================================================
# 11. EXPLICACION Y CONCLUSIONES DE LOS EXPERIMENTOS
# ================================================================

print("\n==========================================")
print("EXPLICACION Y CONCLUSIONES DE RESULTADOS")
print("==========================================")

# Identificamos automaticamente la mejor configuracion de cada experimento
# segun el F1-score obtenido en validacion.
mejor_a = df_exp_a.loc[df_exp_a["f1_score"].idxmax()]
mejor_b = df_exp_b.loc[df_exp_b["f1_score"].idxmax()]
mejor_c = df_exp_c.loc[df_exp_c["f1_score"].idxmax()]

print(
    f"\n1. Experimento A (KNN): El mejor F1-score en validacion "
    f"({mejor_a['f1_score']:.2%}) se obtuvo con n_neighbors = {mejor_a['n_neighbors']}.\n"
    "   Explicacion: Con valores muy pequenos de k (como k=3), el modelo es mas\n"
    "   sensible al ruido individual de los vecinos mas cercanos. Al aumentar k\n"
    "   a valores intermedios (7 o 9), la frontera de decision se suaviza y mejora\n"
    "   la generalizacion; si k crece demasiado (15), puede perder detalle local."
)

print(
    f"\n2. Experimento B (Random Forest): El mejor F1-score en validacion "
    f"({mejor_b['f1_score']:.2%}) se obtuvo con n_estimators = {mejor_b['n_estimators']}.\n"
    "   Explicacion: Aumentar la cantidad de arboles estabiliza las predicciones\n"
    "   al promediar mas votos independientes, reduciendo la varianza. Sin embargo,\n"
    "   a partir de cierto punto (por ejemplo entre 100, 200 y 500 arboles), la\n"
    "   ganancia en exactitud y F1-score se estabiliza mientras aumenta el costo computacional."
)

print(
    f"\n3. Experimento C (Decision Tree): El mejor F1-score en validacion "
    f"({mejor_c['f1_score']:.2%}) se obtuvo con max_depth = {mejor_c['max_depth']}.\n"
    "   Explicacion: Con profundidad muy baja (max_depth=2) el arbol es demasiado\n"
    "   simple y presenta subajuste (underfitting). Con profundidades intermedias (3 o 5)\n"
    "   logra capturar las reglas principales sin memorizar el ruido. Cuando se aumenta\n"
    "   demasiado (8 o None), la exactitud en entrenamiento sube pero en validacion\n"
    "   se estanca o cae, evidenciando tendencia al sobreajuste (overfitting)."
)

print("\nArchivos generados:")
print("- actividad_p05_expA_knn.csv")
print("- actividad_p05_expB_rf.csv")
print("- actividad_p05_expC_dt.csv")
print("- actividad_p05_grafica_experimentos.png")
print("\nACTIVIDAD P05 FINALIZADA.")