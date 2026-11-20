# ============================================================
# PROYECTO: SMARTACADEMY
# PRÁCTICA 10 - ETAPA 2
# ARCHIVO: p11modelo_supervisado.py
#
# OBJETIVO:
# Al finalizar esta etapa, se podrá:
#1. Separar características predictoras y variable objetivo.
#2. Dividir datos en entrenamiento y prueba.
#3. Entrenar modelos de clasificación.
#4. Comparar Decision Tree y Random Forest.
#5. Evaluar los resultados mediante métricas de clasificación.
#6. Seleccionar el modelo con mejor desempeño.
#7. Guardar el modelo entrenado mediante Joblib.
#8. Recuperar el modelo para realizar predicciones futuras.
#
# ALGORITMOS:
# 1. Decision Tree.
# 2. Random Forest.
#
# MÉTRICAS:
# Accuracy, Precision, Recall y F1-score.
#
# PERSISTENCIA:
# Joblib.
# ============================================================


# ------------------------------------------------------------
# 1. IMPORTACIÓN DE LIBRERÍAS
# ------------------------------------------------------------

# Pandas permite trabajar con tablas de datos.
import pandas as pd

# NumPy permite realizar operaciones numéricas.
import numpy as np

# Path permite construir rutas de archivos.
from pathlib import Path

# Joblib permite guardar y recuperar modelos entrenados.
import joblib

# Matplotlib permite crear gráficas.
import matplotlib.pyplot as plt

# train_test_split permite dividir los datos.
from sklearn.model_selection import train_test_split

# DecisionTreeClassifier implementa árboles de decisión.
from sklearn.tree import DecisionTreeClassifier

# RandomForestClassifier implementa bosques aleatorios.
from sklearn.ensemble import RandomForestClassifier

# Métricas de evaluación.
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


# ------------------------------------------------------------
# 2. CONFIGURACIÓN GENERAL
# ------------------------------------------------------------

# Definimos la ubicación del proyecto.

RUTA_BASE = Path("unidad3_fia")

# Ubicación del dataset generado en la etapa 1.

ARCHIVO_DATOS = (
    RUTA_BASE /
    "resultados_p10" /
    "smartacademy_datos.csv"
)

# Carpeta donde almacenaremos los modelos.

RUTA_MODELOS = (
    RUTA_BASE /
    "resultados_p10" /
    "modelos"
)

# Creamos la carpeta si no existe.

RUTA_MODELOS.mkdir(
    parents=True,
    exist_ok=True
)

# Definimos las características que utilizarán
# ambos algoritmos.

CARACTERISTICAS = [

    "horas_estudio",

    "asistencia",

    "actividades",

    "participacion"

]

# Nombre de la variable que queremos predecir.

VARIABLE_OBJETIVO = "aprueba"

# Semilla para reproducibilidad.

SEMILLA = 42

# ------------------------------------------------------------
# 3. CARGA DEL CONJUNTO DE DATOS
# ------------------------------------------------------------

def cargar_datos():

    """
    Carga el CSV generado en la etapa 1.

    Verifica:
    - Existencia del archivo.
    - Columnas necesarias.
    - Valores nulos.
    - Valores numéricos.
    - Variable objetivo.
    """

    print("\nCARGANDO DATASET SMARTACADEMY...")

    # --------------------------------------------------------
    # 3.1 Verificar existencia del archivo
    # --------------------------------------------------------

    if not ARCHIVO_DATOS.exists():

        raise FileNotFoundError(
            f"No se encontró el archivo: {ARCHIVO_DATOS}"
        )

    # --------------------------------------------------------
    # 3.2 Leer el archivo CSV
    # --------------------------------------------------------

    datos = pd.read_csv(ARCHIVO_DATOS)

    # --------------------------------------------------------
    # 3.3 Definir columnas necesarias
    # --------------------------------------------------------

    columnas_necesarias = (
        CARACTERISTICAS + [VARIABLE_OBJETIVO]
    )

    # --------------------------------------------------------
    # 3.4 Identificar columnas faltantes
    # --------------------------------------------------------

    columnas_faltantes = [

        columna

        for columna in columnas_necesarias

        if columna not in datos.columns

    ]

    if columnas_faltantes:

        raise ValueError(
            f"Faltan columnas: {columnas_faltantes}"
        )

    # --------------------------------------------------------
    # 3.5 Comprobar valores nulos
    # --------------------------------------------------------

    if datos[columnas_necesarias].isnull().any().any():

        raise ValueError(
            "Existen valores nulos en las variables utilizadas."
        )

    # --------------------------------------------------------
    # 3.6 Comprobar datos numéricos y finitos
    # --------------------------------------------------------

    if not np.isfinite(
        datos[columnas_necesarias].to_numpy(dtype=float)
    ).all():

        raise ValueError(
            "Se encontraron valores no numéricos o infinitos."
        )

    # --------------------------------------------------------
    # 3.7 Comprobar variable objetivo
    # --------------------------------------------------------

    if not set(
        datos[VARIABLE_OBJETIVO].unique()
    ).issubset({0, 1}):

        raise ValueError(
            "La variable aprueba debe contener únicamente 0 y 1."
        )

    # Comprobar que existan ambas clases.

    if datos[VARIABLE_OBJETIVO].nunique() != 2:

        raise ValueError(
            "El dataset debe contener las clases 0 y 1."
        )

    print("Dataset cargado correctamente.")

    print("Total de estudiantes:", len(datos))

    return datos


# ------------------------------------------------------------
# 4. SEPARACIÓN DE VARIABLES X E y
# ------------------------------------------------------------

def preparar_variables(datos):

    """
    Separamos las características predictoras
    de la variable objetivo.

    X = información académica.

    y = resultado de aprobación.
    """

    # Seleccionamos las características.

    X = datos[CARACTERISTICAS].copy()

    # Seleccionamos la variable objetivo.

    y = datos[VARIABLE_OBJETIVO].copy()

    print("\nCARACTERÍSTICAS DEL MODELO:")

    print(X.head())

    print("\nDISTRIBUCIÓN DE LA VARIABLE OBJETIVO:")

    print(y.value_counts())

    return X, y


# ------------------------------------------------------------
# 5. DIVISIÓN DE ENTRENAMIENTO Y PRUEBA
# ------------------------------------------------------------

def dividir_datos(X, y):

    """
    Dividimos los datos:

    80% entrenamiento.
    20% prueba.

    stratify=y conserva aproximadamente
    la proporción de las clases.
    """

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=SEMILLA,

        stratify=y

    )

    print("\nDIVISIÓN DE LOS DATOS:")

    print("Entrenamiento:", len(X_train))

    print("Prueba:", len(X_test))

    return X_train, X_test, y_train, y_test


# ------------------------------------------------------------
# 6. DEFINICIÓN DE LOS ALGORITMOS
# ------------------------------------------------------------

def crear_modelos():

    """
    Definimos dos algoritmos de clasificación.

    Decision Tree:
    Construye un árbol de decisiones.

    Random Forest:
    Construye múltiples árboles y combina sus resultados.
    """

    modelos = {

        "Decision Tree": DecisionTreeClassifier(

            max_depth=5,

            min_samples_split=10,

            min_samples_leaf=5,

            random_state=SEMILLA

        ),

        "Random Forest": RandomForestClassifier(

            n_estimators=200,

            max_depth=8,

            min_samples_split=5,

            min_samples_leaf=2,

            random_state=SEMILLA,

            n_jobs=-1

        )

    }

    return modelos


# ------------------------------------------------------------
# 7. ENTRENAMIENTO Y EVALUACIÓN
# ------------------------------------------------------------

def evaluar_modelos(

    modelos,

    X_train,

    X_test,

    y_train,

    y_test

):

    """
    Entrenamos cada algoritmo y calculamos
    sus métricas de evaluación.

    El conjunto de prueba se mantiene separado
    durante el entrenamiento y se utiliza aquí
    para la evaluación final de esta etapa.
    """

    resultados = {}

    modelos_entrenados = {}

    for nombre, modelo in modelos.items():

        print("\n" + "=" * 60)

        print("ENTRENANDO:", nombre)

        print("=" * 60)

        # ----------------------------------------------------
        # 7.1 Entrenamiento
        # ----------------------------------------------------

        modelo.fit(

            X_train,

            y_train

        )

        # ----------------------------------------------------
        # 7.2 Predicciones
        # ----------------------------------------------------

        predicciones = modelo.predict(X_test)

        # ----------------------------------------------------
        # 7.3 Cálculo de métricas
        # ----------------------------------------------------

        accuracy = accuracy_score(

            y_test,

            predicciones

        )

        precision = precision_score(

            y_test,

            predicciones,

            zero_division=0

        )

        recall = recall_score(

            y_test,

            predicciones,

            zero_division=0

        )

        f1 = f1_score(

            y_test,

            predicciones,

            zero_division=0

        )

        # ----------------------------------------------------
        # 7.4 Almacenamiento de resultados
        # ----------------------------------------------------

        resultados[nombre] = {

            "Accuracy": accuracy,

            "Precision": precision,

            "Recall": recall,

            "F1-score": f1

        }

        # Guardamos el modelo entrenado.

        modelos_entrenados[nombre] = modelo

        # ----------------------------------------------------
        # 7.5 Presentación de resultados
        # ----------------------------------------------------

        print("\nMÉTRICAS DE EVALUACIÓN:")

        print(f"Accuracy:  {accuracy:.4f}")

        print(f"Precision: {precision:.4f}")

        print(f"Recall:    {recall:.4f}")

        print(f"F1-score:  {f1:.4f}")

        print("\nREPORTE DE CLASIFICACIÓN:")

        print(

            classification_report(

                y_test,

                predicciones,

                #indicamos explicitamente las dos clases
                labels=[0, 1],

                #nombre correspondiente a cada clase

                target_names=[

                    "No aprueba",

                    "Aprueba"

                ],
                #evitamos divisiones entre cero
                zero_division=0

            )

        )

        # ----------------------------------------------------
        # 7.6 Matriz de confusión
        # ----------------------------------------------------

        matriz = confusion_matrix(

            y_test,

            predicciones,

            labels=[0, 1]

        )

        print("\nMATRIZ DE CONFUSIÓN:")

        print(matriz)

    return resultados, modelos_entrenados


# ------------------------------------------------------------
# 8. COMPARACIÓN DE RESULTADOS
# ------------------------------------------------------------

def comparar_modelos(resultados):

    """
    Creamos una tabla para comparar el rendimiento
    de los algoritmos.
    """

    tabla = pd.DataFrame(resultados).T

    tabla = tabla.sort_values(

        by="F1-score",

        ascending=False

    )

    print("\n" + "=" * 60)

    print("COMPARACIÓN DE ALGORITMOS")

    print("=" * 60)

    print(tabla.round(4))

    # Exportamos los resultados.

    archivo = (

        RUTA_BASE /

        "resultados_p10" /

        "comparacion_modelos.csv"

    )

    tabla.to_csv(

        archivo,

        encoding="utf-8"

    )

    return tabla


# ------------------------------------------------------------
# 9. SELECCIÓN Y PERSISTENCIA DEL MEJOR MODELO
# ------------------------------------------------------------

def guardar_mejor_modelo(

    tabla,

    modelos_entrenados

):

    """
    Seleccionamos el algoritmo con mayor F1-score
    y guardamos el modelo entrenado.

    También guardamos los nombres y el orden
    de las características necesarias para predecir.
    """

    # Obtenemos el nombre del primer modelo de la tabla.

    mejor_nombre = tabla.index[0]

    # Recuperamos el objeto del modelo.

    mejor_modelo = modelos_entrenados[mejor_nombre]

    # Creamos un paquete con el modelo y sus metadatos.

    paquete = {

        "modelo": mejor_modelo,

        "nombre_modelo": mejor_nombre,

        "caracteristicas": CARACTERISTICAS,

        "variable_objetivo": VARIABLE_OBJETIVO,

        "semilla": SEMILLA

    }

    # Definimos el archivo de persistencia.

    archivo_modelo = (

        RUTA_MODELOS /

        "modelo_smartacademy.joblib"

    )

    # Guardamos el paquete completo.

    joblib.dump(

        paquete,

        archivo_modelo

    )

    print("\nMEJOR MODELO SELECCIONADO:")

    print(mejor_nombre)

    print("\nMODELO GUARDADO EN:")

    print(archivo_modelo)

    return archivo_modelo


# ------------------------------------------------------------
# 10. VISUALIZACIÓN DE LA COMPARACIÓN
# ------------------------------------------------------------

def generar_grafica(tabla):

    """
    Construimos una gráfica de barras para comparar
    las métricas de ambos algoritmos.
    """

    # Creamos una figura.

    ax = tabla.plot(

        kind="bar",

        figsize=(11, 6)

    )

    # Título de la gráfica.

    plt.title(

        "Comparación de modelos SmartAcademy"

    )

    # Etiqueta del eje horizontal.

    plt.xlabel("Algoritmo")

    # Etiqueta del eje vertical.

    plt.ylabel("Valor de la métrica")

    # Establecemos los límites del eje vertical.

    plt.ylim(0, 1.05)

    # Rotamos los nombres de los algoritmos.

    plt.xticks(rotation=0)

    # Agregamos leyenda.

    plt.legend(

        title="Métricas",

        loc="lower right"

    )

    # Ajustamos espacios.

    plt.tight_layout()

    # Guardamos la imagen.

    archivo = (

        RUTA_BASE /

        "resultados_p10" /

        "comparacion_modelos.png"

    )

    plt.savefig(

        archivo,

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()

    print("\nGRÁFICA GENERADA:")

    print(archivo)


# ------------------------------------------------------------
# 11. PRUEBA DE RECUPERACIÓN DEL MODELO
# ------------------------------------------------------------

def probar_modelo_guardado(archivo_modelo):

    """
    Recuperamos el modelo desde el archivo Joblib.

    Posteriormente realizamos una predicción
    con un estudiante de ejemplo.
    """

    print("\n" + "=" * 60)

    print("PRUEBA DE PERSISTENCIA DEL MODELO")

    print("=" * 60)

    # Recuperamos el paquete.

    paquete = joblib.load(archivo_modelo)

    # Extraemos el modelo.

    modelo = paquete["modelo"]

    # Creamos un estudiante de ejemplo.

    estudiante = pd.DataFrame(

        [

            {

                "horas_estudio": 5,

                "asistencia": 15,

                "actividades": 9,

                "participacion": 20

            }

        ],

        columns=paquete["caracteristicas"]

    )

    # Realizamos la predicción.

    prediccion = modelo.predict(estudiante)[0]

    # Obtenemos la probabilidad estimada de cada clase.

    probabilidades = modelo.predict_proba(estudiante)[0]

    # Mostramos los resultados.

    print("\nDATOS DEL ESTUDIANTE:")

    print(estudiante)

    print("\nRESULTADO DEL MODELO:")

    if prediccion == 1:

        print("Clasificación: APROBACIÓN ESTIMADA")

    else:

        print("Clasificación: NO APROBACIÓN ESTIMADA")

    print("\nPROBABILIDADES ESTIMADAS:")

    print(

        f"No aprueba: {probabilidades[0] * 100:.2f}%"

    )

    print(

        f"Aprueba: {probabilidades[1] * 100:.2f}%"

    )

    print("\nMODELO UTILIZADO:")

    print(paquete["nombre_modelo"])


# ------------------------------------------------------------
# 12. FUNCIÓN PRINCIPAL
# ------------------------------------------------------------

def main():

    print("\n")

    print("=" * 60)

    print("SMARTACADEMY")

    print("ETAPA 2 - APRENDIZAJE SUPERVISADO")

    print("=" * 60)

    # Paso 1: cargar datos.

    datos = cargar_datos()

    # Paso 2: separar X e y.

    X, y = preparar_variables(datos)

    # Paso 3: dividir datos.

    X_train, X_test, y_train, y_test = dividir_datos(

        X,

        y

    )

    # Paso 4: crear algoritmos.

    modelos = crear_modelos()

    # Paso 5: entrenar y evaluar.

    resultados, modelos_entrenados = evaluar_modelos(

        modelos,

        X_train,

        X_test,

        y_train,

        y_test

    )

    # Paso 6: comparar resultados.

    tabla = comparar_modelos(resultados)

    # Paso 7: guardar el mejor modelo.

    archivo_modelo = guardar_mejor_modelo(

        tabla,

        modelos_entrenados

    )

    # Paso 8: generar gráfica.

    generar_grafica(tabla)

    # Paso 9: recuperar y probar el modelo.

    probar_modelo_guardado(archivo_modelo)

    print("\n" + "=" * 60)

    print("ETAPA 2 FINALIZADA CORRECTAMENTE")

    print("=" * 60)


# ------------------------------------------------------------
# 13. PUNTO DE ENTRADA
# ------------------------------------------------------------

if __name__ == "__main__":

    main()