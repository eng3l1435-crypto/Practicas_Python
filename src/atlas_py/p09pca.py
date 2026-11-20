
# ============================================================
# UNIDAD III - FUNDAMENTOS DE INTELIGENCIA ARTIFICIAL
# PRACTICA 9:
# REDUCCION DE DIMENSIONALIDAD MEDIANTE PCA 
#(Principal Component Analysis o Análisis de Componentes Principales).
#
# CASO: SMARTACADEMY

#El propósito será resolver una situación habitual: tenemos varias características de los estudiantes, 
#pero queremos representarlas gráficamente en dos dimensiones para comprender mejor su distribución, 
#identificar patrones y observar cómo se relacionan los segmentos encontrados con K-Means.



# OBJETIVOS:
#Aplicar PCA para reducir la dimensionalidad de un conjunto de datos académicos, 
#conservando la mayor cantidad posible de variabilidad y visualizando los segmentos identificados mediante K-Means.

#Al finalizar, los estudiantes podrán:

#1. Comprender qué es la dimensionalidad de un conjunto de datos.
#2. Identificar la importancia de estandarizar las variables antes de aplicar PCA.
#3. Calcular componentes principales.
#4. Interpretar la varianza explicada individual y acumulada.
#5. Representar estudiantes en dos y tres dimensiones.
#6. Comparar la distribución original de variables con su proyección PCA.
#7. Exportar las coordenadas y resultados para análisis posteriores.

#El sistema cuenta con cuatro variables académicas:
#   Variable	    Descripción

#horas_estudio	    Horas dedicadas al estudio.
#asistencia	        Porcentaje de asistencia.
#actividades	    Actividades académicas realizadas.
#participacion	    Porcentaje de participación.

#En la práctica 7 se identificaron tres segmentos:
#- Alto compromiso.
#- Compromiso intermedio.
#- Requiere acompañamiento.
#El problema es que visualizar cuatro variables simultáneamente no resulta sencillo en una gráfica convencional.

#PCA nos permitirá transformar las cuatro variables en dos componentes principales:
#- PC1: componente que concentra la mayor variabilidad posible.
#- PC2: componente que concentra la mayor variabilidad restante, sin correlacionarse con PC1.

#La intención es representar a cada estudiante mediante las coordenadas PC1 y PC2, 
#procurando conservar tanta información como sea posible.

#¿Qué es PCA?
#PCA es una técnica de reducción de dimensionalidad que transforma un conjunto de variables 
#originales en un nuevo conjunto de variables llamadas componentes principales.

#imaginemos que tenemos estos cuatro datos de un estudiante:

#Horas de estudio: 8
#Asistencia:       95
#Actividades:      28
#Participación:    92

#PCA analiza las relaciones y variaciones entre las cuatro características para construir nuevas 
#dimensiones que concentren la mayor variabilidad posible.

#Conceptos fundamentales

#- Varianza explicada: porcentaje de la variabilidad original que representa cada componente.
#- Varianza acumulada: suma de la varianza explicada por las primeras componentes.
#- Cargas o loadings: coeficientes que indican cómo contribuye cada variable original a las componentes.
#- Reconstrucción: aproximación de los datos originales a partir de las componentes conservadas.
#Por ejemplo, si PC1 explica el 55% y PC2 el 30%, ambas juntas conservan el 85% de la variabilidad original.

#No significa que se conserve el 85% de cada dato individual, sino el 85% de la varianza total 
#según el criterio de PCA.

#¿Cómo interpretar los resultados?
#Supongamos que el programa obtiene esta distribución hipotética:
#   Componente	    Varianza explicada	    Acumulada
#       PC1	                55%	                55%
#       PC2	                30%	                85%
#       PC3	                10%	                95%
#       PC4	                5%	                100%

#En este ejemplo:
#- PC1 concentra el 55% de la variabilidad.
#- PC2 agrega otro 30%.
#- PC1 y PC2 juntas permiten visualizar el 85% de la variabilidad.
#- PC3 y PC4 contienen el 15% restante.
#La representación bidimensional sería útil para explorar los datos, pero no conservaría toda la información.

#PCA busca conservar la varianza, no necesariamente separar los clusters. Por eso, aunque la gráfica 2D 
#muestre grupos parcialmente mezclados, no significa automáticamente que K-Means haya fallado. Parte de la 
#información que permite distinguir grupos podría estar en PC3 o PC4.

# ============================================================


# ------------------------------------------------------------
# 1. IMPORTACION DE LIBRERIAS
# ------------------------------------------------------------

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

from sklearn.metrics import mean_squared_error


# ------------------------------------------------------------
# 2. CONFIGURACION GENERAL
# ------------------------------------------------------------

SEMILLA = 42

NUM_CLUSTERS = 3

# Porcentaje de varianza acumulada que nos gustaria conservar.
# Es una referencia de trabajo, no una regla obligatoria.

VARIANZA_OBJETIVO = 0.85

# Carpeta para guardar los resultados.

CARPETA_RESULTADOS = "resultados_p09"

os.makedirs(CARPETA_RESULTADOS, exist_ok=True)


# ------------------------------------------------------------
# 3. VARIABLES ORIGINALES
# ------------------------------------------------------------

VARIABLES = [
    "horas_estudio",
    "asistencia",
    "actividades",
    "participacion"
]


# ------------------------------------------------------------
# 4. CARGA DE DATOS
# ------------------------------------------------------------

def cargar_datos():

    """
    Recupera los datos de la practica 7.

    Si existen columnas cluster o perfil_academico,
    se conservan solamente para colorear las graficas.

    Estas etiquetas NO se utilizan como entradas de PCA.
    """

    ruta = "p07_estudiantes_segmentados.csv"

    if not os.path.exists(ruta):

        raise FileNotFoundError(
            "No se encontro p07_estudiantes_segmentados.csv. "
            "Ejecuta primero la practica 7."
        )

    df = pd.read_csv(ruta)

    # Verificamos que existan las variables originales.
    faltantes = [
        columna
        for columna in VARIABLES
        if columna not in df.columns
    ]

    if faltantes:

        raise ValueError(
            "Faltan las siguientes variables: "
            + str(faltantes)
        )

    # Conservamos el identificador si existe.
    if "id_estudiante" not in df.columns:

        df.insert(
            0,
            "id_estudiante",
            [
                f"EST-{i:04d}"
                for i in range(1, len(df) + 1)
            ]
        )

    print("\nDatos cargados correctamente.")

    print("Total de estudiantes:", len(df))

    return df


# ------------------------------------------------------------
# 5. PREPARACION DE VARIABLES
# ------------------------------------------------------------

def preparar_datos(df):

    """
    Selecciona las cuatro variables originales.

    Los valores faltantes se sustituyen por la mediana.

    No se incluyen las columnas cluster, perfil_academico
    ni identificadores en el entrenamiento de PCA.
    """

    X = df[VARIABLES].copy()

    # Convertimos todas las columnas a formato numerico.
    for columna in VARIABLES:

        X[columna] = pd.to_numeric(
            X[columna],
            errors="coerce"
        )

    # Sustituimos valores faltantes con la mediana.
    X = X.fillna(X.median())

    # Convertimos a matriz NumPy.
    X = X.to_numpy(dtype=float)

    return X


# ------------------------------------------------------------
# 6. ESTANDARIZACION DE VARIABLES
# ------------------------------------------------------------

def estandarizar_datos(X):

    """
    PCA es sensible a la escala de las variables.

    Por ejemplo:
    - Horas de estudio: aproximadamente de 0 a 12.
    - Asistencia: de 0 a 100.
    - Participacion: de 0 a 100.

    Sin estandarizacion, las variables con mayor escala
    podrian dominar el analisis.

    StandardScaler transforma cada variable para que
    tenga media cercana a 0 y desviacion estandar cercana a 1.
    """

    escalador = StandardScaler()

    X_escalado = escalador.fit_transform(X)

    print("\n" + "=" * 60)
    print("ESTANDARIZACION")
    print("=" * 60)

    print("\nMedia de las variables escaladas:")

    print(np.round(X_escalado.mean(axis=0), 4))

    print("\nDesviacion estandar:")

    print(np.round(X_escalado.std(axis=0), 4))

    return X_escalado, escalador


# ------------------------------------------------------------
# 7. OBTENER ETIQUETAS DE CLUSTER
# ------------------------------------------------------------

def obtener_clusters(df, X_escalado):

    """
    Utiliza las etiquetas de P7 si existen.

    Si no existen, entrena K-Means con las variables
    originales estandarizadas.

    Estas etiquetas solo se utilizaran para colorear
    las visualizaciones de PCA.
    """

    if "cluster" in df.columns:

        print("\nSe utilizaran los clusters de la practica 7.")

        etiquetas = df["cluster"].to_numpy()

    else:

        print("\nNo existen etiquetas de P7.")
        print("Se generaran segmentos con K-Means.")

        modelo = KMeans(
            n_clusters=NUM_CLUSTERS,
            init="k-means++",
            n_init=20,
            random_state=SEMILLA
        )

        etiquetas = modelo.fit_predict(X_escalado)

    # Convertimos las etiquetas a numeros enteros.
    etiquetas = np.asarray(etiquetas).astype(int)

    return etiquetas


# ------------------------------------------------------------
# 8. ENTRENAMIENTO DE PCA
# ------------------------------------------------------------

def entrenar_pca(X_escalado):

    """
    Entrena PCA conservando inicialmente las cuatro
    componentes posibles.

    Como tenemos cuatro variables originales, podemos
    obtener hasta cuatro componentes principales.
    """

    modelo_pca = PCA(n_components=4)

    datos_pca = modelo_pca.fit_transform(X_escalado)

    print("\n" + "=" * 60)
    print("RESULTADOS GENERALES DE PCA")
    print("=" * 60)

    print("\nVarianza explicada por componente:")

    print(
        np.round(
            modelo_pca.explained_variance_ratio_,
            4
        )
    )

    print("\nVarianza explicada en porcentaje:")

    print(
        np.round(
            modelo_pca.explained_variance_ratio_ * 100,
            2
        )
    )

    # Calculamos la varianza acumulada.
    varianza_acumulada = np.cumsum(
        modelo_pca.explained_variance_ratio_
    )

    print("\nVarianza acumulada:")

    print(
        np.round(
            varianza_acumulada * 100,
            2
        )
    )

    return modelo_pca, datos_pca, varianza_acumulada


# ------------------------------------------------------------
# 9. TABLA DE VARIANZA EXPLICADA
# ------------------------------------------------------------

def generar_tabla_varianza(modelo_pca):

    """
    Construye una tabla para conocer cuanta informacion
    representa cada componente principal.
    """

    varianza = modelo_pca.explained_variance_ratio_

    acumulada = np.cumsum(varianza)

    df_varianza = pd.DataFrame({

        "componente": [
            "PC1",
            "PC2",
            "PC3",
            "PC4"
        ],

        "varianza_explicada": varianza,

        "porcentaje_explicado": varianza * 100,

        "varianza_acumulada": acumulada,

        "porcentaje_acumulado": acumulada * 100

    })

    print("\nTabla de varianza explicada:")

    print(
        df_varianza.round(4).to_string(index=False)
    )

    df_varianza.to_csv(
        f"{CARPETA_RESULTADOS}/p09_varianza_explicada.csv",
        index=False
    )

    return df_varianza


# ------------------------------------------------------------
# 10. ANALISIS DE CARGAS DE LAS COMPONENTES
# ------------------------------------------------------------

def analizar_cargas(modelo_pca):

    """
    Las cargas permiten analizar la contribucion de las
    variables originales a cada componente principal.

    Cada fila representa una componente.
    Cada columna representa una variable original.

    Los signos y magnitudes ayudan a interpretar la
    direccion y contribucion de cada variable.
    """

    cargas = pd.DataFrame(
        modelo_pca.components_,
        columns=VARIABLES,
        index=[
            "PC1",
            "PC2",
            "PC3",
            "PC4"
        ]
    )

    print("\n" + "=" * 60)
    print("CARGAS DE LAS COMPONENTES")
    print("=" * 60)

    print(cargas.round(4))

    cargas.to_csv(
        f"{CARPETA_RESULTADOS}/p09_cargas_componentes.csv"
    )

    # Graficamos las cargas.
    plt.figure(figsize=(10, 5))

    plt.imshow(
        cargas.values,
        aspect="auto",
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    plt.colorbar(label="Carga de componente")

    plt.xticks(
        range(len(VARIABLES)),
        VARIABLES,
        rotation=25,
        ha="right"
    )

    plt.yticks(
        range(4),
        cargas.index
    )

    plt.title("Cargas de las componentes principales")

    plt.tight_layout()

    plt.savefig(
        f"{CARPETA_RESULTADOS}/p09_cargas_componentes.png",
        dpi=300
    )

    plt.close()

    return cargas


# ------------------------------------------------------------
# 11. DETERMINAR NUMERO DE COMPONENTES NECESARIAS
# ------------------------------------------------------------

def determinar_componentes(modelo_pca):

    """
    Identifica el numero minimo de componentes que permite
    alcanzar el porcentaje de varianza objetivo.

    En esta practica utilizaremos 85% como referencia.

    Si ninguna reduccion permite alcanzar el objetivo,
    se conservaran las cuatro componentes.
    """

    acumulada = np.cumsum(
        modelo_pca.explained_variance_ratio_
    )

    cantidad = np.argmax(
        acumulada >= VARIANZA_OBJETIVO
    ) + 1

    if acumulada[-1] < VARIANZA_OBJETIVO:

        cantidad = len(acumulada)

    print("\n" + "=" * 60)
    print("SELECCION DE COMPONENTES")
    print("=" * 60)

    print("Varianza objetivo:")

    print(f"{VARIANZA_OBJETIVO * 100:.2f}%")

    print("Componentes necesarias:", cantidad)

    print(
        "Varianza acumulada conservada:",
        round(acumulada[cantidad - 1] * 100, 2),
        "%"
    )

    return cantidad


# ------------------------------------------------------------
# 12. GRAFICA DE VARIANZA EXPLICADA
# ------------------------------------------------------------

def graficar_varianza(df_varianza):

    """
    Genera dos visualizaciones:

    1. Grafica de sedimentacion:
       Permite observar la varianza de cada componente.

    2. Varianza acumulada:
       Permite observar cuanta variabilidad se conserva
       al agregar componentes.
    """

    # Grafica 1: sedimentacion.
    plt.figure(figsize=(8, 5))

    plt.plot(
        df_varianza["componente"],
        df_varianza["porcentaje_explicado"],
        marker="o"
    )

    plt.title("Grafica de sedimentacion PCA")
    plt.xlabel("Componentes principales")
    plt.ylabel("Varianza explicada (%)")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        f"{CARPETA_RESULTADOS}/p09_sedimentacion.png",
        dpi=300
    )

    plt.close()

    # Grafica 2: varianza acumulada.
    plt.figure(figsize=(8, 5))

    plt.plot(
        df_varianza["componente"],
        df_varianza["porcentaje_acumulado"],
        marker="o"
    )

    plt.axhline(
        y=VARIANZA_OBJETIVO * 100,
        linestyle="--",
        label="Objetivo de varianza"
    )

    plt.title("Varianza acumulada PCA")
    plt.xlabel("Componentes principales")
    plt.ylabel("Varianza acumulada (%)")

    plt.ylim(0, 105)

    plt.legend()

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        f"{CARPETA_RESULTADOS}/p09_varianza_acumulada.png",
        dpi=300
    )

    plt.close()


# ------------------------------------------------------------
# 13. VISUALIZACION PCA EN DOS DIMENSIONES
# ------------------------------------------------------------

def visualizar_pca_2d(datos_pca, etiquetas):

    """
    Representa a los estudiantes utilizando PC1 y PC2.

    Los colores representan los clusters de K-Means.

    Importante:
    PCA no utiliza las etiquetas de cluster para calcular
    las componentes.
    """

    plt.figure(figsize=(10, 7))

    dispersion = plt.scatter(
        datos_pca[:, 0],
        datos_pca[:, 1],
        c=etiquetas,
        cmap="viridis",
        alpha=0.75,
        s=45
    )

    plt.title("SmartAcademy - Proyeccion PCA en 2D")

    plt.xlabel("Componente principal 1 (PC1)")

    plt.ylabel("Componente principal 2 (PC2)")

    plt.colorbar(
        dispersion,
        label="Cluster K-Means"
    )

    plt.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        f"{CARPETA_RESULTADOS}/p09_pca_2d.png",
        dpi=300
    )

    plt.close()


# ------------------------------------------------------------
# 14. VISUALIZACION PCA EN TRES DIMENSIONES
# ------------------------------------------------------------

def visualizar_pca_3d(datos_pca, etiquetas):

    """
    Representa a los estudiantes mediante PC1, PC2 y PC3.

    Esto permite observar una dimension adicional,
    siempre que exista variabilidad suficiente en PC3.
    """

    figura = plt.figure(figsize=(10, 7))

    eje = figura.add_subplot(
        111,
        projection="3d"
    )

    dispersion = eje.scatter(
        datos_pca[:, 0],
        datos_pca[:, 1],
        datos_pca[:, 2],
        c=etiquetas,
        cmap="viridis",
        alpha=0.75,
        s=35
    )

    eje.set_title("SmartAcademy - Proyeccion PCA en 3D")

    eje.set_xlabel("PC1")

    eje.set_ylabel("PC2")

    eje.set_zlabel("PC3")

    figura.colorbar(
        dispersion,
        ax=eje,
        label="Cluster K-Means",
        shrink=0.7
    )

    plt.tight_layout()

    plt.savefig(
        f"{CARPETA_RESULTADOS}/p09_pca_3d.png",
        dpi=300
    )

    plt.close()


# ------------------------------------------------------------
# 15. EVALUACION DEL ERROR DE RECONSTRUCCION
# ------------------------------------------------------------

def evaluar_reconstruccion(
    X_escalado,
    datos_pca,
    modelo_pca,
    cantidad_componentes
):

    """
    Reconstruye los datos utilizando solamente las primeras
    componentes seleccionadas.

    Compara los datos estandarizados originales con los
    reconstruidos mediante error cuadratico medio (MSE).

    Un menor MSE significa menor error de reconstruccion.
    """

    # Conservamos las primeras componentes.
    datos_reducidos = datos_pca[:, :cantidad_componentes]

    # Reconstruimos los datos estandarizados.
    reconstruidos = (
        datos_reducidos
        @ modelo_pca.components_[:cantidad_componentes, :]
    ) + modelo_pca.mean_

    # Calculamos el error.
    error = mean_squared_error(
        X_escalado,
        reconstruidos
    )

    print("\n" + "=" * 60)
    print("EVALUACION DE RECONSTRUCCION")
    print("=" * 60)

    print("Componentes utilizadas:", cantidad_componentes)

    print("Error cuadratico medio:", round(error, 6))

    return reconstruidos, error


# ------------------------------------------------------------
# 16. EXPORTAR COORDENADAS PCA
# ------------------------------------------------------------

def exportar_coordenadas(
    df,
    datos_pca,
    etiquetas
):

    """
    Exporta las coordenadas de cada estudiante.

    Estas coordenadas pueden utilizarse posteriormente
    para visualizaciones, dashboards o almacenamiento
    en MongoDB Atlas.
    """

    resultado = pd.DataFrame({

        "id_estudiante": df["id_estudiante"],

        "cluster": etiquetas,

        "PC1": datos_pca[:, 0],

        "PC2": datos_pca[:, 1],

        "PC3": datos_pca[:, 2],

        "PC4": datos_pca[:, 3]

    })

    resultado.to_csv(
        f"{CARPETA_RESULTADOS}/p09_coordenadas_pca.csv",
        index=False
    )

    print("\nCoordenadas PCA exportadas correctamente.")

    return resultado


# ------------------------------------------------------------
# 17. EXPORTACION DE DATOS RECONSTRUIDOS
# ------------------------------------------------------------

def exportar_reconstruccion(
    df,
    reconstruidos,
    escalador
):

    """
    Convierte los datos reconstruidos a sus unidades
    originales y los exporta.

    Es importante recordar que al utilizar menos componentes
    obtenemos una aproximacion de los datos originales.
    """

    reconstruidos_originales = escalador.inverse_transform(
        reconstruidos
    )

    df_reconstruido = pd.DataFrame(
        reconstruidos_originales,
        columns=VARIABLES
    )

    df_reconstruido.insert(
        0,
        "id_estudiante",
        df["id_estudiante"].to_numpy()
    )

    df_reconstruido.to_csv(
        f"{CARPETA_RESULTADOS}/p09_datos_reconstruidos.csv",
        index=False
    )

    return df_reconstruido


# ------------------------------------------------------------
# 18. FUNCION PRINCIPAL
# ------------------------------------------------------------

def main():

    print("\n" + "=" * 65)

    print("SMARTACADEMY - PRACTICA 9")

    print("REDUCCION DE DIMENSIONALIDAD MEDIANTE PCA")

    print("=" * 65)

    # PASO 1: cargar datos.
    df = cargar_datos()

    # PASO 2: preparar variables.
    X = preparar_datos(df)

    # PASO 3: estandarizar variables.
    X_escalado, escalador = estandarizar_datos(X)

    # PASO 4: obtener etiquetas para colorear las graficas.
    etiquetas = obtener_clusters(
        df,
        X_escalado
    )

    # PASO 5: entrenar PCA.
    modelo_pca, datos_pca, acumulada = entrenar_pca(
        X_escalado
    )

    # PASO 6: generar tabla de varianza.
    df_varianza = generar_tabla_varianza(
        modelo_pca
    )

    # PASO 7: analizar cargas.
    cargas = analizar_cargas(
        modelo_pca
    )

    # PASO 8: determinar componentes necesarias.
    cantidad_componentes = determinar_componentes(
        modelo_pca
    )

    # PASO 9: generar graficas de varianza.
    graficar_varianza(
        df_varianza
    )

    # PASO 10: visualizar en dos dimensiones.
    visualizar_pca_2d(
        datos_pca,
        etiquetas
    )

    # PASO 11: visualizar en tres dimensiones.
    visualizar_pca_3d(
        datos_pca,
        etiquetas
    )

    # PASO 12: evaluar reconstruccion.
    reconstruidos, error = evaluar_reconstruccion(
        X_escalado,
        datos_pca,
        modelo_pca,
        cantidad_componentes
    )

    # PASO 13: exportar coordenadas.
    coordenadas = exportar_coordenadas(
        df,
        datos_pca,
        etiquetas
    )

    # PASO 14: exportar datos reconstruidos.
    df_reconstruido = exportar_reconstruccion(
        df,
        reconstruidos,
        escalador
    )

    # --------------------------------------------------------
    # RESUMEN FINAL
    # --------------------------------------------------------

    print("\n" + "=" * 65)

    print("RESUMEN FINAL DE LA PRACTICA 9")

    print("=" * 65)

    print("Estudiantes analizados:", len(df))

    print("Variables originales:", len(VARIABLES))

    print("Componentes principales:", len(modelo_pca.components_))

    print(
        "Varianza de PC1:",
        round(
            modelo_pca.explained_variance_ratio_[0] * 100,
            2
        ),
        "%"
    )

    print(
        "Varianza acumulada PC1 y PC2:",
        round(
            acumulada[1] * 100,
            2
        ),
        "%"
    )

    print(
        "Componentes seleccionadas:",
        cantidad_componentes
    )

    print("Error de reconstruccion:", round(error, 6))

    print("\nArchivos generados en:")

    print(CARPETA_RESULTADOS)

    print("\nPRACTICA 9 FINALIZADA CORRECTAMENTE.")


# ------------------------------------------------------------
# 19. PUNTO DE ENTRADA
# ------------------------------------------------------------

if __name__ == "__main__":

    main()
