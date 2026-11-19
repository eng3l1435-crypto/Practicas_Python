# ============================================================
# UNIDAD III - FUNDAMENTOS DE INTELIGENCIA ARTIFICIAL
# PRACTICA 8:
# ANALISIS AVANZADO Y OPTIMIZACION DE SEGMENTOS ACADEMICOS
#
# CASO: SMARTACADEMY
#
# OBJETIVOS:
# 1. Comparar inicializaciones de K-Means++ y random.
# 2. Comparar StandardScaler y MinMaxScaler.
# 3. Evaluar estabilidad mediante Adjusted Rand Index.
# 4. Detectar posibles estudiantes atipicos.
# 5. Analizar centroides y distancias.
# 6. Exportar resultados para analisis posterior.
#
# NOTA: este archivo incluye las ACTIVIDADES 1 a 7 para los alumnos.
# Estan marcadas con el texto ">>> ACTIVIDAD N <<<".
#
#En esta práctica se continuara directamente con el trabajo de la Práctica 7 (K-Means). 
#Ya conseguimos identificar tres perfiles académicos: alto compromiso, 
#compromiso intermedio y estudiantes que requieren acompañamiento.

#Ahora iremos un paso más allá: no solamente agruparemos estudiantes, sino que 
#investigaremos qué tan confiables son esos grupos, cómo cambian al modificar 
#el escalamiento y cuáles estudiantes presentan comportamientos atípicos respecto a su segmento.

#También prepararemos los resultados para que posteriormente podamos almacenarlos en MongoDB Atlas.

#SmartAcademy desea comprobar si los tres segmentos académicos identificados en la práctica anterior 
#son consistentes.

#Por ejemplo, si un estudiante pertenece al segmento de alto compromiso, queremos saber si 
#continúa perteneciendo a un grupo semejante cuando cambiamos la inicialización o ejecutamos 
#nuevamente el algoritmo con otra semilla aleatoria.

#También interesa detectar estudiantes cuyo comportamiento se encuentre muy alejado del centro de su segmento. 
#Estos casos podrían requerir una revisión más detallada, aunque una distancia elevada no demuestra por sí 
#sola que exista un problema académico.

# ============================================================


# ------------------------------------------------------------
# 1. IMPORTACION DE LIBRERIAS
# ------------------------------------------------------------

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler
)

from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    adjusted_rand_score
)

# >>> ACTIVIDAD 7 <<<
# Se necesita para encontrar al vecino mas cercano de cada estudiante.
from sklearn.neighbors import NearestNeighbors


# ------------------------------------------------------------
# 2. CONFIGURACION GENERAL
# ------------------------------------------------------------

# Numero de grupos que identificamos en la practica 7.
# En esta practica comprobaremos que tan consistentes son.
# (ACTIVIDAD 1: tambien puedes cambiar este valor a 2 y a 4 a mano.)

NUM_CLUSTERS = 3

# Semilla para reproducibilidad.
SEMILLA = 42

# Numero de inicializaciones internas de K-Means.
# (ACTIVIDAD 2: tambien puedes cambiar este valor a 5 y a 50 a mano.)

NUM_INICIALIZACIONES = 20

# Percentil utilizado para identificar posibles casos atipicos.
# Se consideran candidatos aquellos que superan el percentil 95
# de las distancias a sus respectivos centroides.
# (ACTIVIDAD 4: tambien puedes cambiar este valor a 90 y a 97.5 a mano.)

PERCENTIL_ATIPICOS = 95

# Nombre de la carpeta donde guardaremos los resultados.

CARPETA_RESULTADOS = "resultados_p08"

os.makedirs(CARPETA_RESULTADOS, exist_ok=True)


# ------------------------------------------------------------
# 3. DEFINICION DE VARIABLES
# ------------------------------------------------------------

# Estas son las cuatro variables que utilizamos en la practica 7.

VARIABLES = [
    "horas_estudio",
    "asistencia",
    "actividades",
    "participacion"
]


# ------------------------------------------------------------
# 4. GENERACION DE DATOS SI NO EXISTE EL CSV DE P7
# ------------------------------------------------------------

def generar_datos_smartacademy():

    """
    Genera 500 estudiantes con tres perfiles simulados.

    Los perfiles se utilizan exclusivamente para construir
    un conjunto de datos didactico.

    K-Means no recibe estas etiquetas.
    """

    # Generador aleatorio reproducible.
    rng = np.random.default_rng(42)

    # --------------------------------------------------------
    # Perfil 1: alto compromiso
    # --------------------------------------------------------

    alto = np.column_stack([
        rng.normal(8.0, 1.0, 180),
        rng.normal(92.0, 5.0, 180),
        rng.normal(27.0, 2.0, 180),
        rng.normal(90.0, 6.0, 180)
    ])

    # --------------------------------------------------------
    # Perfil 2: compromiso intermedio
    # --------------------------------------------------------

    medio = np.column_stack([
        rng.normal(5.0, 1.0, 170),
        rng.normal(75.0, 7.0, 170),
        rng.normal(20.0, 3.0, 170),
        rng.normal(65.0, 8.0, 170)
    ])

    # --------------------------------------------------------
    # Perfil 3: requiere acompañamiento
    # --------------------------------------------------------

    bajo = np.column_stack([
        rng.normal(2.5, 1.0, 150),
        rng.normal(55.0, 8.0, 150),
        rng.normal(12.0, 4.0, 150),
        rng.normal(40.0, 10.0, 150)
    ])

    # Unimos los tres perfiles en un solo conjunto.
    datos = np.vstack([alto, medio, bajo])

    # Mezclamos las filas para evitar que queden ordenadas
    # por perfil.
    rng.shuffle(datos)

    # Limitamos los valores a rangos plausibles.
    datos[:, 0] = np.clip(datos[:, 0], 0, 12)
    datos[:, 1] = np.clip(datos[:, 1], 0, 100)
    datos[:, 2] = np.clip(datos[:, 2], 0, 30)
    datos[:, 3] = np.clip(datos[:, 3], 0, 100)

    # Convertimos los datos a un DataFrame.
    df = pd.DataFrame(datos, columns=VARIABLES)

    # Creamos un identificador para cada estudiante.
    df.insert(
        0,
        "id_estudiante",
        [f"EST-{i:04d}" for i in range(1, len(df) + 1)]
    )

    return df


# ------------------------------------------------------------
# 5. CARGAR LOS DATOS DE LA PRACTICA 7
# ------------------------------------------------------------

def cargar_datos():

    """
    Intenta recuperar el archivo generado en P7.

    Si no existe, genera datos equivalentes.

    Las columnas de cluster y perfil se eliminan para evitar
    que se utilicen como variables de entrada.
    """

    ruta_csv = "p07_estudiantes_segmentados.csv"

    if os.path.exists(ruta_csv):

        print("\nArchivo de P7 encontrado.")
        print("Cargando estudiantes segmentados...")

        df = pd.read_csv(ruta_csv)

        # Verificamos que existan las cuatro variables.
        faltantes = [
            columna
            for columna in VARIABLES
            if columna not in df.columns
        ]

        if faltantes:
            raise ValueError(
                "El archivo de P7 no contiene estas variables: "
                + str(faltantes)
            )

        # Conservamos solo las variables originales.
        # No reutilizamos las etiquetas de los clusters.
        df = df.drop(
            columns=[
                "cluster",
                "perfil_academico"
            ],
            errors="ignore"
        )

        # Si no existe identificador, lo creamos.
        if "id_estudiante" not in df.columns:

            df.insert(
                0,
                "id_estudiante",
                [
                    f"EST-{i:04d}"
                    for i in range(1, len(df) + 1)
                ]
            )

        return df

    else:

        print("\nNo se encontro el CSV de P7.")
        print("Se generara un conjunto de datos equivalente.")

        return generar_datos_smartacademy()


# ------------------------------------------------------------
# 6. PREPARACION DE LOS DATOS
# ------------------------------------------------------------

def preparar_datos(df):

    """
    Verifica valores faltantes y prepara la matriz numerica.

    K-Means necesita datos numericos y no admite valores NaN.
    """

    print("\n" + "=" * 60)
    print("PREPARACION DE DATOS")
    print("=" * 60)

    print("\nNumero de estudiantes:", len(df))

    print("\nValores faltantes por variable:")

    print(df[VARIABLES].isnull().sum())

    # Convertimos las variables a formato numerico.
    for columna in VARIABLES:

        df[columna] = pd.to_numeric(
            df[columna],
            errors="coerce"
        )

    # Sustituimos valores faltantes por la mediana.
    # En un proyecto real se debe justificar esta decision.
    df[VARIABLES] = df[VARIABLES].fillna(
        df[VARIABLES].median()
    )

    # Matriz que recibiran los algoritmos.
    X = df[VARIABLES].to_numpy()

    return df, X


# ------------------------------------------------------------
# 7. COMPARACION DE INICIALIZACIONES DE K-MEANS
# ------------------------------------------------------------

def comparar_inicializaciones(X):

    """
    Compara dos estrategias de inicializacion:

    k-means++:
        Selecciona centroides iniciales procurando que
        se encuentren separados.

    random:
        Selecciona centroides iniciales aleatoriamente.

    Se mantienen constantes el escalamiento, K y n_init
    para que la comparacion sea mas controlada.
    """

    print("\n" + "=" * 60)
    print("COMPARACION DE INICIALIZACIONES")
    print("=" * 60)

    escalador = StandardScaler()

    X_escalado = escalador.fit_transform(X)

    resultados = []
    modelos = {}

    for inicializacion in ["k-means++", "random"]:

        modelo = KMeans(
            n_clusters=NUM_CLUSTERS,
            init=inicializacion,
            n_init=NUM_INICIALIZACIONES,
            random_state=SEMILLA
        )

        etiquetas = modelo.fit_predict(X_escalado)

        # Medimos la separacion y cohesion de los grupos.
        silhouette = silhouette_score(
            X_escalado,
            etiquetas
        )

        resultados.append({
            "inicializacion": inicializacion,
            "inercia": modelo.inertia_,
            "silhouette": silhouette
        })

        modelos[inicializacion] = modelo

        print(f"\nInicializacion: {inicializacion}")
        print("Inercia:", round(modelo.inertia_, 4))
        print("Silhouette:", round(silhouette, 4))

    df_resultados = pd.DataFrame(resultados)

    df_resultados.to_csv(
        f"{CARPETA_RESULTADOS}/p08_inicializaciones.csv",
        index=False
    )

    # Grafica comparativa de Silhouette.
    plt.figure(figsize=(8, 5))

    plt.bar(
        df_resultados["inicializacion"],
        df_resultados["silhouette"]
    )

    plt.title("Comparacion de inicializaciones K-Means")
    plt.xlabel("Metodo de inicializacion")
    plt.ylabel("Silhouette Score")

    plt.tight_layout()

    plt.savefig(
        f"{CARPETA_RESULTADOS}/p08_inicializaciones.png",
        dpi=300
    )

    plt.close()

    return df_resultados, modelos, X_escalado, escalador


# ------------------------------------------------------------
# 8. COMPARACION DE ESCALADORES
# ------------------------------------------------------------

def comparar_escaladores(X):

    """
    Compara StandardScaler y MinMaxScaler.

    StandardScaler:
        Centra los datos alrededor de cero y los expresa
        en unidades de desviacion estandar.

    MinMaxScaler:
        Transforma los datos al intervalo [0, 1].

    La distancia entre estudiantes puede cambiar cuando
    utilizamos diferentes escaladores.
    """

    print("\n" + "=" * 60)
    print("COMPARACION DE ESCALADORES")
    print("=" * 60)

    escaladores = {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler()
    }

    resultados = []

    modelos = {}

    for nombre, escalador in escaladores.items():

        X_transformado = escalador.fit_transform(X)

        modelo = KMeans(
            n_clusters=NUM_CLUSTERS,
            init="k-means++",
            n_init=NUM_INICIALIZACIONES,
            random_state=SEMILLA
        )

        etiquetas = modelo.fit_predict(X_transformado)

        silhouette = silhouette_score(
            X_transformado,
            etiquetas
        )

        resultados.append({
            "escalador": nombre,
            "inercia": modelo.inertia_,
            "silhouette": silhouette
        })

        modelos[nombre] = {
            "modelo": modelo,
            "escalador": escalador,
            "datos_escalados": X_transformado,
            "etiquetas": etiquetas
        }

        print(f"\nEscalador: {nombre}")
        print("Inercia:", round(modelo.inertia_, 4))
        print("Silhouette:", round(silhouette, 4))

    df_resultados = pd.DataFrame(resultados)

    df_resultados.to_csv(
        f"{CARPETA_RESULTADOS}/p08_comparacion_escaladores.csv",
        index=False
    )

    # Grafica de comparacion.
    plt.figure(figsize=(8, 5))

    plt.bar(
        df_resultados["escalador"],
        df_resultados["silhouette"]
    )

    plt.title("Comparacion de escaladores")
    plt.xlabel("Escalador")
    plt.ylabel("Silhouette Score")

    plt.tight_layout()

    plt.savefig(
        f"{CARPETA_RESULTADOS}/p08_escaladores.png",
        dpi=300
    )

    plt.close()

    return df_resultados, modelos


# ------------------------------------------------------------
# 9. ANALISIS DE ESTABILIDAD MEDIANTE ARI
# ------------------------------------------------------------

def analizar_estabilidad(X_escalado):

    """
    Ejecuta K-Means con distintas semillas.

    Posteriormente compara las asignaciones de grupos
    mediante Adjusted Rand Index (ARI).

    ARI:
        1.0 indica coincidencia perfecta entre particiones.
        Cerca de 0 indica coincidencia similar a la esperada
        por azar.
        Puede tomar valores negativos.

    ARI permite comparar particiones aunque los numeros
    asignados a los clusters sean diferentes.
    """

    print("\n" + "=" * 60)
    print("ANALISIS DE ESTABILIDAD")
    print("=" * 60)

    semillas = [10, 20, 30, 40, 50, 42]

    particiones = {}

    for semilla in semillas:

        modelo = KMeans(
            n_clusters=NUM_CLUSTERS,
            init="k-means++",
            n_init=NUM_INICIALIZACIONES,
            random_state=semilla
        )

        etiquetas = modelo.fit_predict(X_escalado)

        particiones[semilla] = etiquetas

    # Comparamos todas las combinaciones de semillas.
    resultados = []

    for i in range(len(semillas)):

        for j in range(i + 1, len(semillas)):

            semilla_a = semillas[i]
            semilla_b = semillas[j]

            ari = adjusted_rand_score(
                particiones[semilla_a],
                particiones[semilla_b]
            )

            resultados.append({
                "semilla_a": semilla_a,
                "semilla_b": semilla_b,
                "ARI": ari
            })

    df_ari = pd.DataFrame(resultados)

    print("\nComparaciones entre ejecuciones:")

    print(df_ari.round(4).to_string(index=False))

    print(
        "\nARI promedio:",
        round(df_ari["ARI"].mean(), 4)
    )

    df_ari.to_csv(
        f"{CARPETA_RESULTADOS}/p08_estabilidad_ari.csv",
        index=False
    )

    # Construimos una matriz de estabilidad.
    matriz = np.eye(len(semillas))

    for _, fila in df_ari.iterrows():

        i = semillas.index(int(fila["semilla_a"]))
        j = semillas.index(int(fila["semilla_b"]))

        matriz[i, j] = fila["ARI"]
        matriz[j, i] = fila["ARI"]

    plt.figure(figsize=(8, 6))

    imagen = plt.imshow(
        matriz,
        vmin=0,
        vmax=1,
        cmap="viridis"
    )

    plt.colorbar(
        imagen,
        label="Adjusted Rand Index"
    )

    plt.xticks(
        range(len(semillas)),
        semillas
    )

    plt.yticks(
        range(len(semillas)),
        semillas
    )

    plt.title("Estabilidad de K-Means mediante ARI")
    plt.xlabel("Semilla")
    plt.ylabel("Semilla")

    for i in range(len(semillas)):

        for j in range(len(semillas)):

            plt.text(
                j,
                i,
                f"{matriz[i, j]:.2f}",
                ha="center",
                va="center",
                color="white"
            )

    plt.tight_layout()

    plt.savefig(
        f"{CARPETA_RESULTADOS}/p08_matriz_estabilidad.png",
        dpi=300
    )

    plt.close()

    return df_ari, particiones


# ------------------------------------------------------------
# 10. ENTRENAMIENTO DEL MODELO FINAL
# ------------------------------------------------------------

def entrenar_modelo_final(X):

    """
    Entrena el modelo final con StandardScaler y K-Means.

    Para esta practica se conserva K=3, porque en P7
    fue la alternativa con mejor Silhouette entre K=2 y K=8.

    En un proyecto real se debe volver a justificar K
    con los datos disponibles.
    """

    escalador = StandardScaler()

    X_escalado = escalador.fit_transform(X)

    modelo = KMeans(
        n_clusters=NUM_CLUSTERS,
        init="k-means++",
        n_init=NUM_INICIALIZACIONES,
        random_state=SEMILLA
    )

    etiquetas = modelo.fit_predict(X_escalado)

    # Calculamos Silhouette individual.
    silhouette_individual = silhouette_samples(
        X_escalado,
        etiquetas
    )

    return (
        modelo,
        escalador,
        X_escalado,
        etiquetas,
        silhouette_individual
    )


# ------------------------------------------------------------
# 11. ANALISIS DE CENTROIDES Y PERFILES
# ------------------------------------------------------------

def analizar_centroides(
    df,
    modelo,
    escalador,
    etiquetas
):

    """
    Recupera los centroides a la escala original.

    Los centroides se encuentran inicialmente en el espacio
    escalado. inverse_transform permite interpretarlos en
    las unidades originales.
    """

    print("\n" + "=" * 60)
    print("ANALISIS DE CENTROIDES")
    print("=" * 60)

    centroides_originales = escalador.inverse_transform(
        modelo.cluster_centers_
    )

    df_centroides = pd.DataFrame(
        centroides_originales,
        columns=VARIABLES
    )

    df_centroides.insert(
        0,
        "cluster",
        range(NUM_CLUSTERS)
    )

    print("\nCentroides en unidades originales:")

    print(df_centroides.round(2).to_string(index=False))

    # Calculamos los promedios observados de cada grupo.
    df_perfiles = df[VARIABLES].copy()

    df_perfiles["cluster"] = etiquetas

    perfiles = df_perfiles.groupby(
        "cluster"
    )[VARIABLES].mean().reset_index()

    print("\nPerfiles promedio de los estudiantes:")

    print(perfiles.round(2).to_string(index=False))

    df_centroides.to_csv(
        f"{CARPETA_RESULTADOS}/p08_centroides.csv",
        index=False
    )

    perfiles.to_csv(
        f"{CARPETA_RESULTADOS}/p08_perfiles.csv",
        index=False
    )

    return df_centroides, perfiles


# ------------------------------------------------------------
# 12. DETECCION DE POSIBLES CASOS ATIPICOS
# ------------------------------------------------------------

def detectar_atipicos(
    df,
    X_escalado,
    modelo,
    etiquetas,
    silhouette_individual
):

    """
    Calcula la distancia euclidiana de cada estudiante
    respecto al centroide de su propio cluster.

    Se considera candidato a revision cuando la distancia
    supera el percentil 95 de todas las distancias.

    Este umbral es una regla exploratoria, no una prueba
    estadistica definitiva de anomalia.
    """

    print("\n" + "=" * 60)
    print("DETECCION DE POSIBLES CASOS ATIPICOS")
    print("=" * 60)

    # Recuperamos el centroide correspondiente a cada alumno.
    centroides_asignados = modelo.cluster_centers_[etiquetas]

    # Calculamos la distancia euclidiana.
    distancias = np.linalg.norm(
        X_escalado - centroides_asignados,
        axis=1
    )

    # Calculamos el umbral de distancia.
    umbral = np.percentile(
        distancias,
        PERCENTIL_ATIPICOS
    )

    # Creamos una copia para conservar la informacion.
    df_atipicos = df.copy()

    df_atipicos["cluster"] = etiquetas

    df_atipicos["distancia_centroide"] = distancias

    df_atipicos["silhouette_individual"] = silhouette_individual

    df_atipicos["posible_atipico"] = (
        df_atipicos["distancia_centroide"] > umbral
    )

    # Identificamos los registros candidatos.
    candidatos = df_atipicos[
        df_atipicos["posible_atipico"]
    ].copy()

    print(f"\nUmbral de distancia (percentil {PERCENTIL_ATIPICOS}):")
    print(round(umbral, 4))

    print("\nTotal de candidatos:")
    print(len(candidatos))

    print(
        "\nPorcentaje de candidatos:",
        round(
            len(candidatos) / len(df_atipicos) * 100,
            2
        ),
        "%"
    )

    print("\nPrimeros candidatos a revision:")

    print(
        candidatos[
            [
                "id_estudiante",
                "cluster",
                "distancia_centroide",
                "silhouette_individual"
            ]
        ].head(10).round(4).to_string(index=False)
    )

    # Exportamos todos los estudiantes con sus distancias.
    df_atipicos.to_csv(
        f"{CARPETA_RESULTADOS}/p08_analisis_distancias.csv",
        index=False
    )

    # Exportamos solo los candidatos.
    candidatos.to_csv(
        f"{CARPETA_RESULTADOS}/p08_posibles_atipicos.csv",
        index=False
    )

    # Grafica de distancias.
    plt.figure(figsize=(10, 5))

    plt.scatter(
        range(len(distancias)),
        distancias
    )

    plt.axhline(
        y=umbral,
        linestyle="--",
        label=f"Umbral P{PERCENTIL_ATIPICOS}"
    )

    plt.title("Distancia de estudiantes a sus centroides")
    plt.xlabel("Estudiante")
    plt.ylabel("Distancia euclidiana")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"{CARPETA_RESULTADOS}/p08_distancias_atipicos.png",
        dpi=300
    )

    plt.close()

    return df_atipicos, candidatos, umbral


# ============================================================
# ACTIVIDADES PARA LOS ALUMNOS (1 A 7)
# Cada funcion reutiliza lo que ya calcula la practica y
# guarda su resultado en la carpeta resultados_p08.
# Se llaman desde main(), despues del Paso 10.
# ============================================================


# ------------------------------------------------------------
# >>> ACTIVIDAD 1 <<<
# Cambiar NUM_CLUSTERS = 3 por 2 y posteriormente por 4.
# Se hace un ciclo con K = 2, 3 y 4 y se comparan inercia,
# silhouette, tamano de cada grupo y perfiles promedio.
# La inercia siempre baja al aumentar K, por eso no sirve sola
# para elegir K: hay que mirar tambien silhouette.
# ------------------------------------------------------------

def actividad_1_variar_clusters(df, X):

    print("\n" + "=" * 60)
    print("ACTIVIDAD 1: VARIAR NUM_CLUSTERS (2, 3, 4)")
    print("=" * 60)

    X_esc = StandardScaler().fit_transform(X)

    filas = []

    for k in [2, 3, 4]:

        m = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=NUM_INICIALIZACIONES,
            random_state=SEMILLA
        )

        et = m.fit_predict(X_esc)

        filas.append({
            "K": k,
            "inercia": m.inertia_,
            "silhouette": silhouette_score(X_esc, et),
            "tamanos": np.bincount(et).tolist()
        })

        print(f"\nK = {k} | perfiles promedio (unidades originales):")
        print(df[VARIABLES].groupby(et).mean().round(2))

    res = pd.DataFrame(filas)

    print("\nResumen:")
    print(res.round(4).to_string(index=False))

    res.to_csv(
        f"{CARPETA_RESULTADOS}/p08_act1_variar_k.csv",
        index=False
    )


# ------------------------------------------------------------
# >>> ACTIVIDAD 2 <<<
# Cambiar NUM_INICIALIZACIONES = 20 por 5 y 50.
# Se calcula tambien el ARI contra la version con 20 para ver
# si el resultado realmente cambia.
# ------------------------------------------------------------

def actividad_2_variar_n_init(X):

    print("\n" + "=" * 60)
    print("ACTIVIDAD 2: VARIAR n_init (5, 20, 50)")
    print("=" * 60)

    X_esc = StandardScaler().fit_transform(X)

    referencia = KMeans(
        n_clusters=NUM_CLUSTERS,
        init="k-means++",
        n_init=20,
        random_state=SEMILLA
    ).fit_predict(X_esc)

    filas = []

    for n in [5, 20, 50]:

        m = KMeans(
            n_clusters=NUM_CLUSTERS,
            init="k-means++",
            n_init=n,
            random_state=SEMILLA
        )

        et = m.fit_predict(X_esc)

        filas.append({
            "n_init": n,
            "inercia": m.inertia_,
            "silhouette": silhouette_score(X_esc, et),
            "ARI_vs_n_init_20": adjusted_rand_score(referencia, et)
        })

    res = pd.DataFrame(filas)

    print(res.round(4).to_string(index=False))

    res.to_csv(
        f"{CARPETA_RESULTADOS}/p08_act2_variar_n_init.csv",
        index=False
    )


# ------------------------------------------------------------
# >>> ACTIVIDAD 3 <<<
# Agregar una nueva variable llamada entregas_puntuales y
# evaluar su efecto.
# Si el CSV no la trae, se SIMULA (supuesto didactico: porcentaje
# de entregas a tiempo, relacionado con actividades y asistencia
# mas ruido). Se compara el modelo con 4 vs 5 variables.
# ------------------------------------------------------------

def actividad_3_nueva_variable(df):

    print("\n" + "=" * 60)
    print("ACTIVIDAD 3: NUEVA VARIABLE entregas_puntuales")
    print("=" * 60)

    df3 = df.copy()

    if "entregas_puntuales" not in df3.columns:

        rng = np.random.default_rng(SEMILLA)

        base = (
            0.6 * (df3["actividades"] / 30 * 100)
            + 0.4 * df3["asistencia"]
        )

        df3["entregas_puntuales"] = np.clip(
            rng.normal(base, 8),
            0,
            100
        )

    variables_5 = VARIABLES + ["entregas_puntuales"]

    resultados = {}

    for nombre, columnas in [
        ("4 variables", VARIABLES),
        ("5 variables", variables_5)
    ]:

        Xe = StandardScaler().fit_transform(
            df3[columnas].to_numpy()
        )

        et = KMeans(
            n_clusters=NUM_CLUSTERS,
            init="k-means++",
            n_init=NUM_INICIALIZACIONES,
            random_state=SEMILLA
        ).fit_predict(Xe)

        resultados[nombre] = (et, silhouette_score(Xe, et))

        print(f"Silhouette con {nombre}:",
              round(resultados[nombre][1], 4))

    ari = adjusted_rand_score(
        resultados["4 variables"][0],
        resultados["5 variables"][0]
    )

    print("ARI entre ambas segmentaciones:", round(ari, 4))

    print("\nCorrelacion de la nueva variable con las demas:")
    print(df3[variables_5].corr()["entregas_puntuales"].round(3))

    pd.DataFrame([{
        "silhouette_4_variables": resultados["4 variables"][1],
        "silhouette_5_variables": resultados["5 variables"][1],
        "ARI_4_vs_5": ari
    }]).to_csv(
        f"{CARPETA_RESULTADOS}/p08_act3_nueva_variable.csv",
        index=False
    )


# ------------------------------------------------------------
# >>> ACTIVIDAD 4 <<<
# Modificar PERCENTIL_ATIPICOS = 95 por 90 y 97.5.
# Comparar cuantos estudiantes se identifican.
# Se recalcula el umbral con cada percentil sobre las mismas
# distancias del modelo final.
# ------------------------------------------------------------

def actividad_4_variar_percentil(X_escalado, modelo, etiquetas):

    print("\n" + "=" * 60)
    print("ACTIVIDAD 4: VARIAR PERCENTIL DE ATIPICOS (90, 95, 97.5)")
    print("=" * 60)

    distancias = np.linalg.norm(
        X_escalado - modelo.cluster_centers_[etiquetas],
        axis=1
    )

    filas = []

    for p in [90, 95, 97.5]:

        umbral = np.percentile(distancias, p)

        n = int((distancias > umbral).sum())

        filas.append({
            "percentil": p,
            "umbral": umbral,
            "n_atipicos": n,
            "porcentaje": n / len(distancias) * 100
        })

    res = pd.DataFrame(filas)

    print(res.round(3).to_string(index=False))

    res.to_csv(
        f"{CARPETA_RESULTADOS}/p08_act4_percentiles.csv",
        index=False
    )


# ------------------------------------------------------------
# >>> ACTIVIDAD 5 <<<
# Comparar los perfiles academicos obtenidos con StandardScaler
# y MinMaxScaler.
# Reutiliza modelos_scaler, que ya devuelve comparar_escaladores().
# ------------------------------------------------------------

def actividad_5_perfiles_escaladores(df, modelos_scaler):

    print("\n" + "=" * 60)
    print("ACTIVIDAD 5: PERFILES CON StandardScaler vs MinMaxScaler")
    print("=" * 60)

    et_std = modelos_scaler["StandardScaler"]["etiquetas"]
    et_mm = modelos_scaler["MinMaxScaler"]["etiquetas"]

    print("ARI entre escaladores:",
          round(adjusted_rand_score(et_std, et_mm), 4))

    # Los numeros de cluster pueden venir permutados, por eso
    # se usa una tabla cruzada para ver la correspondencia.
    tabla = pd.crosstab(et_std, et_mm)

    print("\nTabla cruzada (filas=Standard, columnas=MinMax):")
    print(tabla)

    print("Estudiantes en un grupo distinto (aprox.):",
          int(len(df) - tabla.max(axis=1).sum()))

    for nombre, et in [
        ("StandardScaler", et_std),
        ("MinMaxScaler", et_mm)
    ]:

        perfil = df[VARIABLES].groupby(et).mean().round(2)

        print(f"\nPerfiles promedio con {nombre}:")
        print(perfil)

        perfil.to_csv(
            f"{CARPETA_RESULTADOS}/p08_act5_perfiles_{nombre}.csv"
        )


# ------------------------------------------------------------
# >>> ACTIVIDAD 6 <<<
# Explicar si los segmentos son estables entre semillas.
# Reutiliza df_ari de analizar_estabilidad() y agrega un
# contraste con n_init=1, donde la semilla si puede afectar.
# Interpretacion: ARI cercano a 1 = segmentos estables; cambiar
# la semilla solo cambia el NUMERO de la etiqueta, no el grupo.
# ------------------------------------------------------------

def actividad_6_estabilidad_semillas(X_escalado, df_ari):

    print("\n" + "=" * 60)
    print("ACTIVIDAD 6: ESTABILIDAD ENTRE SEMILLAS")
    print("=" * 60)

    print(
        "ARI con n_init=%d: minimo=%.4f  promedio=%.4f" % (
            NUM_INICIALIZACIONES,
            df_ari["ARI"].min(),
            df_ari["ARI"].mean()
        )
    )

    semillas = [10, 20, 30, 40, 50, 42]

    particiones = [
        KMeans(
            n_clusters=NUM_CLUSTERS,
            init="k-means++",
            n_init=1,
            random_state=s
        ).fit_predict(X_escalado)
        for s in semillas
    ]

    aris = [
        adjusted_rand_score(particiones[i], particiones[j])
        for i in range(len(particiones))
        for j in range(i + 1, len(particiones))
    ]

    print(
        "ARI con n_init=1:  minimo=%.4f  promedio=%.4f" % (
            min(aris),
            np.mean(aris)
        )
    )

    print("\nInterpretacion: si el ARI es cercano a 1, los segmentos "
          "son estables entre semillas.")


# ------------------------------------------------------------
# >>> ACTIVIDAD 7 <<<
# Investigar por que dos estudiantes con caracteristicas
# similares podrian terminar en grupos diferentes.
# Se buscan parejas de vecinos mas cercanos con cluster distinto
# y se calcula el "margen": distancia al 2do centroide menos la
# distancia al 1er centroide. Margen pequeno = el estudiante esta
# en la frontera entre dos grupos.
# ------------------------------------------------------------

def actividad_7_vecinos_frontera(df, X_escalado, modelo, etiquetas):

    print("\n" + "=" * 60)
    print("ACTIVIDAD 7: SIMILARES EN GRUPOS DIFERENTES")
    print("=" * 60)

    # Se revisan los 5 vecinos mas cercanos de cada estudiante
    # (el primero es el propio estudiante, con distancia 0).
    distancias_vecino, indices = NearestNeighbors(
        n_neighbors=6
    ).fit(X_escalado).kneighbors(X_escalado)

    d_centroides = np.sort(modelo.transform(X_escalado), axis=1)

    margen = d_centroides[:, 1] - d_centroides[:, 0]

    # Pares (i, j) muy cercanos entre si pero con cluster distinto.
    pares = []

    for i in range(len(X_escalado)):

        for pos in range(1, 6):

            j = indices[i, pos]

            if etiquetas[i] != etiquetas[j] and i < j:

                pares.append((i, j, distancias_vecino[i, pos]))

    pares.sort(key=lambda t: t[2])

    print("Pares de vecinos cercanos con cluster distinto:", len(pares))

    filas = []

    for i, j, dij in pares[:5]:

        print(
            f"{df.id_estudiante[i]} (cluster {etiquetas[i]}, "
            f"margen {margen[i]:.3f}) vs "
            f"{df.id_estudiante[j]} (cluster {etiquetas[j]}, "
            f"margen {margen[j]:.3f}) | distancia: {dij:.3f}"
        )

        filas.append({
            "estudiante_a": df.id_estudiante[i],
            "cluster_a": etiquetas[i],
            "margen_a": margen[i],
            "estudiante_b": df.id_estudiante[j],
            "cluster_b": etiquetas[j],
            "margen_b": margen[j],
            "distancia_entre_ellos": dij
        })

    pd.DataFrame(filas).to_csv(
        f"{CARPETA_RESULTADOS}/p08_act7_vecinos_frontera.csv",
        index=False
    )


# ------------------------------------------------------------
# 13. VISUALIZACION DE LOS SEGMENTOS
# ------------------------------------------------------------


def visualizar_segmentos(df, etiquetas):

    """
    Representa dos variables en un grafico de dispersion.

    Se utilizan horas de estudio y asistencia para facilitar
    la interpretacion visual.

    La visualizacion es una proyeccion de dos variables:
    los cuatro atributos siguen participando en K-Means.
    """

    plt.figure(figsize=(10, 6))

    dispersion = plt.scatter(
        df["horas_estudio"],
        df["asistencia"],
        c=etiquetas,
        cmap="viridis",
        alpha=0.75
    )

    plt.title("Segmentacion academica de SmartAcademy")
    plt.xlabel("Horas de estudio")
    plt.ylabel("Asistencia (%)")

    plt.colorbar(
        dispersion,
        label="Cluster"
    )

    plt.tight_layout()

    plt.savefig(
        f"{CARPETA_RESULTADOS}/p08_segmentacion_final.png",
        dpi=300
    )

    plt.close()


# ------------------------------------------------------------
# 14. EXPORTACION DEL REPORTE INTEGRAL
# ------------------------------------------------------------

def exportar_reporte(
    df,
    etiquetas,
    silhouette_individual,
    df_centroides,
    candidatos
):

    """
    Construye un archivo general con:
    - Identificador del estudiante.
    - Variables originales.
    - Cluster asignado.
    - Silhouette individual.
    - Indicador de posible caso atipico.
    """

    reporte = df.copy()

    reporte["cluster"] = etiquetas

    reporte["silhouette_individual"] = silhouette_individual

    # Recuperamos los identificadores de los candidatos.
    ids_atipicos = set(
        candidatos["id_estudiante"].tolist()
    )

    reporte["posible_atipico"] = (
        reporte["id_estudiante"].isin(ids_atipicos)
    )

    # Incorporamos el nombre descriptivo del cluster.
    # El nombre se asigna ordenando los centroides por
    # horas de estudio, de menor a mayor.

    orden = df_centroides.sort_values(
        "horas_estudio"
    )["cluster"].tolist()

    nombres = {}

    if len(orden) == 3:

        nombres = {
            orden[0]: "Requiere acompanamiento",
            orden[1]: "Compromiso intermedio",
            orden[2]: "Alto compromiso"
        }

    reporte["perfil_academico"] = (
        reporte["cluster"].map(nombres)
    )

    reporte.to_csv(
        f"{CARPETA_RESULTADOS}/p08_reporte_integral.csv",
        index=False
    )

    print("\nReporte integral exportado correctamente.")


# ------------------------------------------------------------
# 15. FUNCION PRINCIPAL
# ------------------------------------------------------------

def main():

    print("\n" + "=" * 60)
    print("SMARTACADEMY - PRACTICA 8")
    print("ANALISIS AVANZADO DE SEGMENTOS ACADEMICOS")
    print("=" * 60)

    # Paso 1: cargar datos.
    df = cargar_datos()

    # Paso 2: preparar datos.
    df, X = preparar_datos(df)

    # Paso 3: comparar inicializaciones.
    resultados_init, modelos_init, X_std, scaler_std = (
        comparar_inicializaciones(X)
    )

    # Paso 4: comparar escaladores.
    resultados_scaler, modelos_scaler = comparar_escaladores(X)

    # Paso 5: analizar estabilidad.
    df_ari, particiones = analizar_estabilidad(X_std)

    # Paso 6: entrenar modelo final.
    (
        modelo,
        escalador,
        X_escalado,
        etiquetas,
        silhouette_individual
    ) = entrenar_modelo_final(X)

    # Paso 7: analizar centroides y perfiles.
    df_centroides, perfiles = analizar_centroides(
        df,
        modelo,
        escalador,
        etiquetas
    )

    # Paso 8: detectar posibles casos atipicos.
    (
        df_atipicos,
        candidatos,
        umbral
    ) = detectar_atipicos(
        df,
        X_escalado,
        modelo,
        etiquetas,
        silhouette_individual
    )

    # Paso 9: visualizar los segmentos.
    visualizar_segmentos(
        df,
        etiquetas
    )

    # Paso 10: exportar reporte integral.
    exportar_reporte(
        df,
        etiquetas,
        silhouette_individual,
        df_centroides,
        candidatos
    )

    # --------------------------------------------------------
    # ACTIVIDADES PARA LOS ALUMNOS
    # --------------------------------------------------------

    actividad_1_variar_clusters(df, X)                               # ACTIVIDAD 1
    actividad_2_variar_n_init(X)                                     # ACTIVIDAD 2
    actividad_3_nueva_variable(df)                                   # ACTIVIDAD 3
    actividad_4_variar_percentil(X_escalado, modelo, etiquetas)      # ACTIVIDAD 4
    actividad_5_perfiles_escaladores(df, modelos_scaler)             # ACTIVIDAD 5
    actividad_6_estabilidad_semillas(X_std, df_ari)                  # ACTIVIDAD 6
    actividad_7_vecinos_frontera(df, X_escalado, modelo, etiquetas)  # ACTIVIDAD 7

    # --------------------------------------------------------
    # RESUMEN FINAL
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("RESUMEN FINAL DE LA PRACTICA 8")
    print("=" * 60)

    print("Estudiantes analizados:", len(df))

    print("Clusters:", NUM_CLUSTERS)

    print(
        "Silhouette final:",
        round(
            silhouette_score(X_escalado, etiquetas),
            4
        )
    )

    print("Posibles atipicos:", len(candidatos))

    print(
        "Porcentaje de posibles atipicos:",
        round(
            len(candidatos) / len(df) * 100,
            2
        ),
        "%"
    )

    print("\nArchivos generados en:")
    print(CARPETA_RESULTADOS)

    print("\nPractica 8 finalizada correctamente.")


# ------------------------------------------------------------
# 16. PUNTO DE ENTRADA DEL PROGRAMA
# ------------------------------------------------------------

if __name__ == "__main__":

    main()

#Se creará una carpeta llamada resultados_p08.

#   Archivo	                        Contenido
#p08_inicializaciones.csv	        Resultados de k-means++ y random.
#p08_comparacion_escaladores.csv	Resultados de StandardScaler y MinMaxScaler.
#p08_estabilidad_ari.csv	        Comparaciones de estabilidad entre semillas.
#p08_centroides.csv	                Centroides en unidades originales.
#p08_perfiles.csv	                Características promedio de cada segmento.
#p08_analisis_distancias.csv	    Distancia de cada estudiante a su centroide.
#p08_posibles_atipicos.csv	        Estudiantes que superan el umbral de distancia.
#p08_reporte_integral.csv	        Reporte general de segmentación.
#
# Archivos nuevos de las actividades:
#p08_act1_variar_k.csv              Actividad 1: K = 2, 3, 4.
#p08_act2_variar_n_init.csv         Actividad 2: n_init = 5, 20, 50.
#p08_act3_nueva_variable.csv        Actividad 3: 4 vs 5 variables.
#p08_act4_percentiles.csv           Actividad 4: P90, P95, P97.5.
#p08_act5_perfiles_*.csv            Actividad 5: perfiles por escalador.
#p08_act7_vecinos_frontera.csv      Actividad 7: pares en la frontera.

#También se generarán gráficas PNG para comparar inicializaciones, escaladores, estabilidad, distancias y segmentos.

#7. Conceptos que deben interpretar los estudiantes

#A. Inicialización
#- k-means++ busca una selección inicial de centroides más adecuada.
#- random selecciona centroides aleatoriamente.
#- Una inercia menor indica menor dispersión interna, pero no debe utilizarse sola para determinar qué modelo es mejor.

#B. Comparación de escaladores
#StandardScaler y MinMaxScaler modifican las distancias entre los estudiantes. Por ello, es posible que un 
#mismo estudiante cambie de segmento.

#No se debe comparar directamente la inercia de ambos escaladores como si estuviera expresada en la misma escala. 
#Para esta comparación resulta más informativo observar el Silhouette Score y los perfiles resultantes.

#C. Estabilidad con ARI
#Resultado ARI	Interpretación general
#Cercano a 1	Alta coincidencia entre agrupaciones.
#Cercano a 0	Coincidencia similar a la esperada por azar.
#Menor que 0	Coincidencia inferior a la esperada por azar.

#D. Posibles estudiantes atípicos

#El programa utiliza el percentil 95 de las distancias a los centroides. 
#Los estudiantes que superen ese valor se marcan para revisión.

#Esto no significa que tengan bajo rendimiento, que vayan a reprobar o que pertenezcan a un grupo incorrecto. 
#Solamente indica que su comportamiento es relativamente distante del centro de su segmento.

#8. Actividades para los alumnos
#1. Cambiar NUM_CLUSTERS = 3 por 2 y posteriormente por 4. Comparar los resultados.
#2. Cambiar NUM_INICIALIZACIONES = 20 por 5 y 50.
#3. Agregar una nueva variable llamada entregas_puntuales y evaluar su efecto.
#4. Modificar PERCENTIL_ATIPICOS = 95 por 90 y 97.5. Comparar cuántos estudiantes se identifican.
#5. Comparar los perfiles académicos obtenidos con StandardScaler y MinMaxScaler.
#6. Explicar si los segmentos son estables entre las diferentes semillas.
#7. Investigar por qué dos estudiantes con características similares podrían terminar en grupos diferentes.