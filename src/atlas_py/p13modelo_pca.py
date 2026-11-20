
# ============================================================
# PROYECTO: SMARTACADEMY
# ETAPA 4: REDUCCIÓN DE DIMENSIONALIDAD
# ALGORITMO: PCA

#Hasta ahora hemos construido tres elementos importantes:
#1. Generamos un conjunto de datos de 500 estudiantes.
#2. Entrenamos un modelo supervisado para estimar la aprobación.
#3. Aplicamos K-Means para descubrir perfiles académicos.
#Ahora utilizaremos PCA para analizar y visualizar esos perfiles desde una perspectiva multidimensional.

# Archivo: p13modelo_pca.py

# Objetivo:
#Implementar el algoritmo Principal Component Analysis (PCA) para reducir 
#las cuatro características académicas de SmartAcademy a dos componentes principales.

#Esto permitirá:

#- Reducir la dimensionalidad del conjunto de datos.
#- Identificar qué características explican mayor variabilidad.
#- Visualizar los perfiles académicos en dos dimensiones.
#- Analizar si los grupos descubiertos por K-Means presentan separación.
#- Generar información que posteriormente podremos integrar en una interfaz gráfica.

#¿Qué información obtendremos?
#Nuestro análisis generará tres resultados principales:

#Elemento               	Utilidad
#Varianza explicada 	    Mide cuánta información conserva cada componente.
#Componentes principales	Permiten representar los datos en nuevas dimensiones.
#Gráfica PCA	            Permite visualizar los estudiantes y sus grupos.

#Por ejemplo, si obtenemos:
#- PC1: 65 %.
#- PC2: 20 %.
#La representación bidimensional conservaría el 85 % de la variabilidad original.

#Estos porcentajes son únicamente ilustrativos; obtendremos los valores reales al ejecutar el programa.

# Tecnologías:
# - Python
# - Pandas
# - NumPy
# - Matplotlib
# - Scikit-learn
# - Joblib
# ============================================================


# ============================================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


# ============================================================
# 2. CONFIGURACIÓN DEL PROYECTO
# ============================================================

# Identificar la ubicación de este archivo.
DIRECTORIO_BASE = os.path.dirname(
    os.path.abspath(__file__)
)

# Archivo original con los datos de los estudiantes.
ARCHIVO_DATOS = os.path.join(
    DIRECTORIO_BASE,
    "resultados_p10",
    "smartacademy_datos.csv"
)

# Archivo generado durante la etapa K-Means.
ARCHIVO_SEGMENTACION = os.path.join(
    DIRECTORIO_BASE,
    "resultados_p10",
    "estudiantes_segmentados.csv"
)

# Modelo K-Means que contiene el escalador.
ARCHIVO_MODELO_KMEANS = os.path.join(
    DIRECTORIO_BASE,
    "resultados_p10",
    "modelos",
    "modelo_kmeans_smartacademy.joblib"
)

# Directorio para almacenar resultados.
DIRECTORIO_RESULTADOS = os.path.join(
    DIRECTORIO_BASE,
    "resultados_p10"
)

# Directorio para guardar modelos.
DIRECTORIO_MODELOS = os.path.join(
    DIRECTORIO_RESULTADOS,
    "modelos"
)

os.makedirs(DIRECTORIO_RESULTADOS, exist_ok=True)
os.makedirs(DIRECTORIO_MODELOS, exist_ok=True)


# ============================================================
# 3. DEFINIR LAS CARACTERÍSTICAS
# ============================================================

# Debemos utilizar exactamente las mismas características
# que fueron utilizadas durante el entrenamiento de K-Means.

CARACTERISTICAS = [
    "horas_estudio",
    "asistencia",
    "actividades",
    "participacion"
]

# Número de componentes que queremos obtener.
NUM_COMPONENTES = 2


# ============================================================
# 4. CARGAR INFORMACIÓN
# ============================================================

def cargar_datos():

    print("\nCARGANDO INFORMACIÓN SMARTACADEMY...")

    # Verificar que existan los archivos necesarios.
    for archivo in [
        ARCHIVO_DATOS,
        ARCHIVO_SEGMENTACION,
        ARCHIVO_MODELO_KMEANS
    ]:

        if not os.path.exists(archivo):

            raise FileNotFoundError(
                f"No se encontró el archivo: {archivo}"
            )

    # Leer los datos originales.
    datos = pd.read_csv(ARCHIVO_DATOS)

    # Leer los resultados de K-Means.
    segmentacion = pd.read_csv(
        ARCHIVO_SEGMENTACION
    )

    # Recuperar el modelo y el escalador.
    paquete_kmeans = joblib.load(
        ARCHIVO_MODELO_KMEANS
    )

    escalador = paquete_kmeans["escalador"]

    print("Datos originales cargados.")
    print("Segmentación cargada.")
    print("Escalador recuperado correctamente.")

    # Validar columnas.
    columnas_requeridas = [
        "id_estudiante",
        *CARACTERISTICAS
    ]

    for columna in columnas_requeridas:

        if columna not in datos.columns:

            raise ValueError(
                f"Falta la columna: {columna}"
            )

    columnas_segmentacion = [
        "id_estudiante",
        "cluster",
        "perfil"
    ]

    for columna in columnas_segmentacion:

        if columna not in segmentacion.columns:

            raise ValueError(
                f"Falta la columna: {columna}"
            )

    # Comprobar que no existan registros duplicados
    # en los identificadores de estudiantes.
    if datos["id_estudiante"].duplicated().any():

        raise ValueError(
            "Existen identificadores duplicados en los datos."
        )

    if segmentacion["id_estudiante"].duplicated().any():

        raise ValueError(
            "Existen identificadores duplicados en la segmentación."
        )

    # Verificar que los dos archivos contienen los mismos
    # estudiantes.
    if set(datos["id_estudiante"]) != set(
        segmentacion["id_estudiante"]
    ):

        raise ValueError(
            "Los estudiantes de ambos archivos no coinciden."
        )

    # Incorporar las etiquetas de K-Means a los datos.
    datos = datos.merge(
        segmentacion[
            [
                "id_estudiante",
                "cluster",
                "perfil"
            ]
        ],
        on="id_estudiante",
        how="inner",
        validate="one_to_one"
    )

    # Verificar valores nulos.
    if datos[
        CARACTERISTICAS + ["cluster", "perfil"]
    ].isnull().any().any():

        raise ValueError(
            "Existen valores nulos en la información."
        )

    # Comprobar que los valores sean numéricos.
    for columna in CARACTERISTICAS:

        datos[columna] = pd.to_numeric(
            datos[columna],
            errors="coerce"
        )

    if datos[CARACTERISTICAS].isnull().any().any():

        raise ValueError(
            "Existen valores no numéricos."
        )

    # Verificar valores infinitos.
    if not np.isfinite(
        datos[CARACTERISTICAS].to_numpy()
    ).all():

        raise ValueError(
            "Existen valores infinitos."
        )

    print("\nTotal de estudiantes:", len(datos))

    return datos, escalador


# ============================================================
# 5. PREPARAR LOS DATOS
# ============================================================

def preparar_datos(datos, escalador):

    print("\nPREPARANDO CARACTERÍSTICAS...")

    # Seleccionar las cuatro variables originales.
    X = datos[CARACTERISTICAS].copy()

    # Aplicar exactamente el escalador utilizado
    # durante la etapa K-Means.
    #
    # IMPORTANTE:
    # Aquí NO utilizamos fit_transform().
    #
    # Utilizamos transform() porque el escalador ya fue
    # entrenado anteriormente con los datos originales.

    X_escalado = escalador.transform(X)

    print("Escalamiento aplicado correctamente.")

    return X_escalado


# ============================================================
# 6. APLICAR PCA
# ============================================================

def aplicar_pca(X_escalado):

    print("\nAPLICANDO ANÁLISIS DE COMPONENTES PRINCIPALES...")

    # Crear el objeto PCA.
    #
    # n_components=2:
    # Queremos reducir las cuatro dimensiones originales
    # a solamente dos componentes principales.

    modelo_pca = PCA(
        n_components=NUM_COMPONENTES
    )

    # Entrenar PCA y transformar los datos.
    componentes = modelo_pca.fit_transform(
        X_escalado
    )

    print("PCA ejecutado correctamente.")

    print(
        "Dimensiones originales:",
        X_escalado.shape[1]
    )

    print(
        "Dimensiones resultantes:",
        componentes.shape[1]
    )

    return modelo_pca, componentes


# ============================================================
# 7. ANALIZAR LA VARIANZA EXPLICADA
# ============================================================

def analizar_varianza(modelo_pca):

    print("\nANÁLISIS DE VARIANZA EXPLICADA")

    # Obtener el porcentaje de información que conserva
    # cada componente principal.
    varianza = modelo_pca.explained_variance_ratio_

    # Calcular la varianza acumulada.
    varianza_acumulada = np.cumsum(varianza)

    for i, valor in enumerate(varianza):

        print(
            f"PC{i + 1}: {valor * 100:.2f}%"
        )

    print(
        "\nVarianza acumulada:",
        f"{varianza_acumulada[-1] * 100:.2f}%"
    )

    # Crear tabla de resultados.
    tabla_varianza = pd.DataFrame({

        "componente": [
            "PC1",
            "PC2"
        ],

        "varianza_explicada": varianza,

        "varianza_porcentual": varianza * 100,

        "varianza_acumulada": varianza_acumulada * 100

    })

    # Exportar resultados.
    archivo = os.path.join(
        DIRECTORIO_RESULTADOS,
        "varianza_explicada_pca.csv"
    )

    tabla_varianza.to_csv(
        archivo,
        index=False
    )

    print("\nTabla de varianza exportada:", archivo)

    return tabla_varianza


# ============================================================
# 8. ANALIZAR LAS CARGAS DE LOS COMPONENTES
# ============================================================

def analizar_cargas(modelo_pca):

    print("\nANÁLISIS DE CARGAS DE LOS COMPONENTES")

    # Cada fila representa un componente principal.
    # Cada columna representa una variable original.
    #
    # Las cargas indican cómo contribuye cada variable
    # a la construcción de cada componente.

    cargas = pd.DataFrame(
        modelo_pca.components_,
        columns=CARACTERISTICAS,
        index=["PC1", "PC2"]
    )

    print(cargas.round(4))

    # Exportar las cargas.
    archivo = os.path.join(
        DIRECTORIO_RESULTADOS,
        "cargas_componentes_pca.csv"
    )

    cargas.to_csv(archivo)

    print("\nCargas exportadas:", archivo)

    return cargas


# ============================================================
# 9. CREAR DATAFRAME CON LOS COMPONENTES
# ============================================================

def crear_resultados(datos, componentes):

    print("\nGENERANDO RESULTADOS PCA...")

    # Crear una copia de los datos.
    resultados = datos.copy()

    # Agregar los dos componentes principales.
    resultados["PC1"] = componentes[:, 0]

    resultados["PC2"] = componentes[:, 1]

    # Exportar los resultados.
    archivo = os.path.join(
        DIRECTORIO_RESULTADOS,
        "estudiantes_pca.csv"
    )

    resultados.to_csv(
        archivo,
        index=False
    )

    print("Archivo generado:", archivo)

    return resultados


# ============================================================
# 10. GRAFICAR VARIANZA EXPLICADA
# ============================================================

def graficar_varianza(tabla_varianza):

    print("\nGENERANDO GRÁFICA DE VARIANZA...")

    plt.figure(figsize=(9, 5))

    plt.bar(
        tabla_varianza["componente"],
        tabla_varianza["varianza_porcentual"]
    )

    plt.title(
        "SMARTACADEMY - VARIANZA EXPLICADA POR PCA"
    )

    plt.xlabel("Componentes principales")

    plt.ylabel("Varianza explicada (%)")

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    archivo = os.path.join(
        DIRECTORIO_RESULTADOS,
        "grafica_varianza_pca.png"
    )

    plt.savefig(
        archivo,
        dpi=300
    )

    plt.show()

    print("Gráfica guardada:", archivo)


# ============================================================
# 11. GRAFICAR ESTUDIANTES EN DOS DIMENSIONES
# ============================================================

def graficar_segmentacion_pca(resultados):

    print("\nGENERANDO VISUALIZACIÓN PCA...")

    plt.figure(figsize=(11, 7))

    # Obtener los perfiles encontrados por K-Means.
    perfiles = resultados["perfil"].unique()

    # Cada perfil tendrá su propia agrupación visual.
    for perfil in perfiles:

        grupo = resultados[
            resultados["perfil"] == perfil
        ]

        plt.scatter(

            grupo["PC1"],

            grupo["PC2"],

            label=perfil,

            alpha=0.75,

            s=45

        )

    plt.title(
        "SMARTACADEMY - SEGMENTACIÓN MEDIANTE PCA"
    )

    plt.xlabel("Componente principal 1 (PC1)")

    plt.ylabel("Componente principal 2 (PC2)")

    plt.axhline(
        y=0,
        linewidth=0.7,
        alpha=0.4
    )

    plt.axvline(
        x=0,
        linewidth=0.7,
        alpha=0.4
    )

    plt.legend(
        title="Perfiles académicos"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    archivo = os.path.join(
        DIRECTORIO_RESULTADOS,
        "grafica_segmentacion_pca.png"
    )

    plt.savefig(
        archivo,
        dpi=300
    )

    plt.show()

    print("Gráfica guardada:", archivo)


# ============================================================
# 12. GUARDAR EL MODELO PCA
# ============================================================

def guardar_modelo_pca(modelo_pca, escalador):

    print("\nGUARDANDO MODELO PCA...")

    # Guardamos el PCA junto con el escalador.
    #
    # Esto permitirá transformar nuevos estudiantes
    # utilizando exactamente el mismo procedimiento.

    paquete_pca = {

        "modelo_pca": modelo_pca,

        "escalador": escalador,

        "caracteristicas": CARACTERISTICAS,

        "numero_componentes": NUM_COMPONENTES

    }

    archivo = os.path.join(
        DIRECTORIO_MODELOS,
        "modelo_pca_smartacademy.joblib"
    )

    joblib.dump(
        paquete_pca,
        archivo
    )

    print("Modelo guardado correctamente.")

    print(archivo)


# ============================================================
# 13. FUNCIÓN PRINCIPAL
# ============================================================

def main():

    print("\n" + "=" * 60)

    print("SMARTACADEMY")

    print("ETAPA 4 - ANÁLISIS DE COMPONENTES PRINCIPALES")

    print("=" * 60)

    # PASO 1:
    # Recuperar datos, segmentación y escalador.
    datos, escalador = cargar_datos()

    # PASO 2:
    # Transformar las características originales.
    X_escalado = preparar_datos(
        datos,
        escalador
    )

    # PASO 3:
    # Aplicar PCA.
    modelo_pca, componentes = aplicar_pca(
        X_escalado
    )

    # PASO 4:
    # Analizar la varianza explicada.
    tabla_varianza = analizar_varianza(
        modelo_pca
    )

    # PASO 5:
    # Analizar las cargas de cada componente.
    cargas = analizar_cargas(
        modelo_pca
    )

    # PASO 6:
    # Generar archivo con PC1 y PC2.
    resultados = crear_resultados(
        datos,
        componentes
    )

    # PASO 7:
    # Graficar varianza.
    graficar_varianza(
        tabla_varianza
    )

    # PASO 8:
    # Visualizar los grupos.
    graficar_segmentacion_pca(
        resultados
    )

    # PASO 9:
    # Guardar modelo para futuras predicciones.
    guardar_modelo_pca(
        modelo_pca,
        escalador
    )

    print("\n" + "=" * 60)

    print("ETAPA 4 FINALIZADA CORRECTAMENTE")

    print("=" * 60)


# ============================================================
# 14. EJECUTAR EL PROGRAMA
# ============================================================

if __name__ == "__main__":

    main()


#¿Cómo interpretar los resultados?

#A. Varianza explicada

#Si obtienes, por ejemplo:

#Componente	    Varianza
#PC1        	60 %
#PC2        	25 %
#Total      	85 %

#Significa que los dos componentes conservan el 85 % de la variabilidad de las cuatro variables originales.

#B. Cargas de los componentes

#El archivo cargas_componentes_pca.csv permitirá estudiar qué variables contribuyen más a cada componente.

#Por ejemplo, si PC1 tiene cargas altas en horas de estudio y actividades, podríamos interpretar 
#que representa una dimensión relacionada con la dedicación académica. 
#Esta interpretación deberá basarse en los resultados reales, no únicamente en el signo de las cargas.

#C. Gráfica de segmentación PCA

#Cada punto representa un estudiante. Los colores representan los perfiles asignados previamente por K-Means.

#Si los grupos aparecen separados, tendremos evidencia visual de que sus características presentan 
#diferencias en las dos dimensiones representadas. Si se superponen, no significa automáticamente 
#que K-Means haya fallado: puede ocurrir que la separación se encuentre en dimensiones que PC1 y PC2 
#no representan completamente.