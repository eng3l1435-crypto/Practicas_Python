
# ============================================================
# PROYECTO: SMARTACADEMY
# ETAPA 3: APRENDIZAJE NO SUPERVISADO
# ALGORITMO: K-MEANS
#
# Archivo: p12modelo_kmeans.py
#
# Objetivo:
# Identificar grupos de estudiantes con características
# académicas similares mediante aprendizaje no supervisado.

#1. Objetivo

#Implementar un modelo de aprendizaje no supervisado mediante el algoritmo K-Means, 
#para identificar grupos de estudiantes con características académicas similares.

#A diferencia de la etapa anterior, aquí no vamos a predecir si un estudiante aprueba o reprueba. 
#Buscaremos descubrir patrones de comportamiento a partir de sus características.

#Por ejemplo, SmartAcademy podría identificar:

#- Estudiantes que requieren acompañamiento académico.
#- Estudiantes con compromiso intermedio.
#- Estudiantes con alto compromiso académico.

#Importante: K-Means no conoce estas categorías previamente. Las descubre matemáticamente; 
#nosotros interpretaremos posteriormente los grupos obtenidos.

#2. ¿Qué variables utilizaremos?

#Trabajaremos con las cuatro características del conjunto de datos:

#Variable	        Descripción
#horas_estudio	    Horas dedicadas al estudio.
#asistencia	        Porcentaje de asistencia.
#actividades	    Actividades académicas realizadas.
#participacion	    Nivel de participación.

#No utilizaremos aprueba ni id_estudiante para entrenar el algoritmo.

#Es importante porque queremos descubrir grupos sin proporcionarle previamente la respuesta correcta.

#3. ¿Cómo funciona K-Means?

#K-Means busca dividir los registros en K grupos o clústeres.

#Su funcionamiento general es:

#1. Seleccionar K Definimos cuántos grupos queremos evaluar.

#2. Inicializar centroides El algoritmo establece puntos iniciales que representan los grupos.

#3. Calcular distancias Cada estudiante se asigna al centroide más cercano.

#4. Actualizar centroides Se recalcula la posición central de cada grupo.

#5. Repetir El proceso continúa hasta que los grupos se estabilizan o se alcanza el límite de iteraciones.


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

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# ============================================================
# 2. CONFIGURACIÓN DEL PROYECTO
# ============================================================

RUTA_PROYECTO = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DIRECTORIO_RESULTADOS = os.path.join(
    RUTA_PROYECTO,
    "unidad3_fia",
    "resultados_p10"
)

# Archivo generado en la etapa 1.
ARCHIVO_DATOS = os.path.join(
    DIRECTORIO_RESULTADOS,
    "smartacademy_datos.csv"
)

# Directorio para guardar los modelos.
DIRECTORIO_MODELOS = os.path.join(
    DIRECTORIO_RESULTADOS,
    "modelos"
)

# Crear directorios si todavía no existen.
os.makedirs(DIRECTORIO_RESULTADOS, exist_ok=True)
os.makedirs(DIRECTORIO_MODELOS, exist_ok=True)

# Semilla para reproducir los resultados.
SEMILLA = 42

# Número de grupos que vamos a evaluar.
VALORES_K = range(2, 7)

# Variables que utilizará K-Means.
CARACTERISTICAS = [
    "horas_estudio",
    "asistencia",
    "actividades",
    "participacion"
]


# ============================================================
# 3. CARGAR EL DATASET
# ============================================================

def cargar_datos():

    print("\nCARGANDO DATASET SMARTACADEMY...")

    # Verificar que exista el archivo.
    if not os.path.exists(ARCHIVO_DATOS):
        raise FileNotFoundError(
            f"No se encontró el archivo: {ARCHIVO_DATOS}"
        )

    # Leer el archivo CSV.
    datos = pd.read_csv(ARCHIVO_DATOS)

    # Verificar columnas necesarias.
    columnas_requeridas = [
        "id_estudiante",
        *CARACTERISTICAS
    ]

    columnas_faltantes = [
        columna
        for columna in columnas_requeridas
        if columna not in datos.columns
    ]

    if columnas_faltantes:
        raise ValueError(
            f"Faltan columnas: {columnas_faltantes}"
        )

    # Verificar valores nulos.
    if datos[columnas_requeridas].isnull().any().any():
        raise ValueError(
            "El dataset contiene valores nulos."
        )

    # Verificar que las características sean numéricas.
    for columna in CARACTERISTICAS:

        datos[columna] = pd.to_numeric(
            datos[columna],
            errors="coerce"
        )

    # Verificar nuevamente valores no numéricos.
    if datos[CARACTERISTICAS].isnull().any().any():
        raise ValueError(
            "Existen valores no numéricos en las características."
        )

    # Verificar valores infinitos.
    if not np.isfinite(
        datos[CARACTERISTICAS].to_numpy()
    ).all():
        raise ValueError(
            "Existen valores infinitos en el dataset."
        )

    print("Dataset cargado correctamente.")

    print("Total de estudiantes:", len(datos))

    return datos


# ============================================================
# 4. PREPARAR LAS CARACTERÍSTICAS
# ============================================================

def preparar_datos(datos):

    print("\nPREPARANDO CARACTERÍSTICAS...")

    # Seleccionar exclusivamente las cuatro variables.
    X = datos[CARACTERISTICAS].copy()

    # K-Means utiliza distancias.
    # Por eso necesitamos que las variables tengan escalas
    # comparables.
    #
    # StandardScaler transforma cada variable para que tenga:
    # - Media aproximadamente igual a 0.
    # - Desviación estándar aproximadamente igual a 1.

    escalador = StandardScaler()

    X_escalado = escalador.fit_transform(X)

    print("Características utilizadas:")

    for caracteristica in CARACTERISTICAS:
        print(" -", caracteristica)

    print("\nEscalamiento completado.")

    return X, X_escalado, escalador


# ============================================================
# 5. EVALUAR DIFERENTES VALORES DE K
# ============================================================

def evaluar_k(X_escalado):

    print("\nEVALUANDO DIFERENTES CANTIDADES DE GRUPOS...")

    resultados = []

    # Probar K = 2, 3, 4, 5 y 6.
    for k in VALORES_K:

        # Crear el modelo K-Means.
        modelo = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=20,
            random_state=SEMILLA
        )

        # Entrenar el algoritmo.
        etiquetas = modelo.fit_predict(X_escalado)

        # Inercia: suma de distancias cuadráticas
        # de cada observación a su centroide.
        inercia = modelo.inertia_

        # Silhouette: mide qué tan bien separados
        # están los grupos.
        silhouette = silhouette_score(
            X_escalado,
            etiquetas
        )

        resultados.append({
            "k": k,
            "inercia": inercia,
            "silhouette": silhouette
        })

        print("\nNúmero de grupos:", k)
        print("Inercia:", round(inercia, 4))
        print("Silhouette:", round(silhouette, 4))

    # Convertir los resultados a DataFrame.
    resultados_df = pd.DataFrame(resultados)

    # Exportar resultados.
    archivo = os.path.join(
        DIRECTORIO_RESULTADOS,
        "evaluacion_kmeans.csv"
    )

    resultados_df.to_csv(
        archivo,
        index=False
    )

    print("\nEvaluación exportada:", archivo)

    return resultados_df


# ============================================================
# 6. GRAFICAR EL MÉTODO DEL CODO
# ============================================================

def graficar_codo(resultados):

    print("\nGENERANDO GRÁFICA DEL CODO...")

    plt.figure(figsize=(9, 5))

    plt.plot(
        resultados["k"],
        resultados["inercia"],
        marker="o",
        linewidth=2
    )

    plt.title(
        "SMARTACADEMY - MÉTODO DEL CODO"
    )

    plt.xlabel("Número de grupos (K)")
    plt.ylabel("Inercia")

    plt.xticks(list(VALORES_K))

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    archivo = os.path.join(
        DIRECTORIO_RESULTADOS,
        "grafica_metodo_codo.png"
    )

    plt.savefig(
        archivo,
        dpi=300
    )

    plt.show()

    print("Gráfica guardada:", archivo)


# ============================================================
# 7. GRAFICAR COEFICIENTE SILHOUETTE
# ============================================================

def graficar_silhouette(resultados):

    print("\nGENERANDO GRÁFICA SILHOUETTE...")

    plt.figure(figsize=(9, 5))

    plt.plot(
        resultados["k"],
        resultados["silhouette"],
        marker="o",
        linewidth=2
    )

    plt.title(
        "SMARTACADEMY - EVALUACIÓN SILHOUETTE"
    )

    plt.xlabel("Número de grupos (K)")
    plt.ylabel("Coeficiente Silhouette")

    plt.xticks(list(VALORES_K))

    plt.grid(True, alpha=0.3)

    archivo = os.path.join(
        DIRECTORIO_RESULTADOS,
        "grafica_silhouette.png"
    )

    plt.tight_layout()

    plt.savefig(
        archivo,
        dpi=300
    )

    plt.show()

    print("Gráfica guardada:", archivo)


# ============================================================
# 8. SELECCIONAR EL MEJOR K
# ============================================================

def seleccionar_k(resultados):

    # Elegir el K con el mayor coeficiente Silhouette.
    mejor_fila = resultados.loc[
        resultados["silhouette"].idxmax()
    ]

    mejor_k = int(mejor_fila["k"])

    print("\nMEJOR CONFIGURACIÓN ENCONTRADA")

    print("Número de grupos:", mejor_k)

    print(
        "Silhouette:",
        round(mejor_fila["silhouette"], 4)
    )

    return mejor_k


# ============================================================
# 9. INTERPRETAR LOS GRUPOS
# ============================================================

def interpretar_grupos(modelo, escalador):

    # Los centroides están en escala estandarizada.
    # Los convertimos a las unidades originales para
    # facilitar su interpretación.

    centroides_originales = escalador.inverse_transform(
        modelo.cluster_centers_
    )

    centroides = pd.DataFrame(
        centroides_originales,
        columns=CARACTERISTICAS
    )

    # Calculamos el promedio de las cuatro características
    # para ordenar los grupos de menor a mayor compromiso.
    #
    # Esta puntuación es únicamente una herramienta
    # descriptiva, no una calificación académica.

    puntuaciones = modelo.cluster_centers_.mean(axis=1)

    orden = np.argsort(puntuaciones)

    # Asignar nombres según el número de grupos.
    if modelo.n_clusters == 3:

        nombres = [
            "Requiere acompañamiento",
            "Compromiso intermedio",
            "Alto compromiso"
        ]

    else:

        nombres = [
            f"Perfil {i + 1}"
            for i in range(modelo.n_clusters)
        ]

    # Asociar el identificador original del clúster
    # con el nombre semántico correspondiente.
    mapa_perfiles = {}

    for posicion, cluster_original in enumerate(orden):

        mapa_perfiles[int(cluster_original)] = nombres[
            posicion
        ]

    # Mostrar centroides.
    print("\nPERFILES ACADÉMICOS IDENTIFICADOS")

    for cluster in range(modelo.n_clusters):

        print("\nCluster:", cluster)

        print(
            "Perfil:",
            mapa_perfiles[cluster]
        )

        print(
            centroides.iloc[cluster].round(2).to_dict()
        )

    return mapa_perfiles, centroides


# ============================================================
# 10. ENTRENAR EL MODELO FINAL
# ============================================================

def entrenar_modelo(X_escalado, X_original, datos, mejor_k):

    print("\nENTRENANDO MODELO K-MEANS FINAL...")

    modelo = KMeans(
        n_clusters=mejor_k,
        init="k-means++",
        n_init=20,
        random_state=SEMILLA
    )

    # Entrenar y asignar cada estudiante a un grupo.
    etiquetas = modelo.fit_predict(X_escalado)

    # Interpretar los grupos.
    # Se ajusta el escalador sobre las mismas variables
    # y el modelo se interpreta usando sus centroides.
    escalador = StandardScaler()
    escalador.fit(X_original)

    mapa_perfiles, centroides = interpretar_grupos(
        modelo,
        escalador
    )

    # Crear copia de los datos originales.
    resultados = datos.copy()

    # Agregar el identificador del grupo.
    resultados["cluster"] = etiquetas

    # Agregar el nombre del perfil.
    resultados["perfil"] = [
        mapa_perfiles[int(cluster)]
        for cluster in etiquetas
    ]

    # Exportar estudiantes segmentados.
    archivo_estudiantes = os.path.join(
        DIRECTORIO_RESULTADOS,
        "estudiantes_segmentados.csv"
    )

    resultados.to_csv(
        archivo_estudiantes,
        index=False
    )

    # Generar resumen de perfiles.
    resumen = resultados.groupby(
        ["cluster", "perfil"]
    )[CARACTERISTICAS].mean().round(2)

    archivo_resumen = os.path.join(
        DIRECTORIO_RESULTADOS,
        "resumen_perfiles.csv"
    )

    resumen.to_csv(archivo_resumen)

    print("\nDISTRIBUCIÓN DE ESTUDIANTES")

    print(
        resultados["perfil"].value_counts()
    )

    print("\nRESUMEN DE PERFILES")

    print(resumen)

    # Guardar modelo y escalador.
    paquete_modelo = {
        "modelo": modelo,
        "escalador": escalador,
        "caracteristicas": CARACTERISTICAS,
        "mapa_perfiles": mapa_perfiles,
        "numero_clusters": mejor_k,
        "semilla": SEMILLA
    }

    archivo_modelo = os.path.join(
        DIRECTORIO_MODELOS,
        "modelo_kmeans_smartacademy.joblib"
    )

    joblib.dump(
        paquete_modelo,
        archivo_modelo
    )

    print("\nMODELO GUARDADO CORRECTAMENTE")

    print(archivo_modelo)

    print("\nRESULTADOS EXPORTADOS")

    print(archivo_estudiantes)
    print(archivo_resumen)

    return modelo, escalador, resultados


# ============================================================
# 11. VISUALIZAR LOS GRUPOS
# ============================================================

def visualizar_grupos(resultados):

    print("\nGENERANDO VISUALIZACIÓN DE CLÚSTERES...")

    plt.figure(figsize=(10, 6))

    # Cada punto representa un estudiante.
    # Para esta gráfica utilizamos horas de estudio
    # y asistencia.
    #
    # Las otras dos variables sí participaron en K-Means,
    # aunque no se muestran en estos dos ejes.

    for perfil in resultados["perfil"].unique():

        grupo = resultados[
            resultados["perfil"] == perfil
        ]

        plt.scatter(
            grupo["horas_estudio"],
            grupo["asistencia"],
            label=perfil,
            alpha=0.7,
            s=35
        )

    plt.title(
        "SMARTACADEMY - SEGMENTACIÓN DE ESTUDIANTES"
    )

    plt.xlabel("Horas de estudio")

    plt.ylabel("Asistencia (%)")

    plt.legend()

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    archivo = os.path.join(
        DIRECTORIO_RESULTADOS,
        "segmentacion_estudiantes.png"
    )

    plt.savefig(
        archivo,
        dpi=300
    )

    plt.show()

    print("Gráfica guardada:", archivo)


# ============================================================
# 12. FUNCIÓN PRINCIPAL
# ============================================================

def main():

    print("\n" + "=" * 60)

    print("SMARTACADEMY")

    print("ETAPA 3 - APRENDIZAJE NO SUPERVISADO")

    print("=" * 60)

    # Paso 1: cargar datos.
    datos = cargar_datos()

    # Paso 2: preparar características.
    X_original, X_escalado, escalador = preparar_datos(
        datos
    )

    # Paso 3: evaluar diferentes valores de K.
    evaluacion = evaluar_k(X_escalado)

    # Paso 4: generar gráfica del codo.
    graficar_codo(evaluacion)

    # Paso 5: generar gráfica Silhouette.
    graficar_silhouette(evaluacion)

    # Paso 6: seleccionar mejor K.
    mejor_k = seleccionar_k(evaluacion)

    # Paso 7: entrenar modelo final.
    modelo, escalador_final, resultados = entrenar_modelo(
        X_escalado,
        X_original,
        datos,
        mejor_k
    )

    # Paso 8: visualizar los grupos.
    visualizar_grupos(resultados)

    print("\n" + "=" * 60)

    print("ETAPA 3 FINALIZADA CORRECTAMENTE")

    print("=" * 60)


# ============================================================
# 13. EJECUTAR EL PROGRAMA
# ============================================================

if __name__ == "__main__":

    main()

#¿Cómo interpretar los resultados?
#Hay dos indicadores fundamentales:

#A. Método del codo (inercia)

#La inercia mide qué tan cerca están los estudiantes de sus centroides. 
#Generalmente disminuye cuando incrementamos K.

#Buscamos el punto donde agregar más grupos deja de mejorar considerablemente la compactación.

#B. Coeficiente Silhouette

#Su valor se encuentra entre -1 y 1.

#Resultado	    Interpretación
#Cercano a 1	Grupos bien separados.
#Cercano a 0	Grupos que se superponen.
#Menor que 0	Posibles asignaciones inadecuadas.

#En nuestro programa seleccionamos inicialmente el K con el mayor Silhouette entre 2 y 6. 
#Esto es un criterio matemático de exploración, no una garantía de que sea la mejor 
#segmentación para las necesidades académicas.

#Nota: si el modelo encuentra tres grupos, los nombres serán los tres perfiles académicos definidos. 
#Si encuentra otra cantidad, utilizaremos nombres genéricos para evitar asignar categorías que no correspondan.