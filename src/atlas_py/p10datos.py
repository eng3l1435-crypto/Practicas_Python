# ============================================================
# PROYECTO: SMARTACADEMY
# PRÁCTICA 10 - SISTEMA INTELIGENTE
# ARCHIVO: p10datos.py
#
# OBJETIVO:
#Desarrollar un sistema inteligente que analice información académica de estudiantes para generar 
#una evaluación orientativa de su desempeño, identificar perfiles de comportamiento académico y proporcionar 
#recomendaciones personalizadas.

#El sistema integrará:

#1. Machine Learning supervisado: clasificación para estimar la posibilidad de aprobar.
#2. Machine Learning no supervisado: K-Means para identificar perfiles académicos.
#3. Reducción de dimensionalidad: PCA para visualizar los patrones de comportamiento.
#4. Sistema basado en reglas: generación de recomendaciones académicas.
#5. Interfaz gráfica: captura y presentación de resultados mediante Python y Tkinter.
#6. MongoDB Atlas: almacenamiento de estudiantes y resultados de análisis.

#Se desea desarrollar una herramienta que permita a los docentes identificar estudiantes que podrían 
#necesitar acompañamiento académico.

#El sistema recibirá los siguientes datos:
#   Variable	    Descripción

#Horas de estudio	Tiempo dedicado al estudio
#Asistencia	        Porcentaje de asistencia
#Actividades	    Actividades académicas realizadas
#Participación	    Porcentaje de participación
#Aprueba	        Variable objetivo para el entrenamiento

#A partir de esta información, el sistema realizará tres análisis independientes:

# ============================================================


# ------------------------------------------------------------
# 1. IMPORTACIÓN DE LIBRERÍAS
# ------------------------------------------------------------

# Pandas permite crear y manipular tablas de datos.
import pandas as pd

# NumPy permite generar datos numéricos y realizar operaciones.
import numpy as np

# Path permite administrar rutas de archivos.
from pathlib import Path


# ------------------------------------------------------------
# 2. CONFIGURACIÓN GENERAL
# ------------------------------------------------------------

# Establecemos una semilla para reproducir los mismos resultados.
# Esto es importante cuando realizamos experimentos de IA.

SEMILLA = 42

# Cantidad de estudiantes que tendrá nuestro conjunto de datos.

TOTAL_ESTUDIANTES = 500

# Ruta donde se almacenarán los resultados.

RUTA_RESULTADOS = Path("unidad3_fia/resultados_p10")

# Creamos la carpeta si todavía no existe.

RUTA_RESULTADOS.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 3. GENERACIÓN DE DATOS ACADÉMICOS
# ------------------------------------------------------------

def generar_datos_academicos():

    """
    Genera información académica simulada.

    Cada estudiante tendrá cuatro características:

    - Horas de estudio.
    - Asistencia.
    - Actividades realizadas.
    - Participación.

    También generaremos una variable objetivo:

    - Aprueba: 1 = Sí.
    - Aprueba: 0 = No.

    IMPORTANTE:
    Los datos son sintéticos y únicamente sirven
    para fines educativos y experimentales.
    """

    # Inicializamos el generador aleatorio.

    np.random.seed(SEMILLA)

    # --------------------------------------------------------
    # 3.1 Generamos las horas de estudio
    # --------------------------------------------------------

    horas_estudio = np.random.uniform(
        0,
        10,
        TOTAL_ESTUDIANTES
    )

    # --------------------------------------------------------
    # 3.2 Generamos el porcentaje de asistencia
    # --------------------------------------------------------

    asistencia = np.random.uniform(
        30,
        100,
        TOTAL_ESTUDIANTES
    )

    # --------------------------------------------------------
    # 3.3 Generamos las actividades realizadas
    # --------------------------------------------------------

    actividades = np.random.uniform(
        0,
        30,
        TOTAL_ESTUDIANTES
    )

    # --------------------------------------------------------
    # 3.4 Generamos el porcentaje de participación
    # --------------------------------------------------------

    participacion = np.random.uniform(
        0,
        100,
        TOTAL_ESTUDIANTES
    )

    # --------------------------------------------------------
    # 4. NORMALIZACIÓN Y CONSTRUCCIÓN DE PUNTUACIÓN ACADÉMICA
    # --------------------------------------------------------

    # Convertimos las horas de estudio a una escala de 0 a 100.
    # El máximo esperado es de 10 horas.

    horas_normalizadas = (horas_estudio / 10) * 100

    # Convertimos las actividades a una escala de 0 a 100.
    # El máximo esperado es de 30 actividades.

    actividades_normalizadas = (actividades / 30) * 100

    # Asistencia y participación ya están en escala de 0 a 100.

    # Calculamos una puntuación ponderada.

    puntuacion = (

        horas_normalizadas * 0.30 +

        asistencia * 0.25 +

        actividades_normalizadas * 0.25 +

        participacion * 0.20

    )

    # --------------------------------------------------------
    # 5. GENERACIÓN DE LA VARIABLE OBJETIVO
    # --------------------------------------------------------

    # Agregamos ruido aleatorio para simular variaciones
    # que pueden presentarse en el rendimiento académico.

    ruido = np.random.normal(

        loc=0,

        scale=4,

        size=TOTAL_ESTUDIANTES

    )

    # Clasificamos al estudiante:
    # 1 = Aprueba.
    # 0 = No aprueba.

    aprueba = (

        (puntuacion + ruido) >= 55

    ).astype(int)

    # --------------------------------------------------------
    # 6. CONSTRUCCIÓN DEL DATAFRAME
    # --------------------------------------------------------

    datos = pd.DataFrame({

        "horas_estudio": horas_estudio,

        "asistencia": asistencia,

        "actividades": actividades,

        "participacion": participacion,

        "aprueba": aprueba

    })

    # --------------------------------------------------------
    # 7. REDONDEO DE VALORES
    # --------------------------------------------------------

    # Redondeamos los valores para facilitar su lectura.

    columnas_numericas = [

        "horas_estudio",

        "asistencia",

        "actividades",

        "participacion"

    ]

    datos[columnas_numericas] = datos[
        columnas_numericas
    ].round(2)

    # --------------------------------------------------------
    # 8. GENERACIÓN DE IDENTIFICADORES
    # --------------------------------------------------------

    # Creamos un identificador para cada estudiante.

    datos.insert(

        0,

        "id_estudiante",

        range(1, TOTAL_ESTUDIANTES + 1)

    )

    return datos


# ------------------------------------------------------------
# 9. FUNCIÓN PRINCIPAL
# ------------------------------------------------------------

def main():

    print("\n" + "=" * 60)

    print("SMARTACADEMY - GENERACIÓN DE DATOS")

    print("=" * 60)

    # Generamos los datos.

    datos = generar_datos_academicos()

    # Mostramos los primeros registros.

    print("\nPRIMEROS 10 ESTUDIANTES:")

    print(datos.head(10))

    # Mostramos información general.

    print("\nDIMENSIONES DEL DATASET:")

    print(datos.shape)

    # Revisamos los tipos de datos.

    print("\nTIPOS DE DATOS:")

    print(datos.dtypes)

    # Revisamos los valores nulos.

    print("\nVALORES NULOS:")

    print(datos.isnull().sum())

    # --------------------------------------------------------
    # 10. DISTRIBUCIÓN DE LA VARIABLE OBJETIVO
    # --------------------------------------------------------

    print("\nDISTRIBUCIÓN DE APROBACIÓN:")

    print(

        datos["aprueba"].value_counts()

    )

    print("\nPORCENTAJES DE APROBACIÓN:")

    print(

        datos["aprueba"].value_counts(
            normalize=True
        ).mul(100).round(2)

    )

    # --------------------------------------------------------
    # 11. EXPORTACIÓN DEL DATASET
    # --------------------------------------------------------

    archivo = RUTA_RESULTADOS / "smartacademy_datos.csv"

    datos.to_csv(

        archivo,

        index=False,

        encoding="utf-8"

    )

    print("\nARCHIVO GENERADO:")

    print(archivo)

    print("\nPROCESO FINALIZADO CORRECTAMENTE.")


# ------------------------------------------------------------
# 12. EJECUCIÓN DEL PROGRAMA
# ------------------------------------------------------------

if __name__ == "__main__":

    main()