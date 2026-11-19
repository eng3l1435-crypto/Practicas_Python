
# ==============================================================
# PROYECTO: SMARTACADEMY
# ETAPA 13: PROCESAMIENTO MASIVO CON APACHE SPARK
#
# Archivo:
# p21carga_masiva_spark.py
#
# Objetivo:
# Leer y procesar 50,000 registros académicos mediante PySpark.
#
# Operaciones:
# 1. Lectura del CSV.
# 2. Definición de esquema.
# 3. Validación de datos.
# 4. Detección de duplicados.
# 5. Transformación de variables.
# 6. Estadísticas distribuidas.
# 7. Exportación a formato Parquet.
#
# IMPORTANTE:
# Este módulo no modifica MongoDB Atlas ni los módulos anteriores.
# ==============================================================


# ==============================================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# ==============================================================

from pathlib import Path

from pyspark.sql import SparkSession

from pyspark.sql.types import (
    StructType,
    StructField,
    LongType,
    StringType,
    DoubleType
)

from pyspark.sql.functions import (
    col,
    avg,
    count,
    countDistinct,
    round,
    when,
    lit
)


# ==============================================================
# 2. CONFIGURACIÓN DEL PROYECTO
# ==============================================================

RUTA_PROYECTO = Path(__file__).resolve().parents[1]

RUTA_CSV = (
    RUTA_PROYECTO
    / "resultados_bigdata"
    / "smartacademy_50000.csv"
)

RUTA_PARQUET = (
    RUTA_PROYECTO
    / "resultados_bigdata"
    / "spark_procesado"
)

RUTA_RESUMEN = (
    RUTA_PROYECTO
    / "resultados_bigdata"
    / "resumen_spark"
)

TOTAL_ESPERADO = 50000

# Cantidad de particiones de trabajo para operaciones
# que requieren redistribuir los datos.
NUM_PARTICIONES = 8


# ==============================================================
# 3. DEFINICIÓN DEL ESQUEMA
# ==============================================================

# Definimos los tipos de datos explícitamente.
#
# Esto evita que Spark tenga que inferirlos automáticamente.
# También facilita detectar errores de formato en el CSV.

ESQUEMA = StructType([

    StructField("id_registro", LongType(), False),

    StructField("id_estudiante", StringType(), False),

    StructField("materia", StringType(), False),

    StructField("parcial1", DoubleType(), False),

    StructField("parcial2", DoubleType(), False),

    StructField("parcial3", DoubleType(), False),

    StructField("asistencia_p1", DoubleType(), False),

    StructField("asistencia_p2", DoubleType(), False),

    StructField("asistencia_p3", DoubleType(), False),

    StructField("promedio_final", DoubleType(), False),

    StructField("tendencia", StringType(), False),

    StructField("resultado", StringType(), False)

])


# ==============================================================
# 4. CREAR SESIÓN DE SPARK
# ==============================================================

def crear_sesion_spark():

    """
    Crea la sesión de Apache Spark.

    local[*] indica que Spark utilizará los procesadores
    disponibles en el equipo local.

    No estamos utilizando todavía un clúster externo.
    Estamos simulando el procesamiento distribuido localmente.
    """

    spark = (
        SparkSession.builder

        .appName("SMARTACADEMY_CARGA_MASIVA")

        .master("local[*]")

        # Configuración para el procesamiento de datos.
        .config(
            "spark.sql.shuffle.partitions",
            str(NUM_PARTICIONES)
        )

        # Permite optimizaciones de ejecución.
        .config(
            "spark.sql.adaptive.enabled",
            "true"
        )

        # Evita iniciar la interfaz web de Spark.
        .config(
            "spark.ui.enabled",
            "false"
        )

        .getOrCreate()
    )

    # Reducir mensajes informativos en consola.
    spark.sparkContext.setLogLevel("WARN")

    return spark


# ==============================================================
# 5. LEER ARCHIVO CSV
# ==============================================================

def cargar_datos(spark):

    """
    Lee los 50,000 registros desde el archivo CSV.

    Se utiliza el esquema definido anteriormente.
    """

    print("\n1. LEYENDO ARCHIVO CSV")

    print("-" * 65)

    # Comprobar que el archivo existe.
    if not RUTA_CSV.exists():

        raise FileNotFoundError(
            f"No se encontró el archivo: {RUTA_CSV}"
        )

    # Lectura mediante Spark.
    df = (

        spark.read

        .format("csv")

        .option("header", "true")

        .option("encoding", "UTF-8")

        # Si un registro no cumple el esquema, se genera un error.
        .option("mode", "FAILFAST")

        .schema(ESQUEMA)

        .load(str(RUTA_CSV))
    )

    print("Archivo leído correctamente.")

    print("Esquema de datos:")

    df.printSchema()

    return df


# ==============================================================
# 6. VALIDACIÓN DE LOS DATOS
# ==============================================================

def validar_datos(df):

    """
    Comprueba la integridad de los datos.

    Validaciones:
    - Cantidad de registros.
    - Identificadores únicos.
    - Valores nulos.
    - Calificaciones válidas.
    - Asistencias válidas.
    - Cantidad de materias por estudiante.
    """

    print("\n2. VALIDANDO INFORMACIÓN")

    print("-" * 65)

    # ----------------------------------------------------------
    # 6.1 Cantidad de registros
    # ----------------------------------------------------------

    total = df.count()

    print(f"Registros encontrados: {total:,}")

    print(f"Registros esperados: {TOTAL_ESPERADO:,}")

    if total != TOTAL_ESPERADO:

        raise ValueError(
            "La cantidad de registros no coincide."
        )

    # ----------------------------------------------------------
    # 6.2 Identificadores únicos
    # ----------------------------------------------------------

    identificadores = (
        df.select("id_registro")
        .distinct()
        .count()
    )

    duplicados = total - identificadores

    print(f"Registros duplicados: {duplicados}")

    if duplicados > 0:

        raise ValueError(
            "Se encontraron identificadores duplicados."
        )

    # ----------------------------------------------------------
    # 6.3 Valores nulos
    # ----------------------------------------------------------

    columnas_obligatorias = df.columns

    condiciones_nulos = [
        col(nombre).isNull()
        for nombre in columnas_obligatorias
    ]

    # Combinar las condiciones con OR.
    condicion_nulos = condiciones_nulos[0]

    for condicion in condiciones_nulos[1:]:

        condicion_nulos = condicion_nulos | condicion

    registros_nulos = (
        df.filter(condicion_nulos)
        .count()
    )

    print(f"Registros con valores nulos: {registros_nulos}")

    if registros_nulos > 0:

        raise ValueError(
            "Se encontraron registros con valores nulos."
        )

    # ----------------------------------------------------------
    # 6.4 Validar calificaciones
    # ----------------------------------------------------------

    columnas_calificaciones = [
        "parcial1",
        "parcial2",
        "parcial3",
        "promedio_final"
    ]

    condicion_calificaciones = None

    for nombre in columnas_calificaciones:

        condicion = (
            (col(nombre) < 0)
            | (col(nombre) > 10)
        )

        if condicion_calificaciones is None:

            condicion_calificaciones = condicion

        else:

            condicion_calificaciones = (
                condicion_calificaciones | condicion
            )

    calificaciones_invalidas = (
        df.filter(condicion_calificaciones)
        .count()
    )

    print(
        "Registros con calificaciones inválidas: "
        f"{calificaciones_invalidas}"
    )

    if calificaciones_invalidas > 0:

        raise ValueError(
            "Existen calificaciones fuera del rango 0-10."
        )

    # ----------------------------------------------------------
    # 6.5 Validar asistencias
    # ----------------------------------------------------------

    columnas_asistencia = [
        "asistencia_p1",
        "asistencia_p2",
        "asistencia_p3"
    ]

    condicion_asistencia = None

    for nombre in columnas_asistencia:

        condicion = (
            (col(nombre) < 0)
            | (col(nombre) > 100)
        )

        if condicion_asistencia is None:

            condicion_asistencia = condicion

        else:

            condicion_asistencia = (
                condicion_asistencia | condicion
            )

    asistencias_invalidas = (
        df.filter(condicion_asistencia)
        .count()
    )

    print(
        "Registros con asistencia inválida: "
        f"{asistencias_invalidas}"
    )

    if asistencias_invalidas > 0:

        raise ValueError(
            "Existen asistencias fuera del rango 0-100."
        )

    # ----------------------------------------------------------
    # 6.6 Materias por estudiante
    # ----------------------------------------------------------

    materias_por_estudiante = (

        df.groupBy("id_estudiante")

        .agg(
            countDistinct("materia").alias(
                "total_materias"
            )
        )
    )

    estudiantes_incorrectos = (

        materias_por_estudiante

        .filter(
            col("total_materias") != 5
        )

        .count()
    )

    print(
        "Estudiantes que no tienen cinco materias: "
        f"{estudiantes_incorrectos}"
    )

    if estudiantes_incorrectos > 0:

        raise ValueError(
            "Existen estudiantes con una cantidad "
            "incorrecta de materias."
        )

    print("\nVALIDACIÓN COMPLETADA CORRECTAMENTE.")

    return total


# ==============================================================
# 7. TRANSFORMACIÓN DE DATOS
# ==============================================================

def transformar_datos(df):

    """
    Genera nuevas variables académicas.

    Las nuevas columnas serán:
    - asistencia_promedio
    - variacion_calificacion
    - tendencia_recalculada

    Se conserva la columna tendencia original.
    """

    print("\n3. TRANSFORMANDO DATOS")

    print("-" * 65)

    # ----------------------------------------------------------
    # 7.1 Calcular promedio de asistencia
    # ----------------------------------------------------------

    df = df.withColumn(

        "asistencia_promedio",

        round(

            (
                col("asistencia_p1")
                + col("asistencia_p2")
                + col("asistencia_p3")
            ) / 3,

            2
        )
    )

    # ----------------------------------------------------------
    # 7.2 Calcular variación de calificación
    # ----------------------------------------------------------

    df = df.withColumn(

        "variacion_calificacion",

        round(
            col("parcial3") - col("parcial1"),
            2
        )
    )

    # ----------------------------------------------------------
    # 7.3 Recalcular tendencia
    # ----------------------------------------------------------

    # Esta variable sirve para comprobar la etiqueta original.
    # No reemplaza la etiqueta de entrenamiento.

    df = df.withColumn(

        "tendencia_recalculada",

        when(
            col("variacion_calificacion") >= 1,
            lit("Favorable")
        )

        .when(
            col("variacion_calificacion") <= -1,
            lit("Descendente")
        )

        .otherwise(
            lit("Estable")
        )
    )

    print("Nuevas variables calculadas.")

    return df


# ==============================================================
# 8. ANÁLISIS ESTADÍSTICO DISTRIBUIDO
# ==============================================================

def analizar_datos(df):

    """
    Calcula indicadores utilizando operaciones distribuidas.
    """

    print("\n4. ANÁLISIS ESTADÍSTICO CON SPARK")

    print("-" * 65)

    # ----------------------------------------------------------
    # 8.1 Promedio general de calificaciones
    # ----------------------------------------------------------

    print("\nPROMEDIO GENERAL")

    df.select(

        round(
            avg("promedio_final"),
            2
        ).alias("promedio_general")

    ).show()

    # ----------------------------------------------------------
    # 8.2 Promedio por materia
    # ----------------------------------------------------------

    print("\nPROMEDIO POR MATERIA")

    promedio_materias = (

        df.groupBy("materia")

        .agg(

            round(
                avg("promedio_final"),
                2
            ).alias("promedio_materia"),

            count("*").alias("total_registros")

        )

        .orderBy("materia")
    )

    promedio_materias.show(truncate=False)

    # ----------------------------------------------------------
    # 8.3 Distribución de tendencias
    # ----------------------------------------------------------

    print("\nDISTRIBUCIÓN DE TENDENCIAS")

    tendencias = (

        df.groupBy("tendencia")

        .agg(
            count("*").alias("cantidad")
        )

        .orderBy("tendencia")
    )

    tendencias.show()

    # ----------------------------------------------------------
    # 8.4 Distribución de resultados
    # ----------------------------------------------------------

    print("\nRESULTADOS ACADÉMICOS")

    resultados = (

        df.groupBy("resultado")

        .agg(
            count("*").alias("cantidad")
        )

    )

    resultados.show()

    # ----------------------------------------------------------
    # 8.5 Comparación de tendencia original y recalculada
    # ----------------------------------------------------------

    print("\nCOMPROBACIÓN DE ETIQUETAS")

    diferencias = (

        df.filter(

            col("tendencia") != col("tendencia_recalculada")

        ).count()

    )

    print(
        "Diferencias entre tendencia original y recalculada: "
        f"{diferencias}"
    )

    # La etiqueta original se conserva porque se generó
    # utilizando los valores previos al redondeo.
    # Una diferencia pequeña puede aparecer en casos cercanos
    # a los límites definidos para la clasificación.


# ==============================================================
# 9. GUARDAR RESULTADOS EN PARQUET
# ==============================================================

def guardar_resultados(df):

    """
    Guarda los registros procesados en formato Parquet.

    Se particiona por materia para facilitar consultas
    posteriores por esta variable.
    """

    print("\n5. GUARDANDO RESULTADOS")

    print("-" * 65)

    (

        df.repartition(
            NUM_PARTICIONES,
            "materia"
        )

        .write

        .mode("overwrite")

        .partitionBy("materia")

        .parquet(str(RUTA_PARQUET))

    )

    print(
        f"Datos Parquet guardados en: {RUTA_PARQUET}"
    )

    # ----------------------------------------------------------
    # 9.1 Guardar resumen estadístico
    # ----------------------------------------------------------

    resumen = (

        df.groupBy("materia")

        .agg(

            count("*").alias("registros"),

            round(
                avg("promedio_final"),
                2
            ).alias("promedio"),

            round(
                avg("asistencia_promedio"),
                2
            ).alias("asistencia_promedio")

        )

    )

    (

        resumen

        .coalesce(1)

        .write

        .mode("overwrite")

        .option("header", "true")

        .csv(str(RUTA_RESUMEN))

    )

    print(
        f"Resumen guardado en: {RUTA_RESUMEN}"
    )


# ==============================================================
# 10. PROGRAMA PRINCIPAL
# ==============================================================

def main():

    print("\n" + "=" * 65)

    print("SMARTACADEMY")

    print("ETAPA 13 - PROCESAMIENTO MASIVO CON APACHE SPARK")

    print("=" * 65)

    spark = None

    try:

        # Crear sesión.
        spark = crear_sesion_spark()

        print("\nSESIÓN SPARK INICIADA")

        print(
            "Versión de Spark:",
            spark.version
        )

        # Leer información.
        df = cargar_datos(spark)

        # Mostrar cantidad de particiones iniciales.
        print(
            "\nParticiones iniciales:",
            df.rdd.getNumPartitions()
        )

        # Validar registros.
        validar_datos(df)

        # Transformar información.
        df = transformar_datos(df)

        # Distribuir datos para el análisis.
        df = df.repartition(NUM_PARTICIONES)

        print(
            "\nParticiones de trabajo:",
            df.rdd.getNumPartitions()
        )

        # Análisis estadístico.
        analizar_datos(df)

        # Guardar Parquet y resumen.
        guardar_resultados(df)

        print("\n" + "=" * 65)

        print("ETAPA 13 FINALIZADA CORRECTAMENTE")

        print("=" * 65)

    except Exception as error:

        print("\nERROR DURANTE EL PROCESAMIENTO")

        print("-" * 65)

        print(type(error).__name__, ":", error)

        raise

    finally:

        # Cerrar Spark para liberar recursos.
        if spark is not None:

            spark.stop()

            print("\nSESIÓN SPARK CERRADA")


# ==============================================================
# 11. PUNTO DE ENTRADA
# ==============================================================

if __name__ == "__main__":

    main()
