
# ==============================================================
# PROYECTO: SMARTACADEMY
# ETAPA 14: PERSISTENCIA MASIVA CON SPARK Y MONGODB ATLAS
#
# Archivo:
# p22persistencia_spark_atlas.py
#
# OBJETIVO:
# Almacenar en MongoDB Atlas los 50,000 registros académicos
# previamente procesados con Apache Spark.
#
# ORIGEN:
# resultados_bigdata/spark_procesado
#
# DESTINO:
# Base de datos: smartacademy_db (configurada en .env)
# Colección: registros_academicos_masivos
#
# CARACTERÍSTICAS:
# - Lectura de archivos Parquet.
# - Conexión segura con MongoDB Atlas.
# - Procesamiento por particiones Spark.
# - Inserción masiva por lotes.
# - Operaciones upsert para evitar duplicados.
# - Índices de búsqueda.
# - Validación final de los documentos.
#
# IMPORTANTE:
# Se utiliza una colección independiente para no mezclar
# datos sintéticos con expedientes académicos reales.
# ==============================================================


# ==============================================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# ==============================================================

from pathlib import Path

from datetime import datetime, timezone

from urllib.parse import quote_plus

import os

import certifi

from dotenv import load_dotenv

from pymongo import (
    MongoClient,
    ReplaceOne
)

from pyspark.sql import SparkSession


# ==============================================================
# 2. CONFIGURACIÓN GENERAL
# ==============================================================

# Obtener la raíz del proyecto.
RUTA_PROYECTO = Path(__file__).resolve().parents[1]

# Cargar variables de entorno.
load_dotenv(RUTA_PROYECTO / ".env")

# Directorio con los datos procesados por Spark.
RUTA_PARQUET = (
    RUTA_PROYECTO
    / "resultados_bigdata"
    / "spark_procesado"
)

# Nombre de la colección exclusiva para la carga masiva.
NOMBRE_COLECCION = "registros_academicos_masivos"

# Identificador fijo de este lote.
#
# Si ejecutamos nuevamente el programa, los registros del mismo
# lote se actualizarán en lugar de duplicarse.
LOTE_ID = "SMARTACADEMY-50000-V1"

# Cantidad de operaciones por lote de escritura.
TAMANO_LOTE = 500

# Número de particiones de Spark.
NUM_PARTICIONES = 8

# Cantidad esperada de registros.
TOTAL_ESPERADO = 50000


# ==============================================================
# 3. CONSTRUIR URI DE MONGODB
# ==============================================================

def construir_uri():

    """
    Construye la dirección de conexión de MongoDB Atlas.

    Las credenciales se obtienen exclusivamente del archivo .env.

    quote_plus permite utilizar caracteres especiales
    en el usuario y la contraseña.
    """

    usuario = os.getenv("MONGO_USER")

    password = os.getenv("MONGO_PASSWORD")

    cluster = os.getenv("MONGO_CLUSTER")

    if not usuario or not password or not cluster:

        raise ValueError(
            "Faltan variables de conexión en el archivo .env."
        )

    # Codificar usuario y contraseña.
    usuario = quote_plus(usuario)

    password = quote_plus(password)

    # Construir dirección de MongoDB Atlas.
    uri = (
        f"mongodb+srv://{usuario}:{password}"
        f"@{cluster}/"
        "?retryWrites=true&w=majority"
    )

    return uri


# ==============================================================
# 4. OBTENER NOMBRE DE LA BASE DE DATOS
# ==============================================================

def obtener_base_datos():

    """
    Recupera el nombre de la base de datos desde .env.
    """

    nombre = os.getenv(
        "MONGO_DB",
        "smartacademy_db"
    )

    if not nombre:

        raise ValueError(
            "No se configuró MONGO_DB en el archivo .env."
        )

    return nombre


# ==============================================================
# 5. VALIDAR CONEXIÓN CON MONGODB ATLAS
# ==============================================================

def conectar_mongodb():

    """
    Abre una conexión con MongoDB Atlas y verifica
    que el servidor responda.
    """

    print("\n1. CONECTANDO CON MONGODB ATLAS")

    print("-" * 65)

    uri = construir_uri()

    nombre_db = obtener_base_datos()

    # Crear cliente.
    cliente = MongoClient(

        uri,

        # Certificados de confianza.
        tlsCAFile=certifi.where(),

        # Tiempo máximo de espera.
        serverSelectionTimeoutMS=20000,

        # Reintentar escrituras cuando corresponda.
        retryWrites=True

    )

    # Comprobar conexión.
    cliente.admin.command("ping")

    print("Conexión establecida correctamente.")

    print(f"Base de datos: {nombre_db}")

    print(f"Colección: {NOMBRE_COLECCION}")

    # Crear o recuperar colección.
    db = cliente[nombre_db]

    coleccion = db[NOMBRE_COLECCION]

    # ----------------------------------------------------------
    # Crear índices
    # ----------------------------------------------------------

    # Cada combinación lote + registro debe ser única.
    coleccion.create_index(

        [
            ("id_lote", 1),
            ("id_registro", 1)
        ],

        unique=True,

        name="indice_lote_registro_unico"

    )

    # Índice para consultar estudiantes.
    coleccion.create_index(

        "id_estudiante",

        name="indice_estudiante"

    )

    # Índice para consultar materias.
    coleccion.create_index(

        "materia",

        name="indice_materia"

    )

    print("Índices verificados correctamente.")

    return cliente, db, coleccion, uri


# ==============================================================
# 6. CREAR SESIÓN SPARK
# ==============================================================

def crear_sesion_spark():

    """
    Crea la sesión local de Apache Spark.
    """

    spark = (

        SparkSession.builder

        .appName("SMARTACADEMY_SPARK_MONGODB")

        .master("local[*]")

        .config(
            "spark.sql.shuffle.partitions",
            str(NUM_PARTICIONES)
        )

        .config(
            "spark.ui.enabled",
            "false"
        )

        .getOrCreate()

    )

    spark.sparkContext.setLogLevel("WARN")

    return spark


# ==============================================================
# 7. FUNCIÓN DE INSERCIÓN POR PARTICIÓN
# ==============================================================

def insertar_particion(
    filas,
    uri,
    nombre_db,
    nombre_coleccion,
    lote_id
):

    """
    Recibe una partición de Spark y escribe sus registros
    en MongoDB mediante operaciones bulk_write.

    IMPORTANTE:
    La conexión se crea dentro de cada partición.
    No se comparte un MongoClient entre procesos Spark.

    El uso de ReplaceOne + upsert=True hace que la operación
    sea repetible para los mismos identificadores.
    """

    # Crear conexión independiente para esta partición.
    cliente = MongoClient(

        uri,

        tlsCAFile=certifi.where(),

        serverSelectionTimeoutMS=20000,

        retryWrites=True

    )

    try:

        db = cliente[nombre_db]

        coleccion = db[nombre_coleccion]

        operaciones = []

        # Fecha de carga para esta partición.
        fecha_carga = datetime.now(timezone.utc)

        # Recorrer las filas recibidas.
        for fila in filas:

            # Convertir Row de Spark a diccionario.
            documento = fila.asDict(recursive=True)

            # Agregar metadatos.
            documento["id_lote"] = lote_id

            documento["sistema"] = "SmartAcademy"

            documento["etapa"] = 14

            documento["fecha_carga"] = fecha_carga

            # Crear operación de reemplazo o inserción.
            operacion = ReplaceOne(

                {

                    "id_lote": lote_id,

                    "id_registro": documento["id_registro"]

                },

                documento,

                upsert=True

            )

            operaciones.append(operacion)

            # Escribir cuando se alcance el tamaño del lote.
            if len(operaciones) >= TAMANO_LOTE:

                coleccion.bulk_write(

                    operaciones,

                    ordered=False

                )

                operaciones.clear()

        # Escribir registros restantes.
        if operaciones:

            coleccion.bulk_write(

                operaciones,

                ordered=False

            )

    finally:

        # Cerrar conexión de la partición.
        cliente.close()


# ==============================================================
# 8. LEER PARQUET CON SPARK
# ==============================================================

def cargar_parquet(spark):

    """
    Recupera los archivos Parquet generados en la etapa 13.
    """

    print("\n2. LEYENDO ARCHIVOS PARQUET")

    print("-" * 65)

    if not RUTA_PARQUET.exists():

        raise FileNotFoundError(
            f"No existe el directorio: {RUTA_PARQUET}"
        )

    # Leer archivos Parquet.
    df = spark.read.parquet(str(RUTA_PARQUET))

    total = df.count()

    print(f"Registros leídos: {total:,}")

    if total != TOTAL_ESPERADO:

        raise ValueError(
            f"Se esperaban {TOTAL_ESPERADO} registros "
            f"y se encontraron {total}."
        )

    print("Archivos Parquet validados.")

    return df


# ==============================================================
# 9. EJECUTAR CARGA MASIVA
# ==============================================================

def ejecutar_carga(df, uri, nombre_db):

    """
    Distribuye los registros en particiones y ejecuta
    la escritura masiva en MongoDB.
    """

    print("\n3. INICIANDO CARGA MASIVA")

    print("-" * 65)

    # Redistribuir los datos en ocho particiones.
    df_distribuido = df.repartition(NUM_PARTICIONES)

    print(
        "Particiones de escritura:",
        df_distribuido.rdd.getNumPartitions()
    )

    print(
        "Tamaño del lote de escritura:",
        TAMANO_LOTE
    )

    print(
        "Identificador de lote:",
        LOTE_ID
    )

    # Ejecutar una operación de escritura por partición.
    #
    # No utilizamos collect().
    # Los registros se procesan en las particiones de Spark.

    df_distribuido.foreachPartition(

        lambda filas: insertar_particion(

            filas,

            uri,

            nombre_db,

            NOMBRE_COLECCION,

            LOTE_ID

        )

    )

    print("Proceso de escritura terminado.")


# ==============================================================
# 10. VALIDAR DOCUMENTOS EN MONGODB
# ==============================================================

def validar_persistencia(coleccion):

    """
    Consulta MongoDB para comprobar que se almacenaron
    correctamente los registros del lote.
    """

    print("\n4. VALIDANDO PERSISTENCIA")

    print("-" * 65)

    filtro = {

        "id_lote": LOTE_ID

    }

    # Contar documentos del lote.
    total = coleccion.count_documents(filtro)

    print(
        f"Documentos encontrados en Atlas: {total:,}"
    )

    print(
        f"Documentos esperados: {TOTAL_ESPERADO:,}"
    )

    # Comprobar resultado.
    if total != TOTAL_ESPERADO:

        raise ValueError(
            "La cantidad de documentos en Atlas "
            "no coincide con la cantidad esperada."
        )

    # Comprobar que existan cinco materias.
    materias = coleccion.distinct(

        "materia",

        filtro

    )

    print(
        f"Materias almacenadas: {len(materias)}"
    )

    # Comprobar estudiantes diferentes.
    estudiantes = coleccion.distinct(

        "id_estudiante",

        filtro

    )

    print(
        f"Estudiantes almacenados: {len(estudiantes):,}"
    )

    if len(estudiantes) != 10000:

        raise ValueError(
            "La cantidad de estudiantes no coincide."
        )

    # Mostrar documento de ejemplo.
    ejemplo = coleccion.find_one(

        filtro,

        {

            "_id": 0,

            "id_registro": 1,

            "id_estudiante": 1,

            "materia": 1,

            "promedio_final": 1,

            "tendencia": 1,

            "resultado": 1,

            "id_lote": 1

        }

    )

    print("\nDOCUMENTO DE EJEMPLO")

    print("-" * 65)

    print(ejemplo)

    print("\nVALIDACIÓN COMPLETADA.")

    return total


# ==============================================================
# 11. PROGRAMA PRINCIPAL
# ==============================================================

def main():

    print("\n" + "=" * 65)

    print("SMARTACADEMY")

    print("ETAPA 14 - PERSISTENCIA MASIVA EN MONGODB ATLAS")

    print("=" * 65)

    spark = None

    cliente = None

    try:

        # ------------------------------------------------------
        # 11.1 Conectar con MongoDB
        # ------------------------------------------------------

        cliente, db, coleccion, uri = conectar_mongodb()

        nombre_db = db.name

        # ------------------------------------------------------
        # 11.2 Crear sesión Spark
        # ------------------------------------------------------

        spark = crear_sesion_spark()

        print("\nSESIÓN SPARK INICIADA")

        print("Versión:", spark.version)

        # ------------------------------------------------------
        # 11.3 Leer datos Parquet
        # ------------------------------------------------------

        df = cargar_parquet(spark)

        # ------------------------------------------------------
        # 11.4 Cargar información a Atlas
        # ------------------------------------------------------

        ejecutar_carga(

            df,

            uri,

            nombre_db

        )

        # ------------------------------------------------------
        # 11.5 Validar almacenamiento
        # ------------------------------------------------------

        total = validar_persistencia(coleccion)

        # ------------------------------------------------------
        # 11.6 Mostrar resultados
        # ------------------------------------------------------

        print("\n" + "=" * 65)

        print("ETAPA 14 FINALIZADA CORRECTAMENTE")

        print("=" * 65)

        print(
            f"Registros almacenados: {total:,}"
        )

        print(
            f"Base de datos: {nombre_db}"
        )

        print(
            f"Colección: {NOMBRE_COLECCION}"
        )

    except Exception as error:

        print("\nERROR DURANTE LA PERSISTENCIA")

        print("-" * 65)

        print(
            type(error).__name__,
            ":",
            error
        )

        raise

    finally:

        # Cerrar sesión Spark.
        if spark is not None:

            spark.stop()

            print("\nSESIÓN SPARK CERRADA")

        # Cerrar conexión principal.
        if cliente is not None:

            cliente.close()

            print("CONEXIÓN MONGODB CERRADA")


# ==============================================================
# 12. PUNTO DE ENTRADA
# ==============================================================

if __name__ == "__main__":

    main()
