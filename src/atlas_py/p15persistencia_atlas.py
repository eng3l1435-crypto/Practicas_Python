# ============================================================
# PROYECTO: SMARTACADEMY
# ETAPA 6: PERSISTENCIA INTELIGENTE CON MONGODB ATLAS
#
# Archivo:
# p15persistencia_mongodb.py
#
# Objetivo:
# Recuperar el reporte generado en la Etapa 5 y almacenarlo
# en una colección de MongoDB Atlas.
#
# Tecnologías:
# - Python
# - Pandas
# - PyMongo
# - Python-dotenv
# - MongoDB Atlas
# ============================================================


# ------------------------------------------------------------
# 1. IMPORTAR LAS LIBRERÍAS
# ------------------------------------------------------------

import os
import json
import uuid
import certifi

import pandas as pd

from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import quote_plus

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError


# ------------------------------------------------------------
# 2. DEFINIR LAS RUTAS DEL PROYECTO
# ------------------------------------------------------------

# Obtener la ubicación de este archivo.
RUTA_ARCHIVO = Path(__file__).resolve()

# Obtener la carpeta principal del proyecto.
RUTA_PROYECTO = RUTA_ARCHIVO.parents[1]

# Definir la ubicación del reporte generado en la Etapa 5.
RUTA_REPORTE = (
    RUTA_PROYECTO
    / "unidad3_fia"
    / "resultados_p10"
    / "reporte_estudiante.csv"
)

# Definir la ubicación del archivo .env.
RUTA_ENV = RUTA_PROYECTO / ".env"


# ------------------------------------------------------------
# 3. CARGAR LAS VARIABLES DE ENTORNO
# ------------------------------------------------------------

# Leer las variables de configuración desde .env.
load_dotenv(RUTA_ENV)


# Recuperar las credenciales y configuración.
MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")
MONGO_CLUSTER = os.getenv("MONGO_CLUSTER")

MONGO_DB = os.getenv("MONGO_DB")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION")


# ------------------------------------------------------------
# 4. VALIDAR LA CONFIGURACIÓN
# ------------------------------------------------------------

def validar_configuracion():

    """
    Comprueba que las variables necesarias existan.

    No muestra las credenciales en pantalla.
    """

    variables = {
        "MONGO_USER": MONGO_USER,
        "MONGO_PASSWORD": MONGO_PASSWORD,
        "MONGO_CLUSTER": MONGO_CLUSTER,
        "MONGO_DB": MONGO_DB,
        "MONGO_COLLECTION": MONGO_COLLECTION
    }

    faltantes = [
        nombre
        for nombre, valor in variables.items()
        if not valor
    ]

    if faltantes:

        raise ValueError(
            "Faltan variables en el archivo .env: "
            + ", ".join(faltantes)
        )


# ------------------------------------------------------------
# 5. CONSTRUIR LA CADENA DE CONEXIÓN
# ------------------------------------------------------------

def construir_uri():

    """
    Construye la URI de conexión para MongoDB Atlas.

    quote_plus permite codificar caracteres especiales
    que puedan existir en el usuario o contraseña.
    """

    usuario = quote_plus(MONGO_USER)

    contrasena = quote_plus(MONGO_PASSWORD)

    uri = (
        f"mongodb+srv://{usuario}:{contrasena}"
        f"@{MONGO_CLUSTER}/"
        "?retryWrites=true&w=majority"
        "&appName=SmartAcademy"
    )

    return uri


# ------------------------------------------------------------
# 6. ESTABLECER CONEXIÓN CON MONGODB ATLAS
# ------------------------------------------------------------

def conectar_mongodb():

    """
    Crea el cliente de MongoDB y verifica la conexión.

    serverSelectionTimeoutMS limita el tiempo de espera
    si Atlas no está disponible.
    """

    validar_configuracion()

    uri = construir_uri()

    cliente = MongoClient(
        uri,
        serverSelectionTimeoutMS=10000
    )

    # Solicitar información al servidor.
    # Esto permite comprobar que la conexión funciona.
    cliente.admin.command("ping")

    print("Conexión con MongoDB Atlas establecida.")

    # Seleccionar la base de datos.
    base_datos = cliente[MONGO_DB]

    # Seleccionar la colección.
    coleccion = base_datos[MONGO_COLLECTION]

    return cliente, coleccion


# ------------------------------------------------------------
# 7. LEER EL REPORTE GENERADO EN LA ETAPA 5
# ------------------------------------------------------------

def leer_reporte():

    """
    Recupera el archivo CSV generado por el motor
    de recomendaciones de SmartAcademy.
    """

    print("\nLEYENDO REPORTE DE SMARTACADEMY...")

    # Comprobar que el archivo exista.
    if not RUTA_REPORTE.exists():

        raise FileNotFoundError(
            f"No se encontró el reporte: {RUTA_REPORTE}"
        )

    # Leer el archivo CSV.
    dataframe = pd.read_csv(RUTA_REPORTE)

    # Comprobar que tenga información.
    if dataframe.empty:

        raise ValueError(
            "El reporte CSV no contiene registros."
        )

    print("Reporte leído correctamente.")

    print(
        "Registros encontrados:",
        len(dataframe)
    )

    return dataframe


# ------------------------------------------------------------
# 8. CONVERTIR EL REPORTE EN DOCUMENTOS JSON
# ------------------------------------------------------------

def convertir_documentos(dataframe):

    """
    Convierte cada fila del DataFrame en un documento
    compatible con MongoDB.

    Los documentos conservan los campos del reporte CSV.
    """

    # Convertir el DataFrame a una estructura JSON.
    # Esto ayuda a normalizar los valores de Pandas.
    registros = json.loads(
        dataframe.to_json(
            orient="records",
            force_ascii=False
        )
    )

    documentos = []

    for registro in registros:

        documento = {

            # Identificador único de este análisis.
            "id_analisis": str(uuid.uuid4()),

            # Nombre del sistema.
            "sistema": "SmartAcademy",

            # Etapa que generó el análisis.
            "etapa": 5,

            # Fecha y hora de almacenamiento.
            "fecha_registro": datetime.now(timezone.utc),

            # Archivo que originó el registro.
            "archivo_origen": RUTA_REPORTE.name,

            # Información original del reporte.
            "datos_reporte": registro

        }

        documentos.append(documento)

    return documentos


# ------------------------------------------------------------
# 9. CREAR ÍNDICES EN MONGODB
# ------------------------------------------------------------

def crear_indices(coleccion):

    """
    Crea índices para facilitar consultas posteriores.

    Un índice permite localizar registros sin tener
    que recorrer innecesariamente toda la colección.
    """

    # Evitar que se repita el identificador del análisis.
    coleccion.create_index(
        "id_analisis",
        unique=True
    )

    # Facilitar consultas por fecha.
    coleccion.create_index(
        "fecha_registro"
    )

    # Facilitar consultas por sistema.
    coleccion.create_index(
        "sistema"
    )

    print("Índices de MongoDB verificados.")


# ------------------------------------------------------------
# 10. GUARDAR LOS DOCUMENTOS EN MONGODB ATLAS
# ------------------------------------------------------------

def guardar_documentos(coleccion, documentos):

    """
    Inserta los documentos en la colección.

    insert_many permite almacenar varios documentos
    en una sola operación.
    """

    print("\nGUARDANDO INFORMACIÓN EN MONGODB ATLAS...")

    resultado = coleccion.insert_many(documentos)

    print("Documentos almacenados:", len(resultado.inserted_ids))

    return resultado.inserted_ids


# ------------------------------------------------------------
# 11. CONSULTAR EL HISTORIAL DE ANÁLISIS
# ------------------------------------------------------------

def consultar_historial(coleccion):

    """
    Recupera los últimos análisis almacenados.

    Se ordenan por fecha, del más reciente al más antiguo.
    """

    print("\nHISTORIAL DE SMARTACADEMY")

    print("-" * 55)

    registros = coleccion.find(
        {},
        {
            "_id": 0,
            "id_analisis": 1,
            "fecha_registro": 1,
            "sistema": 1,
            "datos_reporte": 1
        }
    ).sort(
        "fecha_registro",
        -1
    ).limit(5)

    for numero, registro in enumerate(registros, start=1):

        print(f"\nANÁLISIS {numero}")

        print(
            "ID:",
            registro.get("id_analisis")
        )

        print(
            "Fecha:",
            registro.get("fecha_registro")
        )

        print(
            "Datos:",
            registro.get("datos_reporte")
        )


# ------------------------------------------------------------
# 12. FUNCIÓN PRINCIPAL
# ------------------------------------------------------------

def main():

    """
    Coordina todo el proceso de persistencia.
    """

    print("=" * 60)

    print("SMARTACADEMY")

    print("ETAPA 6 - PERSISTENCIA CON MONGODB ATLAS")

    print("=" * 60)

    cliente = None

    try:

        # Paso 1: leer el reporte.
        dataframe = leer_reporte()

        # Paso 2: transformar los datos.
        documentos = convertir_documentos(dataframe)

        # Paso 3: conectar con MongoDB Atlas.
        cliente, coleccion = conectar_mongodb()

        # Paso 4: crear índices.
        crear_indices(coleccion)

        # Paso 5: almacenar documentos.
        guardar_documentos(
            coleccion,
            documentos
        )

        # Paso 6: consultar los últimos registros.
        consultar_historial(coleccion)

        print("\n" + "=" * 60)

        print("PROCESO DE PERSISTENCIA FINALIZADO")

        print("=" * 60)

        print(
            "Base de datos:",
            MONGO_DB
        )

        print(
            "Colección:",
            MONGO_COLLECTION
        )

    except FileNotFoundError as error:

        print("\nERROR DE ARCHIVO:")
        print(error)

    except ValueError as error:

        print("\nERROR DE CONFIGURACIÓN:")
        print(error)

    except PyMongoError as error:

        print("\nERROR DE MONGODB:")
        print(error)

    except Exception as error:

        print("\nERROR INESPERADO:")
        print(error)

    finally:

        # Cerrar la conexión para liberar recursos.
        if cliente is not None:

            cliente.close()

            print("\nConexión con MongoDB cerrada.")


# ------------------------------------------------------------
# 13. PUNTO DE ENTRADA
# ------------------------------------------------------------

if __name__ == "__main__":

    main()