# ============================================================
# PROYECTO: SMARTACADEMY
# ETAPA 7: CONSULTAS INTELIGENTES Y ANALISIS ESTADISTICO

# Archivo:
# p16analisis_estadistico.py

# Objetivo:
# Desarrollar un módulo en Python que consulte los análisis almacenados en MongoDB Atlas y 
#genere estadísticas que permitan identificar patrones académicos, perfiles de estudiantes y tendencias de riesgo.

#Al finalizar, se podrán:
#- Consultar documentos almacenados en MongoDB.
#- Aplicar filtros con find().
#- Utilizar el framework aggregate().
#- Agrupar estudiantes mediante $group.
#- Ordenar resultados con $sort.
#- Calcular promedios y cantidades.
#- Generar reportes estadísticos con Pandas.

#El programa tendrá seis operaciones principales:
#   Operación	Técnica MongoDB	        Propósito
#       1	        find()	            Consultar todos los análisis
#       2	        $match	            Filtrar estudiantes con riesgo
#       3	        $group	            Agrupar por perfil
#       4	        $group	            Calcular promedios
#       5	        $sort	            Ordenar por probabilidad de aprobación
#       6	        aggregate()	        Generar estadísticas generales


# Tecnologías:
# - Python
# - Pandas
# - PyMongo
# - MongoDB Atlas
# - Framework de agregación

#Se tendrán cuatro reportes:

#Archivo	                    Contenido
#analisis_historico.csv	        Historial de los análisis
#analisis_riesgos.csv	        Registros con predicción de riesgo
#estadisticas_perfiles.csv	    Promedios por perfil K-Means
#estadisticas_generales.csv	    Estadísticas globales

#El framework de agregación, una herramienta fundamental para trabajar con bases de datos NoSQL.

#   Operador	        Función	            Ejemplo utilizado
#   $match	        Filtrar documentos	    Buscar predicciones de riesgo
#   $group	        Agrupar documentos	    Agrupar por perfil
#   $sum	        Sumar valores	        Contar análisis
#   $avg	        Calcular promedios	    Promedio de asistencia
#   $cond	        Evaluar condiciones	    Contabilizar predicciones
#   $project	    Seleccionar campos	    Construir reportes
#   $sort	        Ordenar documentos	    Ordenar por probabilidad
#   $limit	        Limitar resultados	    Obtener cinco registros

# ============================================================


# ------------------------------------------------------------
# 1. IMPORTAR LIBRERIAS
# ------------------------------------------------------------

import os

import pandas as pd

from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError


# ------------------------------------------------------------
# 2. DEFINIR RUTAS DEL PROYECTO
# ------------------------------------------------------------

# Obtener la ubicación del archivo actual.
RUTA_ARCHIVO = Path(__file__).resolve()

# Obtener la carpeta principal del proyecto.
RUTA_PROYECTO = RUTA_ARCHIVO.parents[1]

# Definir la ubicación donde se guardarán los reportes.
RUTA_RESULTADOS = (
    RUTA_PROYECTO
    / "unidad3_fia"
    / "resultados_p10"
)

# Crear la carpeta si todavía no existe.
RUTA_RESULTADOS.mkdir(
    parents=True,
    exist_ok=True
)

# Cargar las variables de entorno.
load_dotenv(RUTA_PROYECTO / ".env")


# ------------------------------------------------------------
# 3. RECUPERAR VARIABLES DE CONFIGURACION
# ------------------------------------------------------------

MONGO_USER = os.getenv("MONGO_USER")

MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")

MONGO_CLUSTER = os.getenv("MONGO_CLUSTER")

MONGO_DB = os.getenv("MONGO_DB")

MONGO_COLLECTION = os.getenv("MONGO_COLLECTION")


# ------------------------------------------------------------
# 4. VALIDAR VARIABLES DE ENTORNO
# ------------------------------------------------------------

def validar_configuracion():
    
    """
    Comprueba que existan las variables necesarias
    para establecer la conexión con MongoDB Atlas.
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
            "Variables faltantes en .env: "
            + ", ".join(faltantes)
        )


# ------------------------------------------------------------
# 5. CONSTRUIR URI DE MONGODB
# ------------------------------------------------------------

def construir_uri():

    """
    Construye la cadena de conexión.

    quote_plus codifica caracteres especiales que
    pudieran existir en usuario y contraseña.
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
# 6. CONECTAR CON MONGODB ATLAS
# ------------------------------------------------------------

def conectar_mongodb():

    """
    Establece conexión con MongoDB Atlas y selecciona
    la base de datos y colección de SmartAcademy.
    """

    validar_configuracion()

    cliente = MongoClient(
        construir_uri(),
        serverSelectionTimeoutMS=15000
    )

    # Comprobar que el servidor responda.
    cliente.admin.command("ping")

    print("Conexión con MongoDB Atlas establecida.")

    # Seleccionar la base de datos.
    base_datos = cliente[MONGO_DB]

    # Seleccionar la colección.
    coleccion = base_datos[MONGO_COLLECTION]

    return cliente, coleccion


# ------------------------------------------------------------
# 7. CONSULTAR TODOS LOS ANALISIS
# ------------------------------------------------------------

def consultar_todos(coleccion):

    """
    Recupera todos los análisis registrados
    por el sistema SmartAcademy.
    """

    print("\n1. CONSULTA GENERAL DE ANALISIS")

    print("-" * 60)

    # find permite recuperar documentos que cumplan
    # con los criterios indicados.
    cursor = coleccion.find(
        {
            "sistema": "SmartAcademy"
        },
        {
            "_id": 0,
            "id_analisis": 1,
            "fecha_registro": 1,
            "datos_reporte": 1
        }
    ).sort(
        "fecha_registro",
        -1
    )

    registros = list(cursor)

    print(
        "Total de análisis encontrados:",
        len(registros)
    )

    # Mostrar información resumida.
    for numero, registro in enumerate(registros, start=1):

        datos = registro.get("datos_reporte", {})

        print(f"\nANALISIS {numero}")

        print(
            "ID:",
            registro.get("id_analisis")
        )

        print(
            "Horas de estudio:",
            datos.get("horas_estudio")
        )

        print(
            "Asistencia:",
            datos.get("asistencia")
        )

        print(
            "Actividades:",
            datos.get("actividades")
        )

        print(
            "Participación:",
            datos.get("participacion")
        )

        print(
            "Resultado:",
            datos.get("resultado")
        )

        print(
            "Perfil:",
            datos.get("perfil")
        )

    return registros


# ------------------------------------------------------------
# 8. IDENTIFICAR ANALISIS CON RIESGO ESTIMADO
# ------------------------------------------------------------

def consultar_riesgos(coleccion):

    """
    Utiliza $match para filtrar los análisis
    cuya predicción indica posible no aprobación.

    prediccion = 0 representa no aprobación estimada.
    """

    print("\n2. ANALISIS CON RIESGO ESTIMADO")

    print("-" * 60)

    pipeline = [

        # Etapa 1: filtrar los documentos.
        {
            "$match": {
                "sistema": "SmartAcademy",
                "datos_reporte.prediccion": 0
            }
        },

        # Etapa 2: seleccionar los campos necesarios.
        {
            "$project": {

                "_id": 0,

                "horas_estudio":
                    "$datos_reporte.horas_estudio",

                "asistencia":
                    "$datos_reporte.asistencia",

                "actividades":
                    "$datos_reporte.actividades",

                "participacion":
                    "$datos_reporte.participacion",

                "probabilidad_aprobar":
                    "$datos_reporte.probabilidad_aprobar",

                "perfil":
                    "$datos_reporte.perfil"
            }
        },

        # Etapa 3: ordenar por menor probabilidad
        # estimada de aprobación.
        {
            "$sort": {
                "probabilidad_aprobar": 1
            }
        }

    ]

    resultados = list(
        coleccion.aggregate(pipeline)
    )

    print(
        "Análisis con riesgo estimado:",
        len(resultados)
    )

    for resultado in resultados:

        print("\nEstudiante analizado")

        print(
            "Horas:",
            resultado.get("horas_estudio")
        )

        print(
            "Asistencia:",
            resultado.get("asistencia")
        )

        print(
            "Probabilidad de aprobar:",
            resultado.get("probabilidad_aprobar")
        )

        print(
            "Perfil:",
            resultado.get("perfil")
        )

    return resultados


# ------------------------------------------------------------
# 9. AGRUPAR ESTUDIANTES POR PERFIL
# ------------------------------------------------------------

def analizar_perfiles(coleccion):

    """
    Utiliza $group para agrupar los análisis
    de acuerdo con el perfil identificado por K-Means.

    Calcula:
    - Cantidad de análisis.
    - Promedio de horas de estudio.
    - Promedio de asistencia.
    - Promedio de actividades.
    - Promedio de participación.
    """

    print("\n3. DISTRIBUCION POR PERFIL ACADEMICO")

    print("-" * 60)

    pipeline = [

        # Etapa 1: filtrar SmartAcademy.
        {
            "$match": {
                "sistema": "SmartAcademy"
            }
        },

        # Etapa 2: agrupar por perfil.
        {
            "$group": {

                "_id": "$datos_reporte.perfil",

                "total_analisis": {
                    "$sum": 1
                },

                "promedio_horas": {
                    "$avg": "$datos_reporte.horas_estudio"
                },

                "promedio_asistencia": {
                    "$avg": "$datos_reporte.asistencia"
                },

                "promedio_actividades": {
                    "$avg": "$datos_reporte.actividades"
                },

                "promedio_participacion": {
                    "$avg": "$datos_reporte.participacion"
                }
            }
        },

        # Etapa 3: ordenar por cantidad.
        {
            "$sort": {
                "total_analisis": -1
            }
        }

    ]

    resultados = list(
        coleccion.aggregate(pipeline)
    )

    for perfil in resultados:

        print("\nPerfil:", perfil.get("_id"))

        print(
            "Cantidad de análisis:",
            perfil.get("total_analisis")
        )

        print(
            "Promedio de horas:",
            round(perfil.get("promedio_horas") or 0, 2)
        )

        print(
            "Promedio de asistencia:",
            round(perfil.get("promedio_asistencia") or 0, 2)
        )

        print(
            "Promedio de actividades:",
            round(perfil.get("promedio_actividades") or 0, 2)
        )

        print(
            "Promedio de participación:",
            round(perfil.get("promedio_participacion") or 0, 2)
        )

    return resultados


# ------------------------------------------------------------
# 10. GENERAR ESTADISTICAS GENERALES
# ------------------------------------------------------------

def generar_estadisticas(coleccion):

    """
    Utiliza $group para calcular estadísticas
    generales de todos los análisis registrados.

    Se calculan:
    - Total de análisis.
    - Promedio de horas de estudio.
    - Promedio de asistencia.
    - Promedio de actividades.
    - Promedio de participación.
    - Promedio de probabilidad de aprobación.
    - Cantidad de predicciones de riesgo.
    """

    print("\n4. ESTADISTICAS GENERALES")

    print("-" * 60)

    pipeline = [

        # Etapa 1: seleccionar SmartAcademy.
        {
            "$match": {
                "sistema": "SmartAcademy"
            }
        },

        # Etapa 2: calcular estadísticas.
        {
            "$group": {

                "_id": None,

                "total_analisis": {
                    "$sum": 1
                },

                "promedio_horas": {
                    "$avg": "$datos_reporte.horas_estudio"
                },

                "promedio_asistencia": {
                    "$avg": "$datos_reporte.asistencia"
                },

                "promedio_actividades": {
                    "$avg": "$datos_reporte.actividades"
                },

                "promedio_participacion": {
                    "$avg": "$datos_reporte.participacion"
                },

                "promedio_probabilidad_aprobar": {
                    "$avg": "$datos_reporte.probabilidad_aprobar"
                },

                "total_riesgo": {
                    "$sum": {
                        "$cond": [
                            {
                                "$eq": [
                                    "$datos_reporte.prediccion",
                                    0
                                ]
                            },
                            1,
                            0
                        ]
                    }
                },

                "total_aprobacion_estimada": {
                    "$sum": {
                        "$cond": [
                            {
                                "$eq": [
                                    "$datos_reporte.prediccion",
                                    1
                                ]
                            },
                            1,
                            0
                        ]
                    }
                }
            }
        }

    ]

    resultados = list(
        coleccion.aggregate(pipeline)
    )

    if not resultados:

        print("No existen registros para analizar.")

        return None

    estadisticas = resultados[0]

    print(
        "Total de análisis:",
        estadisticas.get("total_analisis")
    )

    print(
        "Promedio de horas:",
        round(estadisticas.get("promedio_horas") or 0, 2)
    )

    print(
        "Promedio de asistencia:",
        round(estadisticas.get("promedio_asistencia") or 0, 2)
    )

    print(
        "Promedio de actividades:",
        round(estadisticas.get("promedio_actividades") or 0, 2)
    )

    print(
        "Promedio de participación:",
        round(estadisticas.get("promedio_participacion") or 0, 2)
    )

    print(
        "Promedio de probabilidad de aprobar:",
        round(
            (estadisticas.get("promedio_probabilidad_aprobar") or 0)
            * 100,
            2
        ),
        "%"
    )

    print(
        "Total de riesgos estimados:",
        estadisticas.get("total_riesgo")
    )

    print(
        "Total de aprobaciones estimadas:",
        estadisticas.get("total_aprobacion_estimada")
    )

    return estadisticas


# ------------------------------------------------------------
# 11. IDENTIFICAR LOS ANALISIS CON MAYOR PROBABILIDAD
# ------------------------------------------------------------

def consultar_mejores_resultados(coleccion):

    """
    Ordena los análisis desde la mayor hasta la menor
    probabilidad estimada de aprobación.

    Limitamos la consulta a cinco documentos.
    """

    print("\n5. MAYORES PROBABILIDADES DE APROBACION")

    print("-" * 60)

    pipeline = [

        {
            "$match": {
                "sistema": "SmartAcademy"
            }
        },

        {
            "$project": {

                "_id": 0,

                "horas_estudio":
                    "$datos_reporte.horas_estudio",

                "asistencia":
                    "$datos_reporte.asistencia",

                "probabilidad_aprobar":
                    "$datos_reporte.probabilidad_aprobar",

                "perfil":
                    "$datos_reporte.perfil"
            }
        },

        {
            "$sort": {
                "probabilidad_aprobar": -1
            }
        },

        {
            "$limit": 5
        }

    ]

    resultados = list(
        coleccion.aggregate(pipeline)
    )

    for numero, resultado in enumerate(resultados, start=1):

        print(f"\nPOSICION {numero}")

        print(
            "Probabilidad de aprobar:",
            round(
                resultado.get("probabilidad_aprobar", 0) * 100,
                2
            ),
            "%"
        )

        print(
            "Horas de estudio:",
            resultado.get("horas_estudio")
        )

        print(
            "Asistencia:",
            resultado.get("asistencia")
        )

        print(
            "Perfil:",
            resultado.get("perfil")
        )

    return resultados


# ------------------------------------------------------------
# 12. EXPORTAR RESULTADOS A ARCHIVOS CSV
# ------------------------------------------------------------

def exportar_reportes(
    registros,
    riesgos,
    perfiles,
    estadisticas
):

    """
    Exporta los resultados de las consultas
    a archivos CSV para su análisis posterior.
    """

    print("\n6. EXPORTACION DE REPORTES")

    print("-" * 60)

    # --------------------------------------------------------
    # Exportar todos los análisis.
    # --------------------------------------------------------

    filas = []

    for registro in registros:

        datos = registro.get("datos_reporte", {})

        filas.append({

            "id_analisis":
                registro.get("id_analisis"),

            "horas_estudio":
                datos.get("horas_estudio"),

            "asistencia":
                datos.get("asistencia"),

            "actividades":
                datos.get("actividades"),

            "participacion":
                datos.get("participacion"),

            "prediccion":
                datos.get("prediccion"),

            "resultado":
                datos.get("resultado"),

            "probabilidad_aprobar":
                datos.get("probabilidad_aprobar"),

            "perfil":
                datos.get("perfil"),

            "cluster":
                datos.get("cluster")
        })

    if filas:

        dataframe = pd.DataFrame(filas)

        ruta = (
            RUTA_RESULTADOS
            / "analisis_historico.csv"
        )

        dataframe.to_csv(
            ruta,
            index=False,
            encoding="utf-8-sig"
        )

        print("Reporte histórico:", ruta)

    # --------------------------------------------------------
    # Exportar análisis con riesgo.
    # --------------------------------------------------------

    if riesgos:

        dataframe_riesgos = pd.DataFrame(riesgos)

        ruta_riesgos = (
            RUTA_RESULTADOS
            / "analisis_riesgos.csv"
        )

        dataframe_riesgos.to_csv(
            ruta_riesgos,
            index=False,
            encoding="utf-8-sig"
        )

        print("Reporte de riesgos:", ruta_riesgos)

    # --------------------------------------------------------
    # Exportar perfiles.
    # --------------------------------------------------------

    if perfiles:

        dataframe_perfiles = pd.DataFrame(perfiles)

        # Cambiar el nombre de la columna _id.
        dataframe_perfiles = dataframe_perfiles.rename(
            columns={
                "_id": "perfil"
            }
        )

        ruta_perfiles = (
            RUTA_RESULTADOS
            / "estadisticas_perfiles.csv"
        )

        dataframe_perfiles.to_csv(
            ruta_perfiles,
            index=False,
            encoding="utf-8-sig"
        )

        print("Reporte de perfiles:", ruta_perfiles)

    # --------------------------------------------------------
    # Exportar estadísticas generales.
    # --------------------------------------------------------

    if estadisticas:

        dataframe_estadisticas = pd.DataFrame(
            [estadisticas]
        )

        ruta_estadisticas = (
            RUTA_RESULTADOS
            / "estadisticas_generales.csv"
        )

        dataframe_estadisticas.to_csv(
            ruta_estadisticas,
            index=False,
            encoding="utf-8-sig"
        )

        print(
            "Reporte de estadísticas:",
            ruta_estadisticas
        )


# ------------------------------------------------------------
# 13. FUNCION PRINCIPAL
# ------------------------------------------------------------

def main():

    """
    Ejecuta todas las consultas y análisis
    estadísticos de SmartAcademy.
    """

    print("=" * 60)

    print("SMARTACADEMY")

    print("ETAPA 7 - CONSULTAS Y ANALISIS ESTADISTICO")

    print("=" * 60)

    cliente = None

    try:

        # Paso 1: conectar con MongoDB.
        cliente, coleccion = conectar_mongodb()

        # Paso 2: consultar todos los análisis.
        registros = consultar_todos(coleccion)

        if not registros:

            print(
                "\nNo existen análisis almacenados "
                "para SmartAcademy."
            )

            return

        # Paso 3: consultar análisis con riesgo.
        riesgos = consultar_riesgos(coleccion)

        # Paso 4: analizar perfiles.
        perfiles = analizar_perfiles(coleccion)

        # Paso 5: estadísticas generales.
        estadisticas = generar_estadisticas(coleccion)

        # Paso 6: mejores resultados estimados.
        consultar_mejores_resultados(coleccion)

        # Paso 7: exportar reportes.
        exportar_reportes(
            registros,
            riesgos,
            perfiles,
            estadisticas
        )

        print("\n" + "=" * 60)

        print("ANALISIS ESTADISTICO FINALIZADO")

        print("=" * 60)

        print(
            "Base de datos:",
            MONGO_DB
        )

        print(
            "Colección:",
            MONGO_COLLECTION
        )

    except ValueError as error:

        print("\nERROR DE CONFIGURACION:")
        print(error)

    except PyMongoError as error:

        print("\nERROR DE MONGODB:")
        print(error)

    except Exception as error:

        print("\nERROR INESPERADO:")
        print(error)

    finally:

        # Cerrar conexión con MongoDB.
        if cliente is not None:

            cliente.close()

            print("\nConexión con MongoDB cerrada.")


# ------------------------------------------------------------
# 14. PUNTO DE ENTRADA
# ------------------------------------------------------------

if __name__ == "__main__":

    main()