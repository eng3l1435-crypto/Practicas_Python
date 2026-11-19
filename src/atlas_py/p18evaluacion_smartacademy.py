
# ============================================================
# PROYECTO: SMARTACADEMY
# Archivo: p18evaluacion_smartacademy.py
# Etapa 9: Evaluación y registro de estudiantes
#
# OBJETIVO:
# Integrar los modelos de Machine Learning desarrollados
# anteriormente en una aplicación web con Streamlit.
#
# FUNCIONALIDADES:
# 1. Capturar información del estudiante.
# 2. Predecir la aprobación mediante Random Forest.
# 3. Identificar su perfil mediante K-Means.
# 4. Proyectar sus características mediante PCA.
# 5. Generar recomendaciones académicas.
# 6. Guardar la evaluación en MongoDB Atlas.
# 7. Consultar el historial individual del estudiante.
#
# IMPORTANTE:
# No se entrenan modelos nuevos.
# Se reutilizan los modelos de las etapas anteriores.
# ============================================================


# ============================================================
# 1. IMPORTAR LIBRERÍAS
# ============================================================

import os
import uuid
import sys

from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import quote_plus

import certifi
import pandas as pd
import streamlit as st

from dotenv import load_dotenv

from pymongo import MongoClient
from pymongo.errors import PyMongoError


# ============================================================
# CONFIGURAR RUTA RAÍZ DEL PROYECTO
# ============================================================

# Obtener la ubicación del archivo actual.
RUTA_ARCHIVO = Path(__file__).resolve()

# Identificar la carpeta principal:
# bigdata_ventas/
RUTA_PROYECTO = RUTA_ARCHIVO.parents[1]

# Agregar la raíz a las rutas de búsqueda de Python.
# Esto permite localizar el paquete unidad3_fia.
if str(RUTA_PROYECTO) not in sys.path:

    sys.path.insert(
        0,
        str(RUTA_PROYECTO)
    )

# Importamos el motor de recomendaciones de la etapa 5.
# Este módulo contiene las funciones que ya desarrollamos.
from unidad3_fia import p14motor_recomendaciones as motor


# ============================================================
# 2. CONFIGURACIÓN GENERAL
# ============================================================

# Cargar las variables de entorno.
load_dotenv(RUTA_PROYECTO / ".env")


# Configuración de MongoDB Atlas.
MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")
MONGO_CLUSTER = os.getenv("MONGO_CLUSTER")

MONGO_DB = os.getenv(
    "MONGO_DB",
    "smartacademy_db"
)

MONGO_COLLECTION = os.getenv(
    "MONGO_COLLECTION",
    "analisis_estudiantes"
)


# ============================================================
# 3. CONFIGURAR INTERFAZ STREAMLIT
# ============================================================

st.set_page_config(
    page_title="SmartAcademy - Evaluación",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 SMARTACADEMY")

st.subheader(
    "Sistema inteligente de evaluación y recomendaciones académicas"
)

st.markdown(
    """
    Esta aplicación integra tres técnicas de inteligencia artificial:

    - **Random Forest:** estimación de aprobación académica.
    - **K-Means:** identificación de perfiles de aprendizaje.
    - **PCA:** reducción de dimensionalidad para análisis exploratorio.
    """
)

st.divider()


# ============================================================
# 4. CARGAR LOS MODELOS EXISTENTES
# ============================================================

@st.cache_resource
def cargar_modelos_smartacademy():

    """
    Carga los modelos previamente entrenados.

    No realiza entrenamiento.
    Los modelos se recuperan desde los archivos
    generados en las etapas anteriores.
    """

    return motor.cargar_modelos()


try:

    modelos = cargar_modelos_smartacademy()

except FileNotFoundError as error:

    st.error(
        "No se encontraron los archivos de los modelos. "
        "Ejecuta primero las etapas anteriores."
    )

    st.exception(error)

    st.stop()

except Exception as error:

    st.error("No fue posible cargar los modelos.")

    st.exception(error)

    st.stop()


st.success(
    "Los modelos de Machine Learning se cargaron correctamente."
)


# ============================================================
# 5. FUNCIONES DE CONEXIÓN A MONGODB ATLAS
# ============================================================

def construir_uri():

    """
    Construye la cadena de conexión de MongoDB Atlas.

    Las credenciales se obtienen del archivo .env.
    No deben escribirse directamente en el código.
    """

    if not all([
        MONGO_USER,
        MONGO_PASSWORD,
        MONGO_CLUSTER
    ]):

        raise ValueError(
            "Faltan variables de conexión en el archivo .env."
        )

    usuario = quote_plus(MONGO_USER)

    contrasena = quote_plus(MONGO_PASSWORD)

    uri = (
        f"mongodb+srv://{usuario}:{contrasena}"
        f"@{MONGO_CLUSTER}/"
        "?retryWrites=true&w=majority"
        "&appName=SmartAcademy"
    )

    return uri


def conectar_mongodb():

    """
    Establece conexión con MongoDB Atlas.

    Retorna:
        cliente: conexión principal.
        coleccion: colección de SmartAcademy.
    """

    uri = construir_uri()

    cliente = MongoClient(
        uri,
        tlsCAFile=certifi.where(),
        serverSelectionTimeoutMS=10000
    )

    # Verificar que Atlas responda.
    cliente.admin.command("ping")

    base_datos = cliente[MONGO_DB]

    coleccion = base_datos[MONGO_COLLECTION]

    # El identificador del análisis debe ser único.
    coleccion.create_index(
        "id_analisis",
        unique=True
    )

    # Un estudiante puede tener muchas evaluaciones.
    # Por eso este índice NO debe ser único.
    coleccion.create_index(
        "id_estudiante"
    )

    coleccion.create_index(
        "fecha_registro"
    )

    return cliente, coleccion


# ============================================================
# 6. CAPTURA DE INFORMACIÓN DEL ESTUDIANTE
# ============================================================

st.header("1. Información del estudiante")

st.info(
    "Utiliza el mismo identificador en evaluaciones posteriores "
    "para consultar el historial del estudiante."
)

with st.form("formulario_evaluacion"):

    id_estudiante = st.text_input(
        "Identificador permanente del estudiante",
        placeholder="Ejemplo: IDGS-2026-001"
    )

    col1, col2 = st.columns(2)

    with col1:

        horas_estudio = st.number_input(
            "Horas de estudio por semana",
            min_value=0.0,
            max_value=24.0,
            value=5.0,
            step=0.5
        )

        asistencia = st.number_input(
            "Porcentaje de asistencia",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            step=1.0
        )

    with col2:

        actividades = st.number_input(
            "Actividades realizadas",
            min_value=0,
            max_value=30,
            value=15,
            step=1
        )

        participacion = st.number_input(
            "Porcentaje de participación",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=1.0
        )

    enviar = st.form_submit_button(
        "Evaluar estudiante y guardar resultados",
        use_container_width=True
    )


# ============================================================
# 7. PROCESAR LA EVALUACIÓN
# ============================================================

if enviar:

    # --------------------------------------------------------
    # 7.1 VALIDAR IDENTIFICADOR
    # --------------------------------------------------------

    if not id_estudiante.strip():

        st.error(
            "Debes proporcionar el identificador del estudiante."
        )

        st.stop()

    # --------------------------------------------------------
    # 7.2 CONSTRUIR DATAFRAME
    # --------------------------------------------------------

    # Se utiliza el mismo orden de características
    # con el que fueron entrenados los modelos.

    estudiante = pd.DataFrame(
        [[
            float(horas_estudio),
            float(asistencia),
            float(actividades),
            float(participacion)
        ]],
        columns=[
            "horas_estudio",
            "asistencia",
            "actividades",
            "participacion"
        ]
    )

    st.divider()

    st.header("2. Resultados del análisis inteligente")

    # --------------------------------------------------------
    # 7.3 RANDOM FOREST
    # --------------------------------------------------------

    try:

        resultado_prediccion = motor.predecir_aprobacion(
            estudiante,
            modelos["supervisado"]
        )

        # ----------------------------------------------------
        # 7.4 K-MEANS
        # ----------------------------------------------------

        resultado_perfil = motor.identificar_perfil(
            estudiante,
            modelos["kmeans"],
            modelos["escalador_kmeans"],
            modelos["mapa_perfiles"]
        )

        # ----------------------------------------------------
        # 7.5 PCA
        # ----------------------------------------------------

        resultado_pca = motor.proyectar_pca(
            estudiante,
            modelos["pca"],
            modelos["escalador_pca"]
        )

        # ----------------------------------------------------
        # 7.6 RECOMENDACIONES
        # ----------------------------------------------------

        recomendaciones = motor.generar_recomendaciones(
            estudiante,
            resultado_prediccion,
            resultado_perfil
        )

    except Exception as error:

        st.error(
            "Ocurrió un problema durante el análisis."
        )

        st.exception(error)

        st.stop()


    # ========================================================
    # 8. MOSTRAR RESULTADOS
    # ========================================================

    st.subheader("Predicción de aprobación")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Probabilidad de aprobar",
            f"{resultado_prediccion['probabilidad_aprobar']:.2f}%"
        )

    with col2:

        st.metric(
            "Probabilidad de no aprobar",
            f"{resultado_prediccion['probabilidad_no_aprobar']:.2f}%"
        )

    with col3:

        st.metric(
            "Perfil académico",
            resultado_perfil["perfil"]
        )

    if resultado_prediccion["prediccion"] == 1:

        st.success(
            resultado_prediccion["resultado"]
        )

    else:

        st.warning(
            resultado_prediccion["resultado"]
        )

    st.caption(
        "Las probabilidades son estimaciones del modelo y no "
        "constituyen una calificación ni una decisión académica definitiva."
    )


    # ========================================================
    # 9. MOSTRAR PERFIL Y COMPONENTES PCA
    # ========================================================

    st.subheader("Análisis de aprendizaje no supervisado")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Cluster identificado",
            resultado_perfil["cluster"]
        )

        st.write(
            "Perfil:",
            resultado_perfil["perfil"]
        )

    with col2:

        st.metric(
            "Componente principal PC1",
            round(resultado_pca["PC1"], 4)
        )

        st.metric(
            "Componente principal PC2",
            round(resultado_pca["PC2"], 4)
        )


    # ========================================================
    # 10. MOSTRAR RECOMENDACIONES
    # ========================================================

    st.subheader("Recomendaciones académicas")

    for numero, recomendacion in enumerate(
        recomendaciones,
        start=1
    ):

        st.write(
            f"**{numero}.** {recomendacion}"
        )


    # ========================================================
    # 11. CONSTRUIR DOCUMENTO PARA MONGODB
    # ========================================================

    # Se genera un identificador nuevo para cada evaluación.
    id_analisis = str(uuid.uuid4())

    # Los datos se guardan dentro de datos_reporte
    # para mantener compatibilidad con el dashboard p17.

    documento = {

        "id_analisis": id_analisis,

        "id_estudiante": id_estudiante.strip(),

        "sistema": "SmartAcademy",

        "etapa": 9,

        "fecha_registro": datetime.now(timezone.utc),

        "archivo_origen": "p18evaluacion_smartacademy.py",

        "datos_reporte": {

            "id_estudiante": id_estudiante.strip(),

            "horas_estudio": float(horas_estudio),

            "asistencia": float(asistencia),

            "actividades": int(actividades),

            "participacion": float(participacion),

            "prediccion": int(
                resultado_prediccion["prediccion"]
            ),

            "resultado": resultado_prediccion["resultado"],

            "probabilidad_aprobar": float(
                resultado_prediccion["probabilidad_aprobar"]
            ),

            "probabilidad_no_aprobar": float(
                resultado_prediccion["probabilidad_no_aprobar"]
            ),

            "cluster": int(
                resultado_perfil["cluster"]
            ),

            "perfil": resultado_perfil["perfil"],

            "PC1": float(
                resultado_pca["PC1"]
            ),

            "PC2": float(
                resultado_pca["PC2"]
            ),

            # Se almacena como una lista BSON,
            # no como una cadena de texto.
            "recomendaciones": recomendaciones

        }

    }


    # ========================================================
    # 12. GUARDAR EVALUACIÓN EN MONGODB ATLAS
    # ========================================================

    cliente = None

    try:

        with st.spinner(
            "Conectando con MongoDB Atlas..."
        ):

            cliente, coleccion = conectar_mongodb()

            resultado = coleccion.insert_one(
                documento
            )

        st.success(
            "La evaluación se almacenó correctamente en MongoDB Atlas."
        )

        st.write(
            "**Identificador del estudiante:**",
            id_estudiante.strip()
        )

        st.write(
            "**Identificador de la evaluación:**",
            id_analisis
        )

        st.write(
            "**Base de datos:**",
            MONGO_DB
        )

        st.write(
            "**Colección:**",
            MONGO_COLLECTION
        )

    except ValueError as error:

        st.error(
            f"Error de configuración: {error}"
        )

    except PyMongoError as error:

        st.error(
            "No fue posible guardar la evaluación en MongoDB Atlas."
        )

        st.exception(error)

    except Exception as error:

        st.error(
            "Ocurrió un error inesperado al guardar la evaluación."
        )

        st.exception(error)

    finally:

        if cliente is not None:

            cliente.close()


# ============================================================
# 13. CONSULTAR HISTORIAL DEL ESTUDIANTE
# ============================================================

st.divider()

st.header("3. Historial de evaluaciones")

st.write(
    "Consulta las evaluaciones anteriores utilizando "
    "el identificador permanente del estudiante."
)

id_consulta = st.text_input(
    "Identificador del estudiante para consultar su historial",
    key="id_consulta",
    placeholder="Ejemplo: IDGS-2026-001"
)

if st.button(
    "Consultar historial",
    use_container_width=True
):

    if not id_consulta.strip():

        st.warning(
            "Escribe el identificador del estudiante."
        )

    else:

        cliente = None

        try:

            cliente, coleccion = conectar_mongodb()

            registros = list(
                coleccion.find(
                    {
                        "id_estudiante": id_consulta.strip(),
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
            )

            if registros:

                st.success(
                    f"Se encontraron {len(registros)} evaluaciones."
                )

                filas = []

                for registro in registros:

                    datos = registro.get(
                        "datos_reporte",
                        {}
                    )

                    filas.append({

                        "ID análisis": registro.get(
                            "id_analisis"
                        ),

                        "Fecha": registro.get(
                            "fecha_registro"
                        ),

                        "Horas de estudio": datos.get(
                            "horas_estudio"
                        ),

                        "Asistencia": datos.get(
                            "asistencia"
                        ),

                        "Actividades": datos.get(
                            "actividades"
                        ),

                        "Participación": datos.get(
                            "participacion"
                        ),

                        "Predicción": datos.get(
                            "resultado"
                        ),

                        "Probabilidad de aprobar": datos.get(
                            "probabilidad_aprobar"
                        ),

                        "Perfil": datos.get(
                            "perfil"
                        )

                    })

                historial = pd.DataFrame(filas)

                st.dataframe(
                    historial,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.info(
                    "No existen evaluaciones para ese estudiante."
                )

        except PyMongoError as error:

            st.error(
                "No fue posible consultar MongoDB Atlas."
            )

            st.exception(error)

        except Exception as error:

            st.error(
                "Ocurrió un error al consultar el historial."
            )

            st.exception(error)

        finally:

            if cliente is not None:

                cliente.close()


# ============================================================
# 14. PIE DE PÁGINA
# ============================================================

st.divider()

st.caption(
    "SMARTACADEMY | Sistema de análisis y recomendaciones "
    "académicas basado en Machine Learning."
)
