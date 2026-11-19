# ============================================================
# PROYECTO: SMARTACADEMY
# ETAPA 8: DASHBOARD ACADEMICO INTELIGENTE
#
# Archivo:
# p17dashboard_smartacademy.py
#
# Objetivo:
# Construir un dashboard web para consultar y visualizar
# los análisis académicos almacenados en MongoDB Atlas.
#
# Tecnologías:
# - Python
# - Streamlit
# - Pandas
# - Plotly
# - PyMongo
# - MongoDB Atlas
# ============================================================


# ------------------------------------------------------------
# 1. IMPORTAR LIBRERIAS
# ------------------------------------------------------------

import os

import streamlit as st
import pandas as pd
import plotly.express as px

from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError


# ------------------------------------------------------------
# 2. CONFIGURAR LA PAGINA
# ------------------------------------------------------------

# Esta configuración debe ejecutarse antes de crear
# cualquier otro elemento visual de Streamlit.

st.set_page_config(

    page_title="SmartAcademy | Dashboard",

    page_icon="ICONO USTEDE DISEÑEN",

    layout="wide",

    initial_sidebar_state="expanded"
)


# ------------------------------------------------------------
# 3. CONFIGURACION DE RUTAS
# ------------------------------------------------------------

# Obtener la ubicación del archivo actual.
RUTA_ARCHIVO = Path(__file__).resolve()

# Obtener la raíz del proyecto.
RUTA_PROYECTO = RUTA_ARCHIVO.parents[1]

# Cargar las variables del archivo .env.
load_dotenv(RUTA_PROYECTO / ".env")


# ------------------------------------------------------------
# 4. RECUPERAR VARIABLES DE ENTORNO
# ------------------------------------------------------------

MONGO_USER = os.getenv("MONGO_USER")

MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")

MONGO_CLUSTER = os.getenv("MONGO_CLUSTER")

MONGO_DB = os.getenv("MONGO_DB")

MONGO_COLLECTION = os.getenv("MONGO_COLLECTION")


# ------------------------------------------------------------
# 5. VALIDAR CONFIGURACION
# ------------------------------------------------------------

def validar_configuracion():

    """
    Comprueba que existan las variables necesarias
    para conectarse a MongoDB Atlas.
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

            "Faltan variables en .env: "

            + ", ".join(faltantes)

        )


# ------------------------------------------------------------
# 6. CONSTRUIR URI DE MONGODB
# ------------------------------------------------------------

def construir_uri():

    """
    Construye la cadena de conexión a MongoDB Atlas.

    quote_plus codifica caracteres especiales de las
    credenciales para evitar problemas en la URI.
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
# 7. CREAR CONEXION A MONGODB
# ------------------------------------------------------------

@st.cache_resource
def obtener_cliente():

    """
    Crea un cliente reutilizable de MongoDB.

    cache_resource evita crear una conexión nueva
    cada vez que Streamlit actualiza la interfaz.
    """

    validar_configuracion()

    cliente = MongoClient(

        construir_uri(),

        serverSelectionTimeoutMS=15000

    )

    # Comprobar que Atlas responda.
    cliente.admin.command("ping")

    return cliente


# ------------------------------------------------------------
# 8. CONSULTAR DOCUMENTOS DE MONGODB
# ------------------------------------------------------------

@st.cache_data(ttl=60)
def cargar_datos():

    """
    Consulta los documentos almacenados en Atlas.

    ttl=60 indica que los datos se mantienen en caché
    durante 60 segundos.
    """

    cliente = obtener_cliente()

    coleccion = cliente[MONGO_DB][MONGO_COLLECTION]

    # Consultar exclusivamente los análisis SmartAcademy.
    documentos = coleccion.find(

        {
            "sistema": "SmartAcademy"
        },

        {
            "_id": 0
        }

    ).sort(

        "fecha_registro",

        -1

    )

    # Convertir el cursor en una lista.
    documentos = list(documentos)

    registros = []

    # --------------------------------------------------------
    # Convertir documentos anidados a registros tabulares.
    # --------------------------------------------------------

    for documento in documentos:

        datos = documento.get("datos_reporte", {})

        registro = {

            "id_analisis":
                documento.get("id_analisis"),

            "fecha_registro":
                documento.get("fecha_registro"),

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

            "probabilidad_no_aprobar":
                datos.get("probabilidad_no_aprobar"),

            "cluster":
                datos.get("cluster"),

            "perfil":
                datos.get("perfil"),

            "PC1":
                datos.get("PC1"),

            "PC2":
                datos.get("PC2"),

            "recomendaciones":
                datos.get("recomendaciones")

        }

        registros.append(registro)

    # Crear un DataFrame de Pandas.
    dataframe = pd.DataFrame(registros)

    return dataframe


# ------------------------------------------------------------
# 9. PREPARAR LOS DATOS
# ------------------------------------------------------------

def preparar_datos(dataframe):

    """
    Convierte las columnas numéricas a tipos apropiados.

    Esto facilita la construcción de gráficas y cálculos.
    """

    columnas_numericas = [

        "horas_estudio",

        "asistencia",

        "actividades",

        "participacion",

        "prediccion",

        "probabilidad_aprobar",

        "probabilidad_no_aprobar",

        "cluster",

        "PC1",

        "PC2"

    ]

    for columna in columnas_numericas:

        if columna in dataframe.columns:

            dataframe[columna] = pd.to_numeric(

                dataframe[columna],

                errors="coerce"

            )

    if "fecha_registro" in dataframe.columns:

        dataframe["fecha_registro"] = pd.to_datetime(

            dataframe["fecha_registro"],

            errors="coerce",

            utc=True

        )

    return dataframe


# ------------------------------------------------------------
# 10. ENCABEZADO DEL DASHBOARD
# ------------------------------------------------------------

def mostrar_encabezado():

    """
    Presenta el título y descripción del sistema.
    """

    st.title(" SMARTACADEMY")

    st.subheader(

        "Dashboard de análisis académico inteligente"

    )

    st.markdown(

        """
        Plataforma de consulta y visualización de resultados
        generados mediante aprendizaje automático y almacenados
        en MongoDB Atlas.
        """

    )

    st.divider()


# ------------------------------------------------------------
# 11. CONSTRUIR INDICADORES KPI
# ------------------------------------------------------------

def mostrar_indicadores(dataframe):

    """
    Presenta indicadores generales del conjunto filtrado.
    """

    st.subheader("1. Indicadores generales")

    total = len(dataframe)

    riesgos = int(

        (dataframe["prediccion"] == 0).sum()

    )

    aprobaciones = int(

        (dataframe["prediccion"] == 1).sum()

    )

    if total > 0:

        porcentaje_riesgo = riesgos / total * 100

        promedio_aprobar = (

            dataframe["probabilidad_aprobar"].mean()

            * 100

        )

        promedio_asistencia = (

            dataframe["asistencia"].mean()

        )

    else:

        porcentaje_riesgo = 0

        promedio_aprobar = 0

        promedio_asistencia = 0

    # Crear cuatro columnas para los indicadores.
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(

        "Total de análisis",

        total

    )

    col2.metric(

        "Riesgos estimados",

        riesgos,

        f"{porcentaje_riesgo:.2f}% del total"

    )

    col3.metric(

        "Aprobaciones estimadas",

        aprobaciones

    )

    col4.metric(

        "Asistencia promedio",

        f"{promedio_asistencia:.2f}%"

    )

    st.metric(

        "Probabilidad promedio estimada de aprobación",

        f"{promedio_aprobar:.2f}%"

    )


# ------------------------------------------------------------
# 12. GRAFICA DE PREDICCIONES
# ------------------------------------------------------------

def mostrar_predicciones(dataframe):

    """
    Construye una gráfica de distribución de predicciones.
    """

    st.subheader("2. Distribución de predicciones")

    conteo = dataframe["prediccion"].value_counts(
        dropna=False
    )

    nombres = {

        0: "Riesgo estimado",

        1: "Aprobación estimada"

    }

    datos_grafica = pd.DataFrame({

        "prediccion": conteo.index,

        "cantidad": conteo.values

    })

    datos_grafica["resultado"] = (

        datos_grafica["prediccion"]

        .map(nombres)

        .fillna("Sin clasificar")

    )

    figura = px.pie(

        datos_grafica,

        names="resultado",

        values="cantidad",

        title="Distribución de resultados estimados",

        hole=0.45

    )

    figura.update_traces(

        textposition="inside",

        textinfo="label+percent+value"

    )

    st.plotly_chart(

        figura,

        use_container_width=True

    )


# ------------------------------------------------------------
# 13. GRAFICA DE PERFILES K-MEANS
# ------------------------------------------------------------

def mostrar_perfiles(dataframe):

    """
    Muestra cuántos análisis corresponden a cada
    perfil identificado por el algoritmo K-Means.
    """

    st.subheader("3. Distribución de perfiles académicos")

    datos = dataframe.dropna(

        subset=["perfil"]

    )

    if datos.empty:

        st.info("No existen perfiles disponibles.")

        return

    conteo = (

        datos.groupby("perfil")

        .size()

        .reset_index(name="cantidad")

    )

    figura = px.bar(

        conteo,

        x="perfil",

        y="cantidad",

        text="cantidad",

        title="Cantidad de análisis por perfil",

        labels={

            "perfil": "Perfil académico",

            "cantidad": "Cantidad de análisis"

        }

    )

    figura.update_traces(

        textposition="outside"

    )

    st.plotly_chart(

        figura,

        use_container_width=True

    )


# ------------------------------------------------------------
# 14. GRAFICA DE CARACTERISTICAS ACADEMICAS
# ------------------------------------------------------------

def mostrar_caracteristicas(dataframe):

    """
    Compara los promedios de las características
    académicas de los análisis filtrados.
    """

    st.subheader("4. Características académicas")

    columnas = [

        "horas_estudio",

        "asistencia",

        "actividades",

        "participacion"

    ]

    nombres = [

        "Horas de estudio",

        "Asistencia",

        "Actividades",

        "Participación"

    ]

    # Calcular los promedios.
    promedios = dataframe[columnas].mean()

    datos = pd.DataFrame({

        "caracteristica": nombres,

        "promedio": promedios.values

    })

    figura = px.bar(

        datos,

        x="caracteristica",

        y="promedio",

        text_auto=".2f",

        title="Promedio de características académicas",

        labels={

            "caracteristica": "Característica",

            "promedio": "Promedio"

        }

    )

    st.plotly_chart(

        figura,

        use_container_width=True

    )


# ------------------------------------------------------------
# 15. VISUALIZACION PCA
# ------------------------------------------------------------

def mostrar_pca(dataframe):

    """
    Representa los análisis utilizando los componentes
    principales PC1 y PC2 calculados en la Etapa 4.
    """

    st.subheader("5. Visualización de componentes PCA")

    datos = dataframe.dropna(

        subset=["PC1", "PC2"]

    )

    if datos.empty:

        st.info("No existen coordenadas PCA disponibles.")

        return

    figura = px.scatter(

        datos,

        x="PC1",

        y="PC2",

        color="perfil",

        hover_data=[

            "horas_estudio",

            "asistencia",

            "actividades",

            "participacion",

            "resultado"

        ],

        title="Proyección de análisis en dos componentes principales",

        labels={

            "PC1": "Componente principal 1",

            "PC2": "Componente principal 2",

            "perfil": "Perfil académico"

        }

    )

    st.plotly_chart(

        figura,

        use_container_width=True

    )

    st.caption(

        "La proyección representa el 51.72% de la varianza "
        "explicada en el conjunto original de cuatro variables. "
        "La visualización es exploratoria."

    )


# ------------------------------------------------------------
# 16. CONSULTA DEL HISTORIAL
# ------------------------------------------------------------

def mostrar_historial(dataframe):

    """
    Presenta los registros almacenados en MongoDB Atlas.
    """

    st.subheader("6. Historial de análisis")

    columnas = [

        "id_analisis",

        "fecha_registro",

        "horas_estudio",

        "asistencia",

        "actividades",

        "participacion",

        "resultado",

        "probabilidad_aprobar",

        "perfil"

    ]

    disponibles = [

        columna

        for columna in columnas

        if columna in dataframe.columns

    ]

    historial = dataframe[disponibles].copy()

    st.dataframe(

        historial,

        use_container_width=True,

        hide_index=True

    )

    # Preparar archivo CSV descargable.
    csv = historial.to_csv(

        index=False

    ).encode("utf-8-sig")

    st.download_button(

        label="Descargar historial CSV",

        data=csv,

        file_name="smartacademy_historial.csv",

        mime="text/csv"

    )


# ------------------------------------------------------------
# 17. APLICAR FILTROS
# ------------------------------------------------------------

def aplicar_filtros(dataframe):

    """
    Permite seleccionar perfiles y resultados
    para modificar las visualizaciones.
    """

    st.sidebar.header("Filtros de consulta")

    dataframe_filtrado = dataframe.copy()

    # --------------------------------------------------------
    # Filtrar por perfil académico.
    # --------------------------------------------------------

    perfiles = sorted(

        dataframe["perfil"]

        .dropna()

        .astype(str)

        .unique()

        .tolist()

    )

    perfiles_seleccionados = st.sidebar.multiselect(

        "Seleccionar perfiles",

        options=perfiles,

        default=perfiles

    )

    if perfiles:

        dataframe_filtrado = dataframe_filtrado[

            dataframe_filtrado["perfil"]

            .astype(str)

            .isin(perfiles_seleccionados)

        ]

    # --------------------------------------------------------
    # Filtrar por resultado estimado.
    # --------------------------------------------------------

    resultados = sorted(

        dataframe_filtrado["resultado"]

        .dropna()

        .astype(str)

        .unique()

        .tolist()

    )

    resultados_seleccionados = st.sidebar.multiselect(

        "Seleccionar resultados",

        options=resultados,

        default=resultados

    )

    if resultados:

        dataframe_filtrado = dataframe_filtrado[

            dataframe_filtrado["resultado"]

            .astype(str)

            .isin(resultados_seleccionados)

        ]

    return dataframe_filtrado


# ------------------------------------------------------------
# 18. APLICACION PRINCIPAL
# ------------------------------------------------------------

def main():

    """
    Controla la ejecución de todo el dashboard.
    """

    mostrar_encabezado()

    # --------------------------------------------------------
    # Actualización manual de los datos.
    # --------------------------------------------------------

    if st.sidebar.button("Actualizar información"):

        cargar_datos.clear()

        st.rerun()

    try:

        # Consultar MongoDB.
        dataframe = cargar_datos()

        # Validar si existen documentos.
        if dataframe.empty:

            st.warning(

                "Todavía no existen análisis almacenados "
                "en MongoDB Atlas."

            )

            st.info(

                "Ejecuta primero las prácticas de generación "
                "de datos y motor de recomendaciones."

            )

            return

        # Preparar columnas numéricas.
        dataframe = preparar_datos(dataframe)

        # Mostrar información de conexión.
        st.sidebar.success(

            "Conexión con MongoDB Atlas activa"

        )

        st.sidebar.write(

            "Base de datos:",

            MONGO_DB

        )

        st.sidebar.write(

            "Colección:",

            MONGO_COLLECTION

        )

        # Aplicar filtros.
        dataframe_filtrado = aplicar_filtros(

            dataframe

        )

        st.sidebar.write(

            "Análisis encontrados:",

            len(dataframe_filtrado)

        )

        # Validar si los filtros devolvieron datos.
        if dataframe_filtrado.empty:

            st.warning(

                "No existen análisis para los filtros seleccionados."

            )

            return

        # ----------------------------------------------------
        # Mostrar indicadores y visualizaciones.
        # ----------------------------------------------------

        mostrar_indicadores(dataframe_filtrado)

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            mostrar_predicciones(dataframe_filtrado)

        with col2:

            mostrar_perfiles(dataframe_filtrado)

        st.divider()

        mostrar_caracteristicas(dataframe_filtrado)

        st.divider()

        mostrar_pca(dataframe_filtrado)

        st.divider()

        mostrar_historial(dataframe_filtrado)

        # ----------------------------------------------------
        # Nota metodológica.
        # ----------------------------------------------------

        st.divider()

        st.info(

            "Las predicciones y probabilidades mostradas "
            "son estimaciones de modelos entrenados con "
            "datos sintéticos. No sustituyen la evaluación "
            "individual del docente."

        )

    except ValueError as error:

        st.error(

            f"Error de configuración: {error}"

        )

    except PyMongoError as error:

        st.error(

            "No fue posible consultar MongoDB Atlas."

        )

        st.code(str(error))

    except Exception as error:

        st.error(

            "Se produjo un error inesperado."

        )

        st.code(str(error))


# ------------------------------------------------------------
# 19. PUNTO DE ENTRADA
# ------------------------------------------------------------

if __name__ == "__main__":

    main()