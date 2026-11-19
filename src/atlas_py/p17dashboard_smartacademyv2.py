
# ============================================================
# PROYECTO: SMARTACADEMY
# ETAPA 10: DASHBOARD INTELIGENTE DE SEGUIMIENTO ACADEMICO
#
# Archivo:
# p17dashboard_smartacademy.py
#
# OBJETIVO:
# Construir un dashboard para analizar la información
# académica almacenada en MongoDB Atlas.
#
# FUNCIONALIDADES:
# 1. Indicadores generales.
# 2. Estudiantes únicos y evaluaciones.
# 3. Distribución de predicciones.
# 4. Distribución de perfiles K-Means.
# 5. Características académicas.
# 6. Visualización PCA.
# 7. Historial individual.
# 8. Evolución de las evaluaciones.
# 9. Exportación de reportes CSV.
#
# TECNOLOGÍAS:
# Python, Streamlit, Pandas, Plotly y MongoDB Atlas.
# ============================================================


# ============================================================
# 1. IMPORTAR LIBRERÍAS
# ============================================================

import os

import streamlit as st
import pandas as pd
import plotly.express as px

from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv

from pymongo import MongoClient
from pymongo.errors import PyMongoError


# ============================================================
# 2. CONFIGURAR LA PÁGINA
# ============================================================

st.set_page_config(
    page_title="SmartAcademy | Seguimiento",
    page_icon="icono",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 3. CONFIGURACIÓN DE RUTAS
# ============================================================

RUTA_ARCHIVO = Path(__file__).resolve()

RUTA_PROYECTO = RUTA_ARCHIVO.parents[1]

load_dotenv(RUTA_PROYECTO / ".env")


# ============================================================
# 4. VARIABLES DE ENTORNO
# ============================================================

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
# 5. VALIDAR CONFIGURACIÓN
# ============================================================

def validar_configuracion():

    """
    Comprueba que estén configuradas las variables
    necesarias para conectar con MongoDB Atlas.
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


# ============================================================
# 6. CONSTRUIR URI DE MONGODB
# ============================================================

def construir_uri():

    """
    Construye la URI de conexión.

    quote_plus permite codificar caracteres especiales
    en el usuario y contraseña.
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


# ============================================================
# 7. CONECTAR CON MONGODB ATLAS
# ============================================================

@st.cache_resource
def obtener_cliente():

    """
    Crea una conexión reutilizable.

    Streamlit conserva el cliente en caché para evitar
    crear una conexión nueva en cada actualización.
    """

    validar_configuracion()

    cliente = MongoClient(
        construir_uri(),
        serverSelectionTimeoutMS=15000
    )

    cliente.admin.command("ping")

    return cliente


# ============================================================
# 8. CONSULTAR INFORMACIÓN DE ATLAS
# ============================================================

@st.cache_data(ttl=60)
def cargar_datos():

    """
    Recupera todos los documentos de SmartAcademy.

    Se incluyen los documentos antiguos y nuevos.

    Los documentos antiguos pueden no tener
    identificador permanente de estudiante.
    """

    cliente = obtener_cliente()

    coleccion = cliente[MONGO_DB][MONGO_COLLECTION]

    documentos = list(
        coleccion.find(
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
    )

    registros = []

    for documento in documentos:

        datos = documento.get(
            "datos_reporte",
            {}
        )

        # Buscar primero el identificador raíz.
        # Si no existe, buscarlo dentro de datos_reporte.
        id_estudiante = documento.get(
            "id_estudiante"
        )

        if not id_estudiante:

            id_estudiante = datos.get(
                "id_estudiante"
            )

        registro = {

            "id_analisis":
                documento.get("id_analisis"),

            "id_estudiante":
                id_estudiante,

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

    return pd.DataFrame(registros)


# ============================================================
# 9. PREPARAR INFORMACIÓN
# ============================================================

def preparar_datos(dataframe):

    """
    Convierte los campos a formatos adecuados
    para los cálculos y visualizaciones.
    """

    dataframe = dataframe.copy()

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

        dataframe[columna] = pd.to_numeric(
            dataframe[columna],
            errors="coerce"
        )

    # Convertir fecha a formato datetime.
    dataframe["fecha_registro"] = pd.to_datetime(
        dataframe["fecha_registro"],
        errors="coerce",
        utc=True
    )

    # --------------------------------------------------------
    # NORMALIZAR PROBABILIDADES
    # --------------------------------------------------------

    # El motor p14 guarda las probabilidades como porcentajes.
    # Ejemplo: 98.90 representa 98.90%.
    #
    # Esta comprobación también permite trabajar con
    # documentos antiguos que eventualmente guarden
    # probabilidades entre 0 y 1.

    for columna in [
        "probabilidad_aprobar",
        "probabilidad_no_aprobar"
    ]:

        valores = dataframe[columna].dropna()

        if not valores.empty and valores.max() <= 1:

            dataframe[columna] = (
                dataframe[columna] * 100
            )

    # Identificadores como texto.
    dataframe["id_estudiante"] = (
        dataframe["id_estudiante"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Clasificar los registros sin identificador.
    dataframe["tipo_registro"] = (
        dataframe["id_estudiante"]
        .apply(
            lambda x:
            "Identificado" if x else "Histórico sin ID"
        )
    )

    return dataframe


# ============================================================
# 10. ENCABEZADO
# ============================================================

def mostrar_encabezado():

    st.title("SMARTACADEMY")

    st.subheader(
        "Dashboard inteligente de seguimiento académico"
    )

    st.markdown(
        """
        Plataforma de análisis académico que integra
        aprendizaje supervisado, aprendizaje no supervisado,
        reducción de dimensionalidad y persistencia de datos.
        """
    )

    st.divider()


# ============================================================
# 11. INDICADORES GENERALES
# ============================================================

def mostrar_indicadores(dataframe):

    """
    Calcula los principales indicadores del conjunto
    de evaluaciones seleccionado.
    """

    st.subheader("1. Indicadores generales")

    total_evaluaciones = len(dataframe)

    # Contabilizar exclusivamente identificadores conocidos.
    estudiantes_unicos = (
        dataframe.loc[
            dataframe["id_estudiante"] != "",
            "id_estudiante"
        ].nunique()
    )

    riesgos = int(
        (dataframe["prediccion"] == 0).sum()
    )

    aprobaciones = int(
        (dataframe["prediccion"] == 1).sum()
    )

    if total_evaluaciones > 0:

        porcentaje_riesgo = (
            riesgos / total_evaluaciones * 100
        )

        promedio_aprobar = (
            dataframe["probabilidad_aprobar"].mean()
        )

        promedio_asistencia = (
            dataframe["asistencia"].mean()
        )

    else:

        porcentaje_riesgo = 0

        promedio_aprobar = 0

        promedio_asistencia = 0

    # Primera fila de indicadores.
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Estudiantes únicos identificados",
        estudiantes_unicos
    )

    col2.metric(
        "Total de evaluaciones",
        total_evaluaciones
    )

    col3.metric(
        "Evaluaciones con riesgo estimado",
        riesgos,
        f"{porcentaje_riesgo:.2f}%"
    )

    # Segunda fila de indicadores.
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Aprobaciones estimadas",
        aprobaciones
    )

    col2.metric(
        "Probabilidad promedio de aprobación",
        f"{promedio_aprobar:.2f}%"
    )

    col3.metric(
        "Asistencia promedio",
        f"{promedio_asistencia:.2f}%"
    )

    # Advertencia metodológica.
    sin_identificador = int(
        (dataframe["id_estudiante"] == "").sum()
    )

    if sin_identificador > 0:

        st.caption(
            f"{sin_identificador} evaluaciones históricas "
            "no tienen identificador permanente y no se "
            "incluyen en el conteo de estudiantes únicos."
        )


# ============================================================
# 12. DISTRIBUCIÓN DE PREDICCIONES
# ============================================================

def mostrar_predicciones(dataframe):

    st.subheader("2. Distribución de predicciones")

    datos = dataframe.dropna(
        subset=["prediccion"]
    ).copy()

    if datos.empty:

        st.info(
            "No existen predicciones disponibles."
        )

        return

    datos["clasificacion"] = datos["prediccion"].map({

        0: "Riesgo estimado",

        1: "Aprobación estimada"

    }).fillna("Sin clasificar")

    conteo = (
        datos.groupby("clasificacion")
        .size()
        .reset_index(name="cantidad")
    )

    figura = px.pie(
        conteo,
        names="clasificacion",
        values="cantidad",
        hole=0.45,
        title="Distribución de resultados estimados"
    )

    figura.update_traces(
        textposition="inside",
        textinfo="label+percent+value"
    )

    st.plotly_chart(
        figura,
        use_container_width=True
    )


# ============================================================
# 13. DISTRIBUCIÓN DE PERFILES K-MEANS
# ============================================================

def mostrar_perfiles(dataframe):

    st.subheader("3. Distribución de perfiles académicos")

    datos = dataframe.dropna(
        subset=["perfil"]
    )

    if datos.empty:

        st.info(
            "No existen perfiles disponibles."
        )

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
        title="Cantidad de evaluaciones por perfil",
        labels={
            "perfil": "Perfil académico",
            "cantidad": "Cantidad de evaluaciones"
        }
    )

    figura.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        figura,
        use_container_width=True
    )


# ============================================================
# 14. CARACTERÍSTICAS ACADÉMICAS
# ============================================================

def mostrar_caracteristicas(dataframe):

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


# ============================================================
# 15. VISUALIZACIÓN PCA
# ============================================================

def mostrar_pca(dataframe):

    st.subheader("5. Visualización PCA")

    datos = dataframe.dropna(
        subset=["PC1", "PC2"]
    )

    if datos.empty:

        st.info(
            "No existen coordenadas PCA disponibles."
        )

        return

    figura = px.scatter(
        datos,
        x="PC1",
        y="PC2",
        color="perfil",
        hover_data=[
            "id_estudiante",
            "horas_estudio",
            "asistencia",
            "actividades",
            "participacion",
            "resultado"
        ],
        title="Proyección de evaluaciones en dos componentes",
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
        "Su interpretación es exploratoria."
    )


# ============================================================
# 16. FILTROS GENERALES
# ============================================================

def aplicar_filtros(dataframe):

    """
    Permite filtrar por estudiante, perfil, resultado
    y periodo de registro.
    """

    st.sidebar.header("Filtros de consulta")

    datos = dataframe.copy()

    # --------------------------------------------------------
    # FILTRO DE IDENTIFICADOR
    # --------------------------------------------------------

    identificadores = sorted(
        datos.loc[
            datos["id_estudiante"] != "",
            "id_estudiante"
        ].unique().tolist()
    )

    opciones_estudiante = [
        "Todos los estudiantes"
    ] + identificadores

    estudiante_seleccionado = st.sidebar.selectbox(
        "Estudiante",
        options=opciones_estudiante
    )

    if estudiante_seleccionado != "Todos los estudiantes":

        datos = datos[
            datos["id_estudiante"] == estudiante_seleccionado
        ]

    # --------------------------------------------------------
    # FILTRO DE PERFIL
    # --------------------------------------------------------

    perfiles = sorted(
        datos["perfil"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    perfiles_seleccionados = st.sidebar.multiselect(
        "Perfiles académicos",
        options=perfiles,
        default=perfiles
    )

    if perfiles:

        datos = datos[
            datos["perfil"]
            .astype(str)
            .isin(perfiles_seleccionados)
        ]

    # --------------------------------------------------------
    # FILTRO DE RESULTADO
    # --------------------------------------------------------

    resultados = sorted(
        datos["resultado"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    resultados_seleccionados = st.sidebar.multiselect(
        "Resultados estimados",
        options=resultados,
        default=resultados
    )

    if resultados:

        datos = datos[
            datos["resultado"]
            .astype(str)
            .isin(resultados_seleccionados)
        ]

    # --------------------------------------------------------
    # FILTRO DE FECHAS
    # --------------------------------------------------------

    fechas = datos["fecha_registro"].dropna()

    if not fechas.empty:

        fecha_minima = fechas.min().date()

        fecha_maxima = fechas.max().date()

        rango_fechas = st.sidebar.date_input(
            "Periodo de consulta",
            value=(fecha_minima, fecha_maxima),
            min_value=fecha_minima,
            max_value=fecha_maxima
        )

        if (
            isinstance(rango_fechas, (tuple, list))
            and len(rango_fechas) == 2
        ):

            inicio, fin = rango_fechas

            fechas_locales = datos[
                "fecha_registro"
            ].dt.date

            datos = datos[
                (fechas_locales >= inicio)
                & (fechas_locales <= fin)
            ]

    return datos, estudiante_seleccionado


# ============================================================
# 17. HISTORIAL GENERAL
# ============================================================

def mostrar_historial(dataframe):

    st.subheader("6. Historial de evaluaciones")

    columnas = [

        "id_estudiante",
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

    historial = historial.sort_values(
        "fecha_registro",
        ascending=False
    )

    st.dataframe(
        historial,
        use_container_width=True,
        hide_index=True
    )

    csv = historial.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        label="Descargar historial CSV",
        data=csv,
        file_name="smartacademy_historial.csv",
        mime="text/csv"
    )


# ============================================================
# 18. SEGUIMIENTO INDIVIDUAL
# ============================================================

def mostrar_seguimiento_individual(
    dataframe,
    estudiante_seleccionado
):

    """
    Analiza la evolución de un estudiante específico.
    """

    st.subheader("7. Seguimiento individual")

    if estudiante_seleccionado == "Todos los estudiantes":

        st.info(
            "Selecciona un estudiante en los filtros "
            "de la barra lateral para consultar su evolución."
        )

        return

    datos = dataframe[
        dataframe["id_estudiante"] == estudiante_seleccionado
    ].copy()

    if datos.empty:

        st.warning(
            "No existen evaluaciones para el estudiante seleccionado."
        )

        return

    datos = datos.sort_values(
        "fecha_registro"
    )

    st.write(
        "**Identificador:**",
        estudiante_seleccionado
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Evaluaciones realizadas",
        len(datos)
    )

    col2.metric(
        "Probabilidad más reciente",
        f"{datos.iloc[-1]['probabilidad_aprobar']:.2f}%"
        if pd.notna(datos.iloc[-1]["probabilidad_aprobar"])
        else "Sin dato"
    )

    col3.metric(
        "Perfil más reciente",
        str(datos.iloc[-1]["perfil"])
        if pd.notna(datos.iloc[-1]["perfil"])
        else "Sin perfil"
    )

    # --------------------------------------------------------
    # EVOLUCIÓN DE PROBABILIDAD
    # --------------------------------------------------------

    st.markdown("**Evolución de probabilidad estimada de aprobación**")

    datos_grafica = datos.dropna(
        subset=["probabilidad_aprobar", "fecha_registro"]
    )

    if not datos_grafica.empty:

        figura = px.line(
            datos_grafica,
            x="fecha_registro",
            y="probabilidad_aprobar",
            markers=True,
            title="Evolución de la probabilidad estimada",
            labels={
                "fecha_registro": "Fecha de evaluación",
                "probabilidad_aprobar": "Probabilidad de aprobación (%)"
            }
        )

        figura.update_yaxes(
            range=[0, 100]
        )

        st.plotly_chart(
            figura,
            use_container_width=True
        )

    # --------------------------------------------------------
    # EVOLUCIÓN DE CARACTERÍSTICAS
    # --------------------------------------------------------

    st.markdown("**Evolución de características académicas**")

    columnas = [
        "horas_estudio",
        "asistencia",
        "actividades",
        "participacion"
    ]

    nombres = {
        "horas_estudio": "Horas de estudio",
        "asistencia": "Asistencia",
        "actividades": "Actividades",
        "participacion": "Participación"
    }

    datos_caracteristicas = datos[
        ["fecha_registro"] + columnas
    ].melt(
        id_vars="fecha_registro",
        var_name="caracteristica",
        value_name="valor"
    )

    datos_caracteristicas["caracteristica"] = (
        datos_caracteristicas["caracteristica"].map(nombres)
    )

    figura = px.line(
        datos_caracteristicas,
        x="fecha_registro",
        y="valor",
        color="caracteristica",
        markers=True,
        title="Cambios en las características académicas",
        labels={
            "fecha_registro": "Fecha",
            "valor": "Valor",
            "caracteristica": "Variable"
        }
    )

    st.plotly_chart(
        figura,
        use_container_width=True
    )

    # --------------------------------------------------------
    # RECOMENDACIONES MÁS RECIENTES
    # --------------------------------------------------------

    st.markdown("**Recomendaciones de la evaluación más reciente**")

    recomendaciones = datos.iloc[-1]["recomendaciones"]

    if isinstance(recomendaciones, list):

        for numero, recomendacion in enumerate(
            recomendaciones,
            start=1
        ):

            st.write(
                f"{numero}. {recomendacion}"
            )

    else:

        st.info(
            "La evaluación más reciente no contiene recomendaciones."
        )

    # Mostrar tabla individual.
    st.markdown("**Historial del estudiante**")

    mostrar_historial(datos)


# ============================================================
# 19. INFORMACIÓN DE CONEXIÓN
# ============================================================

def mostrar_conexion():

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


# ============================================================
# 20. APLICACIÓN PRINCIPAL
# ============================================================

def main():

    """
    Coordina la ejecución del dashboard.
    """

    mostrar_encabezado()

    # --------------------------------------------------------
    # ACTUALIZAR INFORMACIÓN
    # --------------------------------------------------------

    if st.sidebar.button(
        "Actualizar información"
    ):

        cargar_datos.clear()

        st.rerun()

    try:

        # Consultar información.
        dataframe = cargar_datos()

        if dataframe.empty:

            st.warning(
                "Todavía no existen análisis en MongoDB Atlas."
            )

            return

        # Preparar información.
        dataframe = preparar_datos(dataframe)

        mostrar_conexion()

        # ----------------------------------------------------
        # APLICAR FILTROS
        # ----------------------------------------------------

        dataframe_filtrado, estudiante_seleccionado = (
            aplicar_filtros(dataframe)
        )

        st.sidebar.write(
            "Evaluaciones filtradas:",
            len(dataframe_filtrado)
        )

        if dataframe_filtrado.empty:

            st.warning(
                "No existen registros para los filtros seleccionados."
            )

            return

        # ----------------------------------------------------
        # SECCIÓN 1: INDICADORES
        # ----------------------------------------------------

        mostrar_indicadores(
            dataframe_filtrado
        )

        st.divider()

        # ----------------------------------------------------
        # SECCIÓN 2 Y 3: PREDICCIONES Y PERFILES
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            mostrar_predicciones(
                dataframe_filtrado
            )

        with col2:

            mostrar_perfiles(
                dataframe_filtrado
            )

        st.divider()

        # ----------------------------------------------------
        # SECCIÓN 4: CARACTERÍSTICAS
        # ----------------------------------------------------

        mostrar_caracteristicas(
            dataframe_filtrado
        )

        st.divider()

        # ----------------------------------------------------
        # SECCIÓN 5: PCA
        # ----------------------------------------------------

        mostrar_pca(
            dataframe_filtrado
        )

        st.divider()

        # ----------------------------------------------------
        # SECCIÓN 6: HISTORIAL GENERAL
        # ----------------------------------------------------

        mostrar_historial(
            dataframe_filtrado
        )

        st.divider()

        # ----------------------------------------------------
        # SECCIÓN 7: SEGUIMIENTO INDIVIDUAL
        # ----------------------------------------------------

        mostrar_seguimiento_individual(
            dataframe,
            estudiante_seleccionado
        )

        # ----------------------------------------------------
        # NOTA METODOLÓGICA
        # ----------------------------------------------------

        st.divider()

        st.info(
            """
            Las predicciones y probabilidades son estimaciones
            de modelos entrenados con datos sintéticos.
            No representan calificaciones definitivas ni sustituyen
            la evaluación y el acompañamiento del docente.

            Las evaluaciones históricas sin identificador de
            estudiante se conservan para el análisis general,
            pero no se consideran estudiantes únicos ni pueden
            formar parte de un historial individual.
            """
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


# ============================================================
# 21. PUNTO DE ENTRADA
# ============================================================

if __name__ == "__main__":

    main()
