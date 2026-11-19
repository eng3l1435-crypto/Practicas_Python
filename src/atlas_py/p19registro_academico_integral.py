
# ============================================================
# PROYECTO: SMARTACADEMY
# ETAPA 11 - REGISTRO ACADEMICO INTEGRAL
#
# ARCHIVO:
# p19registro_academico.py
#
# OBJETIVO:
# Registrar expedientes, materias, calificaciones,
# asistencia y resultados de inteligencia artificial.
#
# COLECCIONES:
# 1. estudiantes
# 2. evaluaciones_academicas
# 3. analisis_estudiantes
#
# MODELOS REUTILIZADOS:
# Random Forest, K-Means y PCA.
#
# IMPORTANTE:
# No se entrenan modelos nuevos.
# ============================================================


# ============================================================
# 1. IMPORTAR LIBRERIAS
# ============================================================

import os
import sys
import uuid

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
# 2. CONFIGURAR RUTAS DEL PROYECTO
# ============================================================

RUTA_ARCHIVO = Path(__file__).resolve()

RUTA_PROYECTO = RUTA_ARCHIVO.parents[1]

# Agregar la raíz del proyecto a sys.path.
# Esto permite importar el motor p14.
if str(RUTA_PROYECTO) not in sys.path:

    sys.path.insert(
        0,
        str(RUTA_PROYECTO)
    )


# ============================================================
# 3. IMPORTAR MOTOR DE INTELIGENCIA ARTIFICIAL
# ============================================================

from unidad3_fia import p14motor_recomendaciones as motor


# ============================================================
# 4. CARGAR VARIABLES DE ENTORNO
# ============================================================

load_dotenv(
    RUTA_PROYECTO / ".env"
)

MONGO_USER = os.getenv("MONGO_USER")

MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")

MONGO_CLUSTER = os.getenv("MONGO_CLUSTER")

MONGO_DB = os.getenv(
    "MONGO_DB",
    "smartacademy_db"
)

MONGO_COLLECTION_IA = os.getenv(
    "MONGO_COLLECTION",
    "analisis_estudiantes"
)

COLECCION_ESTUDIANTES = "estudiantes"

COLECCION_EVALUACIONES = "evaluaciones_academicas"


# ============================================================
# 5. CONFIGURAR STREAMLIT
# ============================================================

st.set_page_config(
    page_title="SmartAcademy - Registro Académico",
    page_icon="icono",
    layout="wide"
)

st.title("SMARTACADEMY")

st.subheader(
    "Etapa 11 - Registro académico integral"
)

st.markdown(
    """
    Sistema de registro académico que integra:

    - Expediente permanente del estudiante.
    - Materias y docentes.
    - Tres evaluaciones parciales.
    - Calificación final.
    - Control de asistencia.
    - Análisis mediante inteligencia artificial.
    """
)

st.divider()


# ============================================================
# 6. CARGAR MODELOS EXISTENTES
# ============================================================

@st.cache_resource
def cargar_modelos():

    """
    Recupera los modelos entrenados anteriormente.

    No se realiza un nuevo entrenamiento.
    """

    return motor.cargar_modelos()


try:

    modelos = cargar_modelos()

    st.success(
        "Modelos de inteligencia artificial cargados."
    )

except Exception as error:

    st.error(
        "No fue posible cargar los modelos."
    )

    st.exception(error)

    st.stop()


# ============================================================
# 7. CONSTRUIR URI DE MONGODB
# ============================================================

def construir_uri():

    """
    Construye la cadena de conexión a MongoDB Atlas.
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

    return (
        f"mongodb+srv://{usuario}:{contrasena}"
        f"@{MONGO_CLUSTER}/"
        "?retryWrites=true&w=majority"
        "&appName=SmartAcademy"
    )


# ============================================================
# 8. CONECTAR CON MONGODB ATLAS
# ============================================================

def conectar_mongodb():

    """
    Establece conexión con Atlas y prepara las colecciones.
    """

    cliente = MongoClient(
        construir_uri(),
        tlsCAFile=certifi.where(),
        serverSelectionTimeoutMS=15000
    )

    cliente.admin.command("ping")

    base_datos = cliente[MONGO_DB]

    coleccion_estudiantes = base_datos[
        COLECCION_ESTUDIANTES
    ]

    coleccion_evaluaciones = base_datos[
        COLECCION_EVALUACIONES
    ]

    coleccion_ia = base_datos[
        MONGO_COLLECTION_IA
    ]

    # El identificador del estudiante es único.
    coleccion_estudiantes.create_index(
        "id_estudiante",
        unique=True
    )

    # Cada evaluación académica tiene un identificador único.
    coleccion_evaluaciones.create_index(
        "id_evaluacion",
        unique=True
    )

    # Consultas por estudiante y fecha.
    coleccion_evaluaciones.create_index([
        ("id_estudiante", 1),
        ("fecha_registro", -1)
    ])

    # Cada análisis de IA debe tener un identificador único.
    coleccion_ia.create_index(
        "id_analisis",
        unique=True
    )

    return (
        cliente,
        coleccion_estudiantes,
        coleccion_evaluaciones,
        coleccion_ia
    )


# ============================================================
# 9. FUNCION PARA VALIDAR CALIFICACIONES
# ============================================================

def validar_calificacion(valor):

    """
    Comprueba que una calificación esté entre 0 y 10.
    """

    if pd.isna(valor):

        return False

    return 0 <= float(valor) <= 10


# ============================================================
# 10. CALCULAR CALIFICACION FINAL
# ============================================================

def calcular_calificacion_final(
    parcial1,
    parcial2,
    parcial3
):

    """
    Calcula el promedio aritmético de tres parciales.
    """

    return round(
        (
            float(parcial1)
            + float(parcial2)
            + float(parcial3)
        ) / 3,
        2
    )


# ============================================================
# 11. PREPARAR MATERIAS Y CALIFICACIONES
# ============================================================

def procesar_materias(tabla):

    """
    Valida los datos capturados en la tabla de materias.

    Calcula:
    - Calificación final.
    - Total de asistencias.
    - Total de faltas.
    - Porcentaje de asistencia.
    """

    materias = tabla.copy()

    # Eliminar filas sin nombre de materia.
    materias["materia"] = (
        materias["materia"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    materias = materias[
        materias["materia"] != ""
    ].copy()

    if materias.empty:

        raise ValueError(
            "Debes registrar al menos una materia."
        )

    # Convertir campos numéricos.
    columnas_numericas = [
        "parcial1",
        "parcial2",
        "parcial3",
        "clases_programadas",
        "asistencias"
    ]

    for columna in columnas_numericas:

        materias[columna] = pd.to_numeric(
            materias[columna],
            errors="coerce"
        )

    if materias[columnas_numericas].isna().any().any():

        raise ValueError(
            "Completa todas las calificaciones y los datos de asistencia."
        )

    # Validar calificaciones.
    for columna in [
        "parcial1",
        "parcial2",
        "parcial3"
    ]:

        if not materias[columna].apply(
            validar_calificacion
        ).all():

            raise ValueError(
                f"La columna {columna} debe contener valores entre 0 y 10."
            )

    # Validar clases programadas y asistencias.
    if (
        materias["clases_programadas"] < 1
    ).any():

        raise ValueError(
            "Las clases programadas deben ser mayores que cero."
        )

    if (
        materias["asistencias"] < 0
    ).any():

        raise ValueError(
            "Las asistencias no pueden ser negativas."
        )

    if (
        materias["asistencias"]
        > materias["clases_programadas"]
    ).any():

        raise ValueError(
            "Las asistencias no pueden superar las clases programadas."
        )

    # Calcular calificación final por materia.
    materias["calificacion_final"] = materias.apply(
        lambda fila: calcular_calificacion_final(
            fila["parcial1"],
            fila["parcial2"],
            fila["parcial3"]
        ),
        axis=1
    )

    # Calcular faltas.
    materias["faltas"] = (
        materias["clases_programadas"]
        - materias["asistencias"]
    )

    # Calcular porcentaje de asistencia.
    materias["porcentaje_asistencia"] = (
        materias["asistencias"]
        / materias["clases_programadas"]
        * 100
    ).round(2)

    return materias


# ============================================================
# 12. CREAR DOCUMENTO DE MATERIA
# ============================================================

def convertir_materias_documentos(materias):

    """
    Convierte las filas del DataFrame en documentos
    que MongoDB puede almacenar.
    """

    documentos = []

    for _, fila in materias.iterrows():

        documento = {

            "materia": str(fila["materia"]),

            "docente": str(fila["docente"]),

            "calificaciones": {

                "parcial1": float(fila["parcial1"]),

                "parcial2": float(fila["parcial2"]),

                "parcial3": float(fila["parcial3"]),

                "final": float(
                    fila["calificacion_final"]
                )

            },

            "asistencia": {

                "clases_programadas": int(
                    fila["clases_programadas"]
                ),

                "asistencias": int(
                    fila["asistencias"]
                ),

                "faltas": int(
                    fila["faltas"]
                ),

                "porcentaje": float(
                    fila["porcentaje_asistencia"]
                )

            }

        }

        documentos.append(documento)

    return documentos


# ============================================================
# 13. REGISTRAR O ACTUALIZAR EXPEDIENTE
# ============================================================

def guardar_estudiante(
    coleccion,
    id_estudiante,
    nombre,
    carrera,
    grupo,
    cuatrimestre,
    ciclo_escolar
):

    """
    Si el estudiante existe, actualiza sus datos generales.

    Si no existe, crea un expediente nuevo.

    No elimina ni sobrescribe evaluaciones anteriores.
    """

    ahora = datetime.now(timezone.utc)

    coleccion.update_one(

        {
            "id_estudiante": id_estudiante
        },

        {
            "$set": {

                "nombre_completo": nombre,

                "carrera": carrera,

                "grupo": grupo,

                "cuatrimestre": cuatrimestre,

                "ciclo_escolar": ciclo_escolar,

                "fecha_actualizacion": ahora

            },

            "$setOnInsert": {

                "id_estudiante": id_estudiante,

                "fecha_creacion": ahora

            }

        },

        upsert=True

    )


# ============================================================
# 14. CAPTURAR INFORMACION GENERAL
# ============================================================

st.header("1. Expediente del estudiante")

with st.form(
    "formulario_academico",
    clear_on_submit=False
):

    col1, col2 = st.columns(2)

    with col1:

        id_estudiante = st.text_input(
            "Identificador permanente",
            placeholder="Ejemplo: IDGS-2026-001"
        )

        nombre = st.text_input(
            "Nombre completo"
        )

        carrera = st.text_input(
            "Carrera",
            placeholder="Ingeniería en Desarrollo y Gestión de Software"
        )

    with col2:

        grupo = st.text_input(
            "Grupo",
            placeholder="Ejemplo: IDGS-51"
        )

        cuatrimestre = st.selectbox(
            "Cuatrimestre",
            options=[
                "Primero",
                "Segundo",
                "Tercero",
                "Cuarto",
                "Quinto",
                "Sexto",
                "Séptimo",
                "Octavo",
                "Noveno",
                "Décimo",
                "Undécimo"
            ]
        )

        ciclo_escolar = st.text_input(
            "Ciclo escolar",
            placeholder="Ejemplo: 2026-2027"
        )

    st.divider()

    st.header("2. Información de las materias")

    st.write(
        "Agrega una fila por cada materia del estudiante. "
        "Puedes añadir filas desde la tabla."
    )

    tabla_inicial = pd.DataFrame({

        "materia": [""],

        "docente": [""],

        "parcial1": [0.0],

        "parcial2": [0.0],

        "parcial3": [0.0],

        "clases_programadas": [1],

        "asistencias": [1]

    })

    tabla_materias = st.data_editor(

        tabla_inicial,

        num_rows="dynamic",

        use_container_width=True,

        hide_index=True,

        column_config={

            "materia": st.column_config.TextColumn(
                "Materia",
                required=True
            ),

            "docente": st.column_config.TextColumn(
                "Docente"
            ),

            "parcial1": st.column_config.NumberColumn(
                "Parcial 1",
                min_value=0.0,
                max_value=10.0,
                step=0.1,
                format="%.1f"
            ),

            "parcial2": st.column_config.NumberColumn(
                "Parcial 2",
                min_value=0.0,
                max_value=10.0,
                step=0.1,
                format="%.1f"
            ),

            "parcial3": st.column_config.NumberColumn(
                "Parcial 3",
                min_value=0.0,
                max_value=10.0,
                step=0.1,
                format="%.1f"
            ),

            "clases_programadas": st.column_config.NumberColumn(
                "Clases programadas",
                min_value=1,
                step=1
            ),

            "asistencias": st.column_config.NumberColumn(
                "Asistencias",
                min_value=0,
                step=1
            )

        }

    )

    st.divider()

    st.header("3. Variables para inteligencia artificial")

    st.caption(
        "Estas variables mantienen el formato de entrada "
        "de los modelos entrenados en las etapas anteriores."
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

        actividades = st.number_input(
            "Actividades realizadas",
            min_value=0,
            max_value=30,
            value=15,
            step=1
        )

    with col2:

        participacion = st.number_input(
            "Porcentaje de participación",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=1.0
        )

    enviar = st.form_submit_button(
        "Guardar expediente y evaluar estudiante",
        use_container_width=True
    )


# ============================================================
# 15. PROCESAR EL REGISTRO
# ============================================================

if enviar:

    # --------------------------------------------------------
    # VALIDAR DATOS GENERALES
    # --------------------------------------------------------

    if not id_estudiante.strip():

        st.error(
            "El identificador del estudiante es obligatorio."
        )

        st.stop()

    if not nombre.strip():

        st.error(
            "El nombre del estudiante es obligatorio."
        )

        st.stop()

    if not carrera.strip():

        st.error(
            "La carrera es obligatoria."
        )

        st.stop()

    if not grupo.strip():

        st.error(
            "El grupo es obligatorio."
        )

        st.stop()

    if not ciclo_escolar.strip():

        st.error(
            "El ciclo escolar es obligatorio."
        )

        st.stop()

    # --------------------------------------------------------
    # PROCESAR MATERIAS
    # --------------------------------------------------------

    try:

        materias_procesadas = procesar_materias(
            tabla_materias
        )

        documentos_materias = convertir_materias_documentos(
            materias_procesadas
        )

    except ValueError as error:

        st.error(str(error))

        st.stop()

    # --------------------------------------------------------
    # CALCULAR INDICADORES GENERALES
    # --------------------------------------------------------

    promedio_general = round(
        materias_procesadas[
            "calificacion_final"
        ].mean(),
        2
    )

    total_clases = int(
        materias_procesadas[
            "clases_programadas"
        ].sum()
    )

    total_asistencias = int(
        materias_procesadas[
            "asistencias"
        ].sum()
    )

    total_faltas = int(
        materias_procesadas[
            "faltas"
        ].sum()
    )

    asistencia_general = round(
        total_asistencias / total_clases * 100,
        2
    )

    # --------------------------------------------------------
    # PREPARAR DATOS PARA LOS MODELOS
    # --------------------------------------------------------

    estudiante = pd.DataFrame(

        [[
            float(horas_estudio),
            float(asistencia_general),
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

    # --------------------------------------------------------
    # EJECUTAR MODELOS
    # --------------------------------------------------------

    try:

        resultado_prediccion = motor.predecir_aprobacion(
            estudiante,
            modelos["supervisado"]
        )

        resultado_perfil = motor.identificar_perfil(
            estudiante,
            modelos["kmeans"],
            modelos["escalador_kmeans"],
            modelos["mapa_perfiles"]
        )

        resultado_pca = motor.proyectar_pca(
            estudiante,
            modelos["pca"],
            modelos["escalador_pca"]
        )

        recomendaciones = motor.generar_recomendaciones(
            estudiante,
            resultado_prediccion,
            resultado_perfil
        )

    except Exception as error:

        st.error(
            "No fue posible ejecutar los modelos."
        )

        st.exception(error)

        st.stop()

    # --------------------------------------------------------
    # MOSTRAR RESULTADOS ACADÉMICOS
    # --------------------------------------------------------

    st.divider()

    st.header("4. Resultados académicos")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Promedio general",
        f"{promedio_general:.2f}/10"
    )

    col2.metric(
        "Asistencia general",
        f"{asistencia_general:.2f}%"
    )

    col3.metric(
        "Total de faltas",
        total_faltas
    )

    st.dataframe(
        materias_procesadas,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # MOSTRAR RESULTADOS DE INTELIGENCIA ARTIFICIAL
    # --------------------------------------------------------

    st.header("5. Análisis de inteligencia artificial")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Probabilidad de aprobar",
        f"{resultado_prediccion['probabilidad_aprobar']:.2f}%"
    )

    col2.metric(
        "Probabilidad de no aprobar",
        f"{resultado_prediccion['probabilidad_no_aprobar']:.2f}%"
    )

    col3.metric(
        "Perfil K-Means",
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

    st.write(
        "**Componentes PCA:**",
        f"PC1 = {resultado_pca['PC1']:.4f}",
        "|",
        f"PC2 = {resultado_pca['PC2']:.4f}"
    )

    st.subheader("Recomendaciones")

    for numero, recomendacion in enumerate(
        recomendaciones,
        start=1
    ):

        st.write(
            f"{numero}. {recomendacion}"
        )

    # --------------------------------------------------------
    # GENERAR IDENTIFICADORES
    # --------------------------------------------------------

    id_evaluacion = str(uuid.uuid4())

    id_analisis = str(uuid.uuid4())

    fecha_actual = datetime.now(timezone.utc)

    # --------------------------------------------------------
    # CONSTRUIR DOCUMENTO DE EVALUACION ACADEMICA
    # --------------------------------------------------------

    documento_academico = {

        "id_evaluacion": id_evaluacion,

        "id_estudiante": id_estudiante.strip(),

        "sistema": "SmartAcademy",

        "fecha_registro": fecha_actual,

        "ciclo_escolar": ciclo_escolar.strip(),

        "carrera": carrera.strip(),

        "grupo": grupo.strip(),

        "cuatrimestre": cuatrimestre,

        "materias": documentos_materias,

        "resumen_academico": {

            "total_materias": len(documentos_materias),

            "promedio_general": promedio_general,

            "clases_programadas": total_clases,

            "asistencias": total_asistencias,

            "faltas": total_faltas,

            "porcentaje_asistencia": asistencia_general

        },

        "variables_aprendizaje": {

            "horas_estudio": float(horas_estudio),

            "actividades": int(actividades),

            "participacion": float(participacion)

        },

        "id_analisis": id_analisis

    }

    # --------------------------------------------------------
    # CONSTRUIR DOCUMENTO DE INTELIGENCIA ARTIFICIAL
    # --------------------------------------------------------

    documento_ia = {

        "id_analisis": id_analisis,

        "id_evaluacion_academica": id_evaluacion,

        "id_estudiante": id_estudiante.strip(),

        "sistema": "SmartAcademy",

        "etapa": 11,

        "fecha_registro": fecha_actual,

        "archivo_origen": "p19registro_academico.py",

        "datos_reporte": {

            "id_estudiante": id_estudiante.strip(),

            "id_evaluacion_academica": id_evaluacion,

            "horas_estudio": float(horas_estudio),

            "asistencia": float(asistencia_general),

            "actividades": int(actividades),

            "participacion": float(participacion),

            "promedio_general": promedio_general,

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

            "recomendaciones": recomendaciones

        }

    }

    # --------------------------------------------------------
    # GUARDAR INFORMACION EN ATLAS
    # --------------------------------------------------------

    cliente = None

    try:

        with st.spinner(
            "Guardando expediente y evaluación en MongoDB Atlas..."
        ):

            (
                cliente,
                coleccion_estudiantes,
                coleccion_evaluaciones,
                coleccion_ia
            ) = conectar_mongodb()

            # 1. Registrar o actualizar expediente.
            guardar_estudiante(
                coleccion_estudiantes,
                id_estudiante.strip(),
                nombre.strip(),
                carrera.strip(),
                grupo.strip(),
                cuatrimestre,
                ciclo_escolar.strip()
            )

            # 2. Insertar evaluación académica.
            coleccion_evaluaciones.insert_one(
                documento_academico
            )

            # 3. Insertar análisis de IA.
            coleccion_ia.insert_one(
                documento_ia
            )

        st.success(
            "El expediente, la evaluación académica "
            "y el análisis de IA se guardaron correctamente."
        )

        st.write(
            "**ID del estudiante:**",
            id_estudiante.strip()
        )

        st.write(
            "**ID de evaluación académica:**",
            id_evaluacion
        )

        st.write(
            "**ID de análisis de IA:**",
            id_analisis
        )

        st.info(
            "Los registros anteriores del estudiante "
            "se conservaron. Se creó una nueva evaluación."
        )

    except PyMongoError as error:

        st.error(
            "Ocurrió un problema al guardar información en Atlas."
        )

        st.exception(error)

        st.warning(
            "Si hubo un error durante el guardado, verifica las "
            "colecciones antes de repetir la operación. "
            "Las escrituras en varias colecciones no forman "
            "automáticamente una transacción."
        )

    except Exception as error:

        st.error(
            "Ocurrió un error inesperado."
        )

        st.exception(error)

    finally:

        if cliente is not None:

            cliente.close()


# ============================================================
# 16. CONSULTAR EXPEDIENTE E HISTORIAL
# ============================================================

st.divider()

st.header("6. Consulta del expediente académico")

id_consulta = st.text_input(
    "Identificador del estudiante",
    key="consulta_expediente",
    placeholder="Ejemplo: IDGS-2026-001"
)

if st.button(
    "Consultar expediente",
    use_container_width=True
):

    if not id_consulta.strip():

        st.warning(
            "Escribe el identificador del estudiante."
        )

    else:

        cliente = None

        try:

            (
                cliente,
                coleccion_estudiantes,
                coleccion_evaluaciones,
                coleccion_ia
            ) = conectar_mongodb()

            identificador = id_consulta.strip()

            expediente = coleccion_estudiantes.find_one(
                {
                    "id_estudiante": identificador
                },
                {
                    "_id": 0
                }
            )

            evaluaciones = list(
                coleccion_evaluaciones.find(
                    {
                        "id_estudiante": identificador
                    },
                    {
                        "_id": 0
                    }
                ).sort(
                    "fecha_registro",
                    -1
                )
            )

            if expediente is None:

                st.warning(
                    "No se encontró un expediente con ese identificador."
                )

            else:

                st.success(
                    "Expediente localizado correctamente."
                )

                st.subheader("Datos generales")

                st.json(expediente)

                st.subheader("Historial académico")

                st.write(
                    f"Evaluaciones registradas: {len(evaluaciones)}"
                )

                for numero, evaluacion in enumerate(
                    evaluaciones,
                    start=1
                ):

                    with st.expander(
                        f"Evaluación {numero} - "
                        f"{evaluacion.get('fecha_registro')}"
                    ):

                        st.write(
                            "**Ciclo escolar:**",
                            evaluacion.get("ciclo_escolar")
                        )

                        st.write(
                            "**Grupo:**",
                            evaluacion.get("grupo")
                        )

                        st.write(
                            "**Cuatrimestre:**",
                            evaluacion.get("cuatrimestre")
                        )

                        st.write(
                            "**Resumen académico:**"
                        )

                        st.json(
                            evaluacion.get(
                                "resumen_academico",
                                {}
                            )
                        )

                        st.write(
                            "**Materias:**"
                        )

                        st.dataframe(
                            pd.DataFrame(
                                evaluacion.get(
                                    "materias",
                                    []
                                )
                            ),
                            use_container_width=True,
                            hide_index=True
                        )

                        st.write(
                            "**Variables de aprendizaje:**"
                        )

                        st.json(
                            evaluacion.get(
                                "variables_aprendizaje",
                                {}
                            )
                        )

                        # Consultar el análisis de IA relacionado
                        analisis = coleccion_ia.find_one(
                            {
                                "id_analisis": evaluacion.get(
                                    "id_analisis"
                                )
                            },
                            {
                                "_id": 0
                            }
                        )

                        if analisis:

                            st.write(
                                "**Análisis de inteligencia artificial:**"
                            )

                            st.json(
                                analisis.get(
                                    "datos_reporte",
                                    {}
                                )
                            )

        except PyMongoError as error:

            st.error(
                "No fue posible consultar MongoDB Atlas."
            )

            st.exception(error)

        except Exception as error:

            st.error(
                "Ocurrió un error durante la consulta."
            )

            st.exception(error)

        finally:

            if cliente is not None:

                cliente.close()


# ============================================================
# 17. PIE DE PAGINA
# ============================================================

st.divider()

st.caption(
    "SMARTACADEMY | Expediente académico integral "
    "con Machine Learning y MongoDB Atlas."
)
