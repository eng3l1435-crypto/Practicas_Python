
# ============================================================
# PROYECTO: SMARTACADEMY
# ETAPA 18.1: PREPARACION DE DATOS PARA PREDICCION
#
# Archivo:
# p26preparacion_prediccion_cnn.py
#
# Objetivo:
# Preparar los datos para entrenar una CNN 1D que estime
# la calificacion del tercer parcial utilizando solamente:
#
# - Calificacion del parcial 1
# - Calificacion del parcial 2
# - Asistencia del parcial 1
# - Asistencia del parcial 2
#
# El parcial 3 sera exclusivamente la variable objetivo.
# ============================================================

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# 1. CONFIGURACION DE RUTAS
# ============================================================

# Obtenemos la carpeta principal del proyecto.
RUTA_PROYECTO = Path(__file__).resolve().parents[1]

# Archivo original con los 50,000 registros.
ARCHIVO_CSV = (
    RUTA_PROYECTO
    / "resultados_bigdata"
    / "smartacademy_50000.csv"
)

# Carpeta donde guardaremos los nuevos datos.
RUTA_SALIDA = (
    RUTA_PROYECTO
    / "resultados_prediccion_cnn"
)

RUTA_SALIDA.mkdir(
    parents=True,
    exist_ok=True
)

ARCHIVO_NPZ = (
    RUTA_SALIDA
    / "smartacademy_prediccion_parcial3.npz"
)

ARCHIVO_JSON = (
    RUTA_SALIDA
    / "metadatos_prediccion.json"
)


# ============================================================
# 2. CONFIGURACION GENERAL
# ============================================================

SEMILLA = 42

# Proporciones de los conjuntos:
#
# 70% entrenamiento
# 15% validacion
# 15% prueba

PROPORCION_ENTRENAMIENTO = 0.70

PROPORCION_VALIDACION = 0.15

PROPORCION_PRUEBA = 0.15


# ============================================================
# 3. FUNCION PARA MOSTRAR TITULOS
# ============================================================

def mostrar_titulo(mensaje):

    print("\n" + "=" * 65)

    print(mensaje)

    print("=" * 65)


# ============================================================
# 4. CARGAR LOS DATOS ORIGINALES
# ============================================================

mostrar_titulo(
    "SMARTACADEMY - ETAPA 18.1"
)

print("\nCARGANDO DATASET ORIGINAL...")

if not ARCHIVO_CSV.exists():

    raise FileNotFoundError(
        f"No existe el archivo: {ARCHIVO_CSV}"
    )

# Leemos el CSV completo.
datos = pd.read_csv(ARCHIVO_CSV)

print("\nDATASET CARGADO CORRECTAMENTE")

print(
    f"Registros encontrados: {len(datos):,}"
)

print(
    f"Columnas encontradas: {len(datos.columns)}"
)


# ============================================================
# 5. DEFINIR LAS COLUMNAS NECESARIAS
# ============================================================

columnas_requeridas = [

    "id_registro",

    "id_estudiante",

    "materia",

    "parcial1",

    "parcial2",

    "parcial3",

    "asistencia_p1",

    "asistencia_p2",

    "asistencia_p3"

]

# Comprobamos que todas las columnas existan.
columnas_faltantes = [

    columna

    for columna in columnas_requeridas

    if columna not in datos.columns

]

if columnas_faltantes:

    raise ValueError(
        f"Faltan columnas: {columnas_faltantes}"
    )

print("\nCOLUMNAS REQUERIDAS VERIFICADAS")


# ============================================================
# 6. VALIDAR IDENTIFICADORES Y REGISTROS
# ============================================================

mostrar_titulo(
    "VALIDACION DE INTEGRIDAD"
)

# Comprobamos identificadores vacios.
if datos["id_estudiante"].isnull().any():

    raise ValueError(
        "Existen registros sin identificador de estudiante."
    )

if datos["materia"].isnull().any():

    raise ValueError(
        "Existen registros sin materia."
    )

# Cada registro debe tener un identificador unico.
if datos["id_registro"].duplicated().any():

    raise ValueError(
        "Se encontraron identificadores de registro duplicados."
    )

# No debe repetirse una materia para un estudiante.
if datos.duplicated(
    subset=["id_estudiante", "materia"]
).any():

    raise ValueError(
        "Existen materias duplicadas para un estudiante."
    )

# Revisamos que existan cinco materias por estudiante.
materias_por_estudiante = (
    datos.groupby("id_estudiante")["materia"]
    .nunique()
)

if not (materias_por_estudiante == 5).all():

    raise ValueError(
        "No todos los estudiantes tienen cinco materias."
    )

print("Identificadores: correctos")

print("Materias duplicadas: ninguna")

print("Cinco materias por estudiante: validado")

print(
    f"Estudiantes encontrados: "
    f"{datos['id_estudiante'].nunique():,}"
)


# ============================================================
# 7. VALIDAR CALIFICACIONES Y ASISTENCIAS
# ============================================================

mostrar_titulo(
    "VALIDACION DE VARIABLES NUMERICAS"
)

columnas_numericas = [

    "parcial1",

    "parcial2",

    "parcial3",

    "asistencia_p1",

    "asistencia_p2",

    "asistencia_p3"

]

# Convertimos las columnas a valores numericos.
for columna in columnas_numericas:

    datos[columna] = pd.to_numeric(
        datos[columna],
        errors="coerce"
    )

# Revisamos valores nulos o no numericos.
if datos[columnas_numericas].isnull().any().any():

    raise ValueError(
        "Existen valores nulos o no numericos."
    )

if not np.isfinite(
    datos[columnas_numericas].to_numpy()
).all():

    raise ValueError(
        "Se encontraron valores infinitos."
    )

# Las calificaciones deben estar entre 0 y 10.
for columna in [
    "parcial1",
    "parcial2",
    "parcial3"
]:

    if not datos[columna].between(0, 10).all():

        raise ValueError(
            f"Calificaciones fuera de rango en {columna}"
        )

# Las asistencias deben estar entre 0 y 100.
for columna in [
    "asistencia_p1",
    "asistencia_p2",
    "asistencia_p3"
]:

    if not datos[columna].between(0, 100).all():

        raise ValueError(
            f"Asistencias fuera de rango en {columna}"
        )

print("Calificaciones: rango 0 a 10 correcto")

print("Asistencias: rango 0 a 100 correcto")

print("Valores nulos: ninguno")

print("Valores infinitos: ninguno")


# ============================================================
# 8. CONSTRUIR LAS VARIABLES DE ENTRADA Y OBJETIVO
# ============================================================

mostrar_titulo(
    "CONSTRUCCION DEL CONJUNTO DE PREDICCION"
)

# Creamos un arreglo tridimensional.
#
# Dimensiones:
#
# (registros, canales, parciales)
#
# Canal 0: calificaciones.
# Canal 1: asistencias.
#
# Posicion 0: parcial 1.
# Posicion 1: parcial 2.

X = np.zeros(
    (len(datos), 2, 2),
    dtype=np.float32
)

# Canal 0: calificaciones.
X[:, 0, 0] = datos["parcial1"].to_numpy()

X[:, 0, 1] = datos["parcial2"].to_numpy()

# Canal 1: asistencias.
X[:, 1, 0] = datos["asistencia_p1"].to_numpy()

X[:, 1, 1] = datos["asistencia_p2"].to_numpy()

# La variable objetivo es exclusivamente el parcial 3.
y = datos["parcial3"].to_numpy(
    dtype=np.float32
)

# Identificador de cada estudiante.
grupos = datos["id_estudiante"].to_numpy()

# Identificador de la materia para análisis posteriores.
materias = datos["materia"].to_numpy()

print("\nVARIABLES CONSTRUIDAS")

print(f"Entradas X: {X.shape}")

print(f"Objetivo y: {y.shape}")

print("\nVariables de entrada:")

print("Calificacion parcial 1")

print("Calificacion parcial 2")

print("Asistencia parcial 1")

print("Asistencia parcial 2")

print("\nVariable objetivo:")

print("Calificacion parcial 3")

print(
    "\nIMPORTANTE: el parcial 3 no forma parte de X."
)

print(
    "La asistencia del parcial 3 tampoco forma parte de X."
)


# ============================================================
# 9. SEPARAR ESTUDIANTES SIN MEZCLAR SUS REGISTROS
# ============================================================

mostrar_titulo(
    "SEPARACION DE ENTRENAMIENTO, VALIDACION Y PRUEBA"
)

# Primera separacion:
#
# 70% entrenamiento.
# 30% temporal.

separador_1 = GroupShuffleSplit(

    n_splits=1,

    test_size=1 - PROPORCION_ENTRENAMIENTO,

    random_state=SEMILLA

)

indices_train, indices_temporales = next(

    separador_1.split(
        X,
        y,
        groups=grupos
    )

)

# Segunda separacion:
#
# Dividimos el 30% temporal en dos partes iguales.
#
# 15% validacion.
# 15% prueba.

grupos_temporales = grupos[indices_temporales]

separador_2 = GroupShuffleSplit(

    n_splits=1,

    test_size=0.50,

    random_state=SEMILLA

)

indices_val_relativos, indices_test_relativos = next(

    separador_2.split(
        indices_temporales,
        y[indices_temporales],
        groups=grupos_temporales
    )

)

# Convertimos los indices relativos en indices originales.

indices_val = indices_temporales[
    indices_val_relativos
]

indices_test = indices_temporales[
    indices_test_relativos
]


# ============================================================
# 10. CONSTRUIR LOS TRES CONJUNTOS
# ============================================================

X_train_raw = X[indices_train]

X_val_raw = X[indices_val]

X_test_raw = X[indices_test]

y_train = y[indices_train]

y_val = y[indices_val]

y_test = y[indices_test]

grupos_train = grupos[indices_train]

grupos_val = grupos[indices_val]

grupos_test = grupos[indices_test]

materias_train = materias[indices_train]

materias_val = materias[indices_val]

materias_test = materias[indices_test]


# ============================================================
# 11. VALIDAR QUE NO EXISTAN ESTUDIANTES COMPARTIDOS
# ============================================================

ids_train = set(grupos_train)

ids_val = set(grupos_val)

ids_test = set(grupos_test)

# Un estudiante no puede aparecer en dos conjuntos.

assert ids_train.isdisjoint(ids_val)

assert ids_train.isdisjoint(ids_test)

assert ids_val.isdisjoint(ids_test)

print("\nSEPARACION POR ESTUDIANTE CORRECTA")

print(
    f"Estudiantes entrenamiento: {len(ids_train):,}"
)

print(
    f"Estudiantes validacion: {len(ids_val):,}"
)

print(
    f"Estudiantes prueba: {len(ids_test):,}"
)

print("\nREGISTROS POR CONJUNTO")

print(f"Entrenamiento: {len(X_train_raw):,}")

print(f"Validacion:    {len(X_val_raw):,}")

print(f"Prueba:        {len(X_test_raw):,}")


# ============================================================
# 12. NORMALIZAR LAS VARIABLES DE ENTRADA
# ============================================================

mostrar_titulo(
    "NORMALIZACION DE VARIABLES"
)

# Calculamos media y desviacion estandar exclusivamente
# con el conjunto de entrenamiento.
#
# axis=(0, 2):
# - Conserva cada canal.
# - Considera registros y posiciones temporales.
#
# keepdims=True permite utilizar directamente los valores
# para normalizar los tensores tridimensionales.

media_entrenamiento = X_train_raw.mean(
    axis=(0, 2),
    keepdims=True
)

desviacion_entrenamiento = X_train_raw.std(
    axis=(0, 2),
    keepdims=True
)

# Evitamos divisiones entre cero.
desviacion_entrenamiento = np.where(

    desviacion_entrenamiento == 0,

    1.0,

    desviacion_entrenamiento

)

# Aplicamos las mismas estadisticas a los tres conjuntos.
#
# Es importante NO calcular una media diferente para
# validacion o prueba.

X_train = (
    X_train_raw - media_entrenamiento
) / desviacion_entrenamiento

X_val = (
    X_val_raw - media_entrenamiento
) / desviacion_entrenamiento

X_test = (
    X_test_raw - media_entrenamiento
) / desviacion_entrenamiento

# Convertimos a float32 para PyTorch.
X_train = X_train.astype(np.float32)

X_val = X_val.astype(np.float32)

X_test = X_test.astype(np.float32)

print("\nNORMALIZACION COMPLETADA")

print(
    f"Media calificaciones: "
    f"{media_entrenamiento[0, 0, 0]:.4f}"
)

print(
    f"Desviacion calificaciones: "
    f"{desviacion_entrenamiento[0, 0, 0]:.4f}"
)

print(
    f"Media asistencias: "
    f"{media_entrenamiento[0, 1, 0]:.4f}"
)

print(
    f"Desviacion asistencias: "
    f"{desviacion_entrenamiento[0, 1, 0]:.4f}"
)


# ============================================================
# 13. VALIDAR LAS DIMENSIONES FINALES
# ============================================================

mostrar_titulo(
    "VALIDACION FINAL DE TENSORES"
)

assert X_train.ndim == 3

assert X_val.ndim == 3

assert X_test.ndim == 3

assert X_train.shape[1:] == (2, 2)

assert X_val.shape[1:] == (2, 2)

assert X_test.shape[1:] == (2, 2)

assert y_train.ndim == 1

assert y_val.ndim == 1

assert y_test.ndim == 1

assert len(X_train) == len(y_train)

assert len(X_val) == len(y_val)

assert len(X_test) == len(y_test)

assert np.isfinite(X_train).all()

assert np.isfinite(X_val).all()

assert np.isfinite(X_test).all()

assert np.isfinite(y_train).all()

assert np.isfinite(y_val).all()

assert np.isfinite(y_test).all()

print("\nDIMENSIONES CORRECTAS")

print(f"X_train: {X_train.shape}")

print(f"y_train: {y_train.shape}")

print(f"X_val:   {X_val.shape}")

print(f"y_val:   {y_val.shape}")

print(f"X_test:  {X_test.shape}")

print(f"y_test:  {y_test.shape}")


# ============================================================
# 14. ESTADISTICAS DE LA VARIABLE OBJETIVO
# ============================================================

mostrar_titulo(
    "ANALISIS DE CALIFICACIONES DEL PARCIAL 3"
)

print(
    f"Promedio entrenamiento: {y_train.mean():.4f}"
)

print(
    f"Promedio validacion: {y_val.mean():.4f}"
)

print(
    f"Promedio prueba: {y_test.mean():.4f}"
)

print(
    f"Minimo: {y.min():.2f}"
)

print(
    f"Maximo: {y.max():.2f}"
)

print(
    f"Desviacion estandar: {y.std():.4f}"
)


# ============================================================
# 15. GUARDAR LOS DATOS PREPARADOS
# ============================================================

mostrar_titulo(
    "GUARDANDO DATOS PARA PYTORCH"
)

# Guardamos tambien los identificadores como cadenas Unicode.
#
# De esta manera el archivo puede abrirse con allow_pickle=False.
#
# Los identificadores y las materias sirven para auditar
# posteriormente las predicciones, pero NO se utilizaran
# como entradas del modelo.

np.savez_compressed(

    ARCHIVO_NPZ,

    X_train=X_train,

    y_train=y_train,

    X_val=X_val,

    y_val=y_val,

    X_test=X_test,

    y_test=y_test,

    ids_train=grupos_train.astype(str),

    ids_val=grupos_val.astype(str),

    ids_test=grupos_test.astype(str),

    materias_train=materias_train.astype(str),

    materias_val=materias_val.astype(str),

    materias_test=materias_test.astype(str),

    media_entrenamiento=media_entrenamiento,

    desviacion_entrenamiento=desviacion_entrenamiento

)

print(f"\nArchivo NPZ guardado: {ARCHIVO_NPZ}")


# ============================================================
# 16. GUARDAR METADATOS
# ============================================================

metadatos = {

    "proyecto": "SmartAcademy",

    "etapa": 18.1,

    "tipo_problema": "Regresion",

    "objetivo": "Prediccion de calificacion del parcial 3",

    "total_registros": int(len(datos)),

    "total_estudiantes": int(
        datos["id_estudiante"].nunique()
    ),

    "materias_por_estudiante": 5,

    "variables_entrada": [

        "parcial1",

        "parcial2",

        "asistencia_p1",

        "asistencia_p2"

    ],

    "variable_objetivo": "parcial3",

    "variables_excluidas": [

        "parcial3 como entrada",

        "asistencia_p3",

        "promedio_final",

        "tendencia",

        "resultado",

        "id_estudiante",

        "materia"

    ],

    "forma_entrada": [2, 2],

    "separacion": {

        "entrenamiento": 0.70,

        "validacion": 0.15,

        "prueba": 0.15,

        "metodo": "GroupShuffleSplit",

        "grupo": "id_estudiante"

    },

    "semilla": SEMILLA,

    "normalizacion": {

        "metodo": "Z-score",

        "calculada_con": "Solo entrenamiento",

        "media": media_entrenamiento.tolist(),

        "desviacion": desviacion_entrenamiento.tolist()

    },

    "advertencia": (
        "Los datos son sinteticos. Los resultados deben "
        "interpretarse como una validacion experimental "
        "y no como evidencia de rendimiento con datos reales."
    )

}

with open(

    ARCHIVO_JSON,

    "w",

    encoding="utf-8"

) as archivo:

    json.dump(

        metadatos,

        archivo,

        indent=4,

        ensure_ascii=False

    )

print(f"Metadatos guardados: {ARCHIVO_JSON}")


# ============================================================
# 17. RESUMEN FINAL
# ============================================================

mostrar_titulo(
    "ETAPA 18.1 FINALIZADA"
)

print("\nPREPARACION COMPLETADA CORRECTAMENTE")

print(f"Registros procesados: {len(datos):,}")

print(f"Estudiantes: {datos['id_estudiante'].nunique():,}")

print(f"Entrenamiento: {len(X_train):,}")

print(f"Validacion: {len(X_val):,}")

print(f"Prueba: {len(X_test):,}")

print("\nENTRADA DE LA CNN:")

print("(2 canales, 2 parciales)")

print("\nOBJETIVO:")

print("Calificacion del parcial 3")

print("\nARCHIVOS GENERADOS:")

print(ARCHIVO_NPZ)

print(ARCHIVO_JSON)

print("\nNO SE DETECTO FUGA DE INFORMACION POR ESTUDIANTE.")

print("=" * 65)
