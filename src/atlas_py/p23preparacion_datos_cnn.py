
# ============================================================
# PROYECTO: SMARTACADEMY
# ETAPA 15 - PREPARACION DE DATOS PARA CNN 1D
#
# Archivo:
# p23preparacion_datos_cnn.py
#
# Objetivo:
# 1. Leer los 50,000 registros procesados con Spark.
# 2. Seleccionar las variables temporales de cada estudiante.
# 3. Convertir los registros en tensores para una CNN 1D.
# 4. Codificar las tres categorías de tendencia.
# 5. Dividir los datos por estudiante.
# 6. Normalizar usando exclusivamente entrenamiento.
# 7. Guardar los conjuntos para el entrenamiento posterior.
#
# IMPORTANTE:
# No se entrena la red neuronal en esta etapa.
# ============================================================


# ============================================================
# 1. IMPORTACION DE LIBRERIAS
# ============================================================

from pathlib import Path
import json

import numpy as np
import pandas as pd

from pyspark.sql import SparkSession

from sklearn.model_selection import GroupShuffleSplit

import torch


# ============================================================
# 2. CONFIGURACION GENERAL
# ============================================================

# Directorio principal del proyecto.
RUTA_PROYECTO = Path(__file__).resolve().parents[1]

# Directorio donde Spark guardó los archivos Parquet.
RUTA_PARQUET = (
    RUTA_PROYECTO
    / "resultados_bigdata"
    / "spark_procesado"
)

# Directorio donde guardaremos los datos preparados.
RUTA_SALIDA = (
    RUTA_PROYECTO
    / "resultados_cnn"
)

# Crear el directorio si todavía no existe.
RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

# Semilla para reproducir la misma separación de datos.
SEMILLA = 42

# Número esperado de registros.
REGISTROS_ESPERADOS = 50000

# Definición fija de las clases.
# El número representa la categoría que aprenderá la CNN.
MAPA_ETIQUETAS = {
    "Descendente": 0,
    "Estable": 1,
    "Favorable": 2
}

# Variables que representan la evolución académica.
COLUMNAS_CALIFICACIONES = [
    "parcial1",
    "parcial2",
    "parcial3"
]

COLUMNAS_ASISTENCIA = [
    "asistencia_p1",
    "asistencia_p2",
    "asistencia_p3"
]

# Columnas que necesitamos leer.
COLUMNAS_NECESARIAS = [
    "id_registro",
    "id_estudiante",
    "parcial1",
    "parcial2",
    "parcial3",
    "asistencia_p1",
    "asistencia_p2",
    "asistencia_p3",
    "tendencia"
]


# ============================================================
# 3. CREAR SESION SPARK
# ============================================================

def crear_sesion_spark():

    """
    Crea una sesión local de Spark.

    Spark se utiliza para leer los archivos Parquet
    que fueron generados durante la etapa anterior.
    """

    spark = (
        SparkSession.builder
        .appName("SmartAcademy_Preparacion_CNN")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

    # Reducir mensajes informativos de Spark.
    spark.sparkContext.setLogLevel("ERROR")

    return spark


# ============================================================
# 4. LEER LOS DATOS PROCESADOS
# ============================================================

def cargar_datos(spark):

    """
    Lee los archivos Parquet generados por Spark.

    Se seleccionan únicamente las columnas necesarias
    para construir las secuencias temporales.
    """

    print("\n1. LEYENDO DATOS PARQUET")
    print("-" * 60)

    # Verificar que exista el directorio.
    if not RUTA_PARQUET.exists():

        raise FileNotFoundError(
            f"No existe el directorio: {RUTA_PARQUET}"
        )

    # Leer todos los archivos Parquet.
    datos_spark = spark.read.parquet(
        str(RUTA_PARQUET)
    )

    # Verificar que existan todas las columnas.
    columnas_disponibles = set(datos_spark.columns)

    columnas_faltantes = (
        set(COLUMNAS_NECESARIAS)
        - columnas_disponibles
    )

    if columnas_faltantes:

        raise ValueError(
            "Faltan columnas necesarias: "
            f"{columnas_faltantes}"
        )

    # Seleccionar solamente las variables necesarias.
    datos_spark = datos_spark.select(
        *COLUMNAS_NECESARIAS
    )

    # Contar registros.
    total = datos_spark.count()

    print(f"Registros encontrados: {total:,}")

    if total != REGISTROS_ESPERADOS:

        raise ValueError(
            "El número de registros no coincide. "
            f"Esperados: {REGISTROS_ESPERADOS:,}; "
            f"encontrados: {total:,}"
        )

    print("Archivos Parquet leídos correctamente.")

    # Transferimos los 50,000 registros a Pandas.
    # Este conjunto tiene un tamaño manejable para
    # preparar los tensores de entrenamiento.
    datos = datos_spark.toPandas()

    return datos


# ============================================================
# 5. VALIDAR LA CALIDAD DE LOS DATOS
# ============================================================

def validar_datos(datos):

    """
    Comprueba que los registros no tengan:
    - Valores nulos.
    - Identificadores duplicados.
    - Calificaciones fuera del rango 0-10.
    - Asistencias fuera del rango 0-100.
    - Categorías de tendencia desconocidas.
    """

    print("\n2. VALIDANDO CALIDAD DE LOS DATOS")
    print("-" * 60)

    # Revisar valores nulos.
    total_nulos = int(
        datos[COLUMNAS_NECESARIAS]
        .isnull()
        .sum()
        .sum()
    )

    print(f"Valores nulos: {total_nulos}")

    if total_nulos > 0:

        raise ValueError(
            "Se encontraron valores nulos."
        )

    # Revisar registros duplicados.
    duplicados = int(
        datos["id_registro"].duplicated().sum()
    )

    print(f"Identificadores duplicados: {duplicados}")

    if duplicados > 0:

        raise ValueError(
            "Existen identificadores duplicados."
        )

    # Revisar rangos de calificaciones.
    calificaciones = datos[
        COLUMNAS_CALIFICACIONES
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(calificaciones).all():

        raise ValueError(
            "Hay calificaciones no numéricas o infinitas."
        )

    if (
        (calificaciones < 0).any()
        or (calificaciones > 10).any()
    ):

        raise ValueError(
            "Hay calificaciones fuera del rango 0-10."
        )

    # Revisar rangos de asistencia.
    asistencias = datos[
        COLUMNAS_ASISTENCIA
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(asistencias).all():

        raise ValueError(
            "Hay asistencias no numéricas o infinitas."
        )

    if (
        (asistencias < 0).any()
        or (asistencias > 100).any()
    ):

        raise ValueError(
            "Hay asistencias fuera del rango 0-100."
        )

    # Revisar que las tendencias sean conocidas.
    tendencias_encontradas = set(
        datos["tendencia"].unique()
    )

    tendencias_validas = set(
        MAPA_ETIQUETAS.keys()
    )

    if tendencias_encontradas != tendencias_validas:

        raise ValueError(
            "Las categorías de tendencia no coinciden. "
            f"Encontradas: {tendencias_encontradas}"
        )

    # Verificar que cada estudiante tenga cinco materias.
    materias_por_estudiante = (
        datos.groupby("id_estudiante")["id_registro"]
        .count()
    )

    estudiantes_incorrectos = int(
        (materias_por_estudiante != 5).sum()
    )

    print(
        "Estudiantes con cinco registros: "
        f"{len(materias_por_estudiante) - estudiantes_incorrectos:,}"
    )

    if estudiantes_incorrectos > 0:

        raise ValueError(
            "Hay estudiantes que no tienen exactamente "
            "cinco registros académicos."
        )

    print("Validación de datos completada.")


# ============================================================
# 6. CONVERTIR LOS REGISTROS EN TENSORES
# ============================================================

def construir_tensores(datos):

    """
    Convierte los registros en una estructura adecuada
    para una red neuronal convolucional unidimensional.

    Cada registro tendrá:

        Canal 0: calificaciones.
        Canal 1: asistencia.

    Tres posiciones temporales:

        Posición 0: primer parcial.
        Posición 1: segundo parcial.
        Posición 2: tercer parcial.

    Resultado:

        X = [50000, 2, 3]
        y = [50000]

    """

    print("\n3. CONSTRUYENDO TENSORES")
    print("-" * 60)

    # Extraer las calificaciones.
    calificaciones = datos[
        COLUMNAS_CALIFICACIONES
    ].to_numpy(dtype=np.float32)

    # Extraer las asistencias.
    asistencias = datos[
        COLUMNAS_ASISTENCIA
    ].to_numpy(dtype=np.float32)

    # Construir los dos canales.
    #
    # axis=1 indica que cada registro tendrá
    # primero el canal de calificaciones y
    # después el canal de asistencias.
    X = np.stack(
        [
            calificaciones,
            asistencias
        ],
        axis=1
    )

    # Convertir las categorías de texto a números.
    y = (
        datos["tendencia"]
        .map(MAPA_ETIQUETAS)
        .to_numpy(dtype=np.int64)
    )

    # Identificador del estudiante asociado a cada registro.
    # Lo utilizaremos para separar los conjuntos sin mezclar
    # registros del mismo estudiante.
    grupos = datos[
        "id_estudiante"
    ].astype(str).to_numpy()

    print(f"Forma de X: {X.shape}")
    print(f"Forma de y: {y.shape}")

    print("\nEjemplo de un registro:")
    print(X[0])

    print("\nEtiqueta numérica:")
    print(y[0])

    # Verificar dimensiones.
    if X.shape != (REGISTROS_ESPERADOS, 2, 3):

        raise ValueError(
            f"Forma inesperada para X: {X.shape}"
        )

    if y.shape != (REGISTROS_ESPERADOS,):

        raise ValueError(
            f"Forma inesperada para y: {y.shape}"
        )

    print("\nTensores construidos correctamente.")

    return X, y, grupos


# ============================================================
# 7. DIVIDIR LOS DATOS POR ESTUDIANTE
# ============================================================

def dividir_datos(X, y, grupos):

    """
    Separa los datos en:

        70% entrenamiento.
        15% validación.
        15% prueba.

    La separación se realiza por estudiante.

    Esto evita que un mismo estudiante aparezca
    en entrenamiento y también en prueba.

    Es importante porque cada estudiante tiene cinco materias.
    """

    print("\n4. DIVIDIENDO LOS CONJUNTOS")
    print("-" * 60)

    # Primera separación:
    # 85% para entrenamiento + validación.
    # 15% para prueba.
    separador_1 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.15,
        random_state=SEMILLA
    )

    indices_train_val, indices_test = next(
        separador_1.split(
            X,
            y,
            groups=grupos
        )
    )

    # Extraer temporalmente el conjunto de entrenamiento
    # y validación.
    X_train_val = X[indices_train_val]
    y_train_val = y[indices_train_val]
    grupos_train_val = grupos[indices_train_val]

    # Segunda separación:
    # Del 85% restante, reservamos aproximadamente
    # 17.65% para validación.
    #
    # El resultado global será aproximadamente:
    # 70% entrenamiento, 15% validación y 15% prueba.
    separador_2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.1764705882,
        random_state=SEMILLA
    )

    indices_train_rel, indices_val_rel = next(
        separador_2.split(
            X_train_val,
            y_train_val,
            groups=grupos_train_val
        )
    )

    # Recuperar los índices originales.
    indices_train = indices_train_val[
        indices_train_rel
    ]

    indices_val = indices_train_val[
        indices_val_rel
    ]

    # Construir los tres conjuntos.
    X_train = X[indices_train]
    y_train = y[indices_train]
    grupos_train = grupos[indices_train]

    X_val = X[indices_val]
    y_val = y[indices_val]
    grupos_val = grupos[indices_val]

    X_test = X[indices_test]
    y_test = y[indices_test]
    grupos_test = grupos[indices_test]

    # Verificar que ningún estudiante aparezca
    # en más de un conjunto.
    conjunto_train = set(grupos_train)
    conjunto_val = set(grupos_val)
    conjunto_test = set(grupos_test)

    if conjunto_train & conjunto_val:

        raise ValueError(
            "Hay estudiantes compartidos entre "
            "entrenamiento y validación."
        )

    if conjunto_train & conjunto_test:

        raise ValueError(
            "Hay estudiantes compartidos entre "
            "entrenamiento y prueba."
        )

    if conjunto_val & conjunto_test:

        raise ValueError(
            "Hay estudiantes compartidos entre "
            "validación y prueba."
        )

    print(
        f"Entrenamiento: {len(X_train):,} registros "
        f"({len(conjunto_train):,} estudiantes)"
    )

    print(
        f"Validación: {len(X_val):,} registros "
        f"({len(conjunto_val):,} estudiantes)"
    )

    print(
        f"Prueba: {len(X_test):,} registros "
        f"({len(conjunto_test):,} estudiantes)"
    )

    print("Separación por estudiante verificada.")

    return (
        X_train, y_train, grupos_train,
        X_val, y_val, grupos_val,
        X_test, y_test, grupos_test
    )


# ============================================================
# 8. NORMALIZAR LOS DATOS
# ============================================================

def normalizar_datos(
    X_train,
    X_val,
    X_test
):

    """
    Normaliza los dos canales de entrada.

    Se calcula la media y desviación estándar
    exclusivamente con los datos de entrenamiento.

    Posteriormente se aplican esos mismos parámetros
    a validación y prueba.

    Esto evita fuga de información.
    """

    print("\n5. NORMALIZANDO LOS DATOS")
    print("-" * 60)

    # Cambiar temporalmente la forma para calcular
    # estadísticas por canal.
    #
    # Se reúnen las observaciones de los tres parciales.
    datos_train = X_train.transpose(
        0, 2, 1
    ).reshape(-1, 2)

    # Media de cada canal.
    medias = datos_train.mean(
        axis=0
    )

    # Desviación estándar de cada canal.
    desviaciones = datos_train.std(
        axis=0
    )

    # Evitar división entre cero.
    desviaciones = np.where(
        desviaciones == 0,
        1.0,
        desviaciones
    )

    # Normalizar conservando la estructura [N, 2, 3].
    X_train_norm = (
        X_train - medias[None, :, None]
    ) / desviaciones[None, :, None]

    X_val_norm = (
        X_val - medias[None, :, None]
    ) / desviaciones[None, :, None]

    X_test_norm = (
        X_test - medias[None, :, None]
    ) / desviaciones[None, :, None]

    # Asegurar tipo float32 para PyTorch.
    X_train_norm = X_train_norm.astype(
        np.float32
    )

    X_val_norm = X_val_norm.astype(
        np.float32
    )

    X_test_norm = X_test_norm.astype(
        np.float32
    )

    print(
        f"Media calificaciones: {medias[0]:.4f}"
    )

    print(
        f"Desviación calificaciones: {desviaciones[0]:.4f}"
    )

    print(
        f"Media asistencia: {medias[1]:.4f}"
    )

    print(
        f"Desviación asistencia: {desviaciones[1]:.4f}"
    )

    print("Normalización completada.")

    return (
        X_train_norm,
        X_val_norm,
        X_test_norm,
        medias,
        desviaciones
    )


# ============================================================
# 9. VALIDAR DISTRIBUCION DE LAS CLASES
# ============================================================

def mostrar_distribucion(y, nombre):

    """
    Muestra cuántos registros corresponden
    a cada categoría de tendencia.
    """

    print(f"\n{nombre}")

    for etiqueta, codigo in MAPA_ETIQUETAS.items():

        cantidad = int(
            np.sum(y == codigo)
        )

        porcentaje = (
            cantidad / len(y)
        ) * 100

        print(
            f"{etiqueta:12}: "
            f"{cantidad:6,} registros "
            f"({porcentaje:.2f}%)"
        )


def validar_clases(y_train, y_val, y_test):

    """
    Comprueba que las tres categorías estén presentes
    en entrenamiento, validación y prueba.
    """

    print("\n6. DISTRIBUCION DE CLASES")
    print("-" * 60)

    mostrar_distribucion(
        y_train,
        "ENTRENAMIENTO"
    )

    mostrar_distribucion(
        y_val,
        "VALIDACION"
    )

    mostrar_distribucion(
        y_test,
        "PRUEBA"
    )

    clases_esperadas = set(
        MAPA_ETIQUETAS.values()
    )

    for nombre, conjunto in [
        ("Entrenamiento", y_train),
        ("Validación", y_val),
        ("Prueba", y_test)
    ]:

        clases_presentes = set(
            np.unique(conjunto).tolist()
        )

        if clases_presentes != clases_esperadas:

            raise ValueError(
                f"Faltan categorías en {nombre}. "
                f"Encontradas: {clases_presentes}"
            )

    print("\nLas tres clases están presentes en los conjuntos.")


# ============================================================
# 10. GUARDAR LOS DATOS PREPARADOS
# ============================================================

def guardar_datos(
    X_train,
    y_train,
    grupos_train,
    X_val,
    y_val,
    grupos_val,
    X_test,
    y_test,
    grupos_test,
    medias,
    desviaciones
):

    """
    Guarda los tensores y metadatos.

    El archivo NPZ contiene:
    - Entrenamiento.
    - Validación.
    - Prueba.
    - Identificadores de estudiantes.

    El archivo JSON contiene:
    - Mapeo de etiquetas.
    - Parámetros de normalización.
    - Dimensiones.
    - Semilla utilizada.
    """

    print("\n7. GUARDANDO LOS RESULTADOS")
    print("-" * 60)

    ruta_datos = (
        RUTA_SALIDA
        / "smartacademy_datos_cnn.npz"
    )

    ruta_metadatos = (
        RUTA_SALIDA
        / "metadatos_cnn.json"
    )

    # Guardar matrices y etiquetas en formato comprimido.
    np.savez_compressed(
        ruta_datos,

        X_train=X_train,
        y_train=y_train,
        grupos_train=grupos_train,

        X_val=X_val,
        y_val=y_val,
        grupos_val=grupos_val,

        X_test=X_test,
        y_test=y_test,
        grupos_test=grupos_test
    )

    # Guardar la configuración necesaria para
    # interpretar posteriormente las predicciones.
    metadatos = {

        "proyecto": "SmartAcademy",

        "etapa": 15,

        "total_registros": REGISTROS_ESPERADOS,

        "forma_entrada": [
            2,
            3
        ],

        "canales": [
            "calificaciones",
            "asistencia"
        ],

        "momentos_temporales": [
            "parcial1",
            "parcial2",
            "parcial3"
        ],

        "mapa_etiquetas": MAPA_ETIQUETAS,

        "media_normalizacion": (
            medias.tolist()
        ),

        "desviacion_normalizacion": (
            desviaciones.tolist()
        ),

        "semilla": SEMILLA,

        "separacion": {
            "entrenamiento": 0.70,
            "validacion": 0.15,
            "prueba": 0.15,
            "metodo": "GroupShuffleSplit por estudiante"
        },

        "framework": "PyTorch",

        "version_pytorch": torch.__version__,

        "cuda_disponible": torch.cuda.is_available()
    }

    with open(
        ruta_metadatos,
        "w",
        encoding="utf-8"
    ) as archivo:

        json.dump(
            metadatos,
            archivo,
            indent=4,
            ensure_ascii=False
        )

    print(f"Datos guardados: {ruta_datos}")

    print(
        f"Metadatos guardados: {ruta_metadatos}"
    )

    print("Archivos guardados correctamente.")


# ============================================================
# 11. VERIFICAR EL ENTORNO DE PYTORCH
# ============================================================

def verificar_pytorch():

    """
    Comprueba que PyTorch puede reconocer la GPU.

    En esta etapa no se realiza entrenamiento.
    """

    print("\n8. VERIFICANDO PYTORCH")
    print("-" * 60)

    print(
        f"Versión PyTorch: {torch.__version__}"
    )

    print(
        f"CUDA disponible: {torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():

        print(
            "GPU detectada: "
            f"{torch.cuda.get_device_name(0)}"
        )

    else:

        print(
            "Se utilizará CPU cuando se entrene el modelo."
        )


# ============================================================
# 12. PROCESO PRINCIPAL
# ============================================================

def main():

    """
    Coordina todas las actividades de la etapa 15.
    """

    print("=" * 65)

    print(
        "SMARTACADEMY"
    )

    print(
        "ETAPA 15 - PREPARACION DE DATOS PARA CNN 1D"
    )

    print("=" * 65)

    spark = None

    try:

        # Iniciar Spark.
        spark = crear_sesion_spark()

        print(
            "\nSESION SPARK INICIADA"
        )

        # Leer registros.
        datos = cargar_datos(spark)

        # Validar calidad.
        validar_datos(datos)

        # Construir tensores.
        X, y, grupos = construir_tensores(
            datos
        )

        # Dividir conjuntos por estudiante.
        (
            X_train, y_train, grupos_train,
            X_val, y_val, grupos_val,
            X_test, y_test, grupos_test
        ) = dividir_datos(
            X,
            y,
            grupos
        )

        # Normalizar utilizando únicamente entrenamiento.
        (
            X_train,
            X_val,
            X_test,
            medias,
            desviaciones
        ) = normalizar_datos(
            X_train,
            X_val,
            X_test
        )

        # Revisar distribución de las clases.
        validar_clases(
            y_train,
            y_val,
            y_test
        )

        # Guardar todos los resultados.
        guardar_datos(
            X_train,
            y_train,
            grupos_train,

            X_val,
            y_val,
            grupos_val,

            X_test,
            y_test,
            grupos_test,

            medias,
            desviaciones
        )

        # Verificar PyTorch y GPU.
        verificar_pytorch()

        print("\n" + "=" * 65)

        print(
            "ETAPA 15 FINALIZADA CORRECTAMENTE"
        )

        print(
            "Los datos están preparados para la CNN 1D."
        )

        print("=" * 65)

    except Exception as error:

        print("\nERROR EN LA ETAPA 15")

        print(
            f"Detalle: {error}"
        )

        raise

    finally:

        # Cerrar Spark incluso si ocurre algún error.
        if spark is not None:

            spark.stop()

            print(
                "\nSESION SPARK CERRADA"
            )


# ============================================================
# 13. PUNTO DE ENTRADA
# ============================================================

if __name__ == "__main__":

    main()
