
# ==============================================================
# PROYECTO: SMARTACADEMY
# ETAPA 12: GENERACIÓN DE DATOS ACADÉMICOS MASIVOS
#
# Archivo:
# p20generador_datos_masivos.py
#
# Objetivo:
# Generar 50,000 registros académicos sintéticos que posteriormente
# serán procesados mediante Apache Spark y utilizados para entrenar
# una red neuronal convolucional 1D con PyTorch.
#
# Estructura:
# 10,000 estudiantes x 5 materias = 50,000 registros.
#
# Cada registro contiene:
# - Tres calificaciones parciales.
# - Tres porcentajes de asistencia.
# - Promedio final.
# - Tendencia académica.
# - Resultado de aprobación.
#
# IMPORTANTE:
# Los datos son sintéticos y no representan expedientes reales.
# ==============================================================


# ==============================================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# ==============================================================

# NumPy permite generar números aleatorios y realizar cálculos.
import numpy as np

# Pandas permite construir y manipular tablas de datos.
import pandas as pd

# Path permite administrar rutas de archivos.
from pathlib import Path


# ==============================================================
# 2. CONFIGURACIÓN GENERAL
# ==============================================================

# Semilla para obtener resultados reproducibles.
# Si ejecutamos el programa nuevamente, obtendremos los mismos
# datos mientras no modifiquemos la configuración.
SEMILLA = 42

np.random.seed(SEMILLA)

# Cantidad de estudiantes que se simularán.
TOTAL_ESTUDIANTES = 10000

# Cada estudiante tendrá cinco materias.
MATERIAS = [
    "Bases de Datos",
    "Fundamentos de IA",
    "Programacion",
    "Matematicas",
    "Estadistica"
]

# Número de materias por estudiante.
TOTAL_MATERIAS = len(MATERIAS)

# Cantidad esperada de registros.
TOTAL_REGISTROS = TOTAL_ESTUDIANTES * TOTAL_MATERIAS

# Directorio principal del proyecto.
RUTA_PROYECTO = Path(__file__).resolve().parents[1]

# Directorio donde se guardarán los resultados.
RUTA_RESULTADOS = RUTA_PROYECTO / "resultados_bigdata"

# Crear el directorio si todavía no existe.
RUTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

# Archivos de salida.
ARCHIVO_CSV = RUTA_RESULTADOS / "smartacademy_50000.csv"

ARCHIVO_RESUMEN = (
    RUTA_RESULTADOS / "resumen_datos_masivos.csv"
)


# ==============================================================
# 3. DEFINICIÓN DE FUNCIONES
# ==============================================================

def limitar_valor(valor, minimo, maximo):
    """
    Mantiene un valor dentro de un intervalo.

    Por ejemplo:
    Una calificación no puede ser menor que 0 ni mayor que 10.
    """

    return np.clip(valor, minimo, maximo)


def generar_tendencia():
    """
    Selecciona el patrón de evolución académica.

    0 = Favorable
    1 = Estable
    2 = Descendente

    Se asigna un patrón inicial a cada combinación
    estudiante-materia.
    """

    return np.random.choice(
        [0, 1, 2],
        p=[0.40, 0.35, 0.25]
    )


def calcular_tendencia(parcial1, parcial2, parcial3):
    """
    Determina la tendencia con base en el cambio
    entre la primera y la tercera evaluación.

    Favorable:
        Incremento de al menos 1 punto.

    Descendente:
        Disminución de al menos 1 punto.

    Estable:
        Cambios menores a 1 punto.

    Esta regla genera etiquetas iniciales para el
    ejercicio de aprendizaje supervisado.
    """

    diferencia = parcial3 - parcial1

    if diferencia >= 1:
        return "Favorable"

    elif diferencia <= -1:
        return "Descendente"

    else:
        return "Estable"


# ==============================================================
# 4. GENERACIÓN DE UN REGISTRO ACADÉMICO
# ==============================================================

def generar_registro(id_estudiante, materia, habilidad):
    """
    Genera los datos de una materia para un estudiante.

    Parámetros:
    id_estudiante:
        Identificador del estudiante.

    materia:
        Nombre de la materia.

    habilidad:
        Nivel académico simulado del estudiante.

    Devuelve:
        Un diccionario con toda la información académica.
    """

    # ----------------------------------------------------------
    # 4.1 Seleccionar el patrón de evolución
    # ----------------------------------------------------------

    patron = generar_tendencia()

    # ----------------------------------------------------------
    # 4.2 Generar la primera calificación
    # ----------------------------------------------------------

    # La habilidad influye en la calificación inicial.
    parcial1 = np.random.normal(
        loc=habilidad,
        scale=0.8
    )

    parcial1 = limitar_valor(parcial1, 0, 10)

    # ----------------------------------------------------------
    # 4.3 Establecer la evolución de las calificaciones
    # ----------------------------------------------------------

    if patron == 0:

        # Tendencia favorable:
        # Las calificaciones aumentan progresivamente.
        cambio = np.random.uniform(1.1, 2.2)

        parcial2 = parcial1 + cambio * 0.50
        parcial3 = parcial1 + cambio

    elif patron == 1:

        # Tendencia estable:
        # Las calificaciones presentan cambios pequeños.
        cambio = np.random.uniform(-0.4, 0.4)

        parcial2 = parcial1 + cambio
        parcial3 = parcial1 + np.random.uniform(-0.4, 0.4)

    else:

        # Tendencia descendente:
        # Las calificaciones disminuyen progresivamente.
        cambio = np.random.uniform(1.1, 2.2)

        parcial2 = parcial1 - cambio * 0.50
        parcial3 = parcial1 - cambio

    # Limitar las calificaciones al intervalo permitido.
    parcial2 = limitar_valor(parcial2, 0, 10)
    parcial3 = limitar_valor(parcial3, 0, 10)

    # ----------------------------------------------------------
    # 4.4 Generar asistencia por parcial
    # ----------------------------------------------------------

    # La asistencia se relaciona parcialmente con la habilidad
    # y con el patrón académico simulado.

    asistencia_base = 65 + habilidad * 3.2

    if patron == 0:
        asistencia_base += 5

    elif patron == 2:
        asistencia_base -= 5

    asistencia_p1 = limitar_valor(
        np.random.normal(asistencia_base, 7),
        0,
        100
    )

    asistencia_p2 = limitar_valor(
        np.random.normal(asistencia_base, 7),
        0,
        100
    )

    asistencia_p3 = limitar_valor(
        np.random.normal(asistencia_base, 7),
        0,
        100
    )

    # ----------------------------------------------------------
    # 4.5 Calcular el promedio final
    # ----------------------------------------------------------

    promedio_final = (
        parcial1 + parcial2 + parcial3
    ) / 3

    # ----------------------------------------------------------
    # 4.6 Calcular la tendencia observada
    # ----------------------------------------------------------

    tendencia = calcular_tendencia(
        parcial1,
        parcial2,
        parcial3
    )

    # ----------------------------------------------------------
    # 4.7 Determinar el resultado académico
    # ----------------------------------------------------------

    # Se utiliza 7 como calificación mínima de aprobación
    # para este escenario simulado.
    resultado = (
        "Aprobado"
        if promedio_final >= 7
        else "No aprobado"
    )

    # ----------------------------------------------------------
    # 4.8 Construir el registro
    # ----------------------------------------------------------

    registro = {

        "id_estudiante": id_estudiante,

        "materia": materia,

        "parcial1": round(float(parcial1), 2),

        "parcial2": round(float(parcial2), 2),

        "parcial3": round(float(parcial3), 2),

        "asistencia_p1": round(float(asistencia_p1), 2),

        "asistencia_p2": round(float(asistencia_p2), 2),

        "asistencia_p3": round(float(asistencia_p3), 2),

        "promedio_final": round(float(promedio_final), 2),

        "tendencia": tendencia,

        "resultado": resultado
    }

    return registro


# ==============================================================
# 5. PROGRAMA PRINCIPAL
# ==============================================================

def main():

    print("\n" + "=" * 65)

    print("SMARTACADEMY")

    print("ETAPA 12 - GENERACIÓN DE DATOS ACADÉMICOS MASIVOS")

    print("=" * 65)

    # ----------------------------------------------------------
    # 5.1 Mostrar configuración
    # ----------------------------------------------------------

    print("\nCONFIGURACIÓN DEL PROCESO")

    print("-" * 65)

    print(f"Total de estudiantes: {TOTAL_ESTUDIANTES}")

    print(f"Materias por estudiante: {TOTAL_MATERIAS}")

    print(f"Registros esperados: {TOTAL_REGISTROS}")

    # ----------------------------------------------------------
    # 5.2 Generar habilidad académica por estudiante
    # ----------------------------------------------------------

    # Se genera un nivel de habilidad entre 3 y 10.
    # Este valor se utiliza para simular diferencias
    # individuales entre estudiantes.
    habilidades = np.random.uniform(
        3,
        10,
        TOTAL_ESTUDIANTES
    )

    # Lista que almacenará los registros.
    registros = []

    print("\nGENERANDO REGISTROS...")

    # ----------------------------------------------------------
    # 5.3 Recorrer estudiantes
    # ----------------------------------------------------------

    for numero_estudiante in range(
        1,
        TOTAL_ESTUDIANTES + 1
    ):

        # Construir un identificador permanente.
        id_estudiante = (
            f"IDGS-MASIVO-{numero_estudiante:05d}"
        )

        # Obtener habilidad académica.
        habilidad = habilidades[numero_estudiante - 1]

        # ------------------------------------------------------
        # 5.4 Generar las cinco materias del estudiante
        # ------------------------------------------------------

        for materia in MATERIAS:

            registro = generar_registro(
                id_estudiante,
                materia,
                habilidad
            )

            registros.append(registro)

        # Mostrar avance cada 1,000 estudiantes.
        if numero_estudiante % 1000 == 0:

            print(
                f"Estudiantes procesados: "
                f"{numero_estudiante:,}"
            )

    # ----------------------------------------------------------
    # 5.5 Construir DataFrame
    # ----------------------------------------------------------

    print("\nCONSTRUYENDO DATAFRAME...")

    df = pd.DataFrame(registros)

    # Crear un identificador único para cada registro.
    df.insert(
        0,
        "id_registro",
        range(1, len(df) + 1)
    )

    # ----------------------------------------------------------
    # 5.6 Validar cantidad de registros
    # ----------------------------------------------------------

    print("\nVALIDANDO INFORMACIÓN")

    print("-" * 65)

    print(f"Registros generados: {len(df):,}")

    print(f"Registros esperados: {TOTAL_REGISTROS:,}")

    # Comprobar que se generó la cantidad solicitada.
    assert len(df) == TOTAL_REGISTROS, (
        "ERROR: La cantidad de registros no coincide."
    )

    # Comprobar que no existen identificadores duplicados.
    assert df["id_registro"].is_unique, (
        "ERROR: Existen identificadores duplicados."
    )

    # Comprobar calificaciones.
    assert df[
        ["parcial1", "parcial2", "parcial3"]
    ].apply(
        lambda columna: columna.between(0, 10).all()
    ).all(), "ERROR: Calificaciones fuera de rango."

    # Comprobar asistencia.
    assert df[
        ["asistencia_p1", "asistencia_p2", "asistencia_p3"]
    ].apply(
        lambda columna: columna.between(0, 100).all()
    ).all(), "ERROR: Asistencias fuera de rango."

    # ----------------------------------------------------------
    # 5.7 Guardar archivo CSV
    # ----------------------------------------------------------

    print("\nGUARDANDO ARCHIVO CSV...")

    df.to_csv(
        ARCHIVO_CSV,
        index=False,
        encoding="utf-8"
    )

    print(f"Archivo generado: {ARCHIVO_CSV}")

    # ----------------------------------------------------------
    # 5.8 Generar resumen estadístico
    # ----------------------------------------------------------

    print("\nGENERANDO RESUMEN ESTADÍSTICO...")

    resumen_tendencia = (
        df["tendencia"]
        .value_counts()
        .rename_axis("tendencia")
        .reset_index(name="cantidad")
    )

    resumen_tendencia["porcentaje"] = (
        resumen_tendencia["cantidad"]
        / len(df)
        * 100
    ).round(2)

    resumen_tendencia.to_csv(
        ARCHIVO_RESUMEN,
        index=False,
        encoding="utf-8"
    )

    # ----------------------------------------------------------
    # 5.9 Mostrar resultados
    # ----------------------------------------------------------

    print("\nPRIMEROS CINCO REGISTROS")

    print("-" * 65)

    print(df.head().to_string(index=False))

    print("\nDISTRIBUCIÓN DE TENDENCIAS")

    print("-" * 65)

    print(resumen_tendencia.to_string(index=False))

    print("\nDISTRIBUCIÓN DE RESULTADOS")

    print("-" * 65)

    print(
        df["resultado"]
        .value_counts()
        .to_string()
    )

    print("\nESTADÍSTICAS DE CALIFICACIONES")

    print("-" * 65)

    print(
        df[
            [
                "parcial1",
                "parcial2",
                "parcial3",
                "promedio_final"
            ]
        ].describe().round(2)
    )

    print("\n" + "=" * 65)

    print("ETAPA 12 FINALIZADA CORRECTAMENTE")

    print("=" * 65)

    print(f"\nArchivo principal: {ARCHIVO_CSV}")

    print(f"Archivo resumen: {ARCHIVO_RESUMEN}")


# ==============================================================
# 6. PUNTO DE ENTRADA
# ==============================================================

# Esta condición ejecuta main() solamente cuando se ejecuta
# directamente este módulo.
if __name__ == "__main__":

    main()
