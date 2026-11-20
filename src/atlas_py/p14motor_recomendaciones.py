
# ============================================================
# PROYECTO: SMARTACADEMY
# ETAPA 5: MOTOR INTELIGENTE DE RECOMENDACIONES

# Archivo: p14motor_recomendaciones.py

#Ya tenemos los tres modelos necesarios para comenzar a construir el motor inteligente:

#Modelo	            Resultado
#Random Forest	    Predicción de aprobación
#K-Means        	Identificación de cuatro perfiles
#PCA            	Reducción de cuatro dimensiones a dos

#También tenemos guardados los modelos .joblib, por lo que podemos reutilizarlos sin entrenarlos nuevamente.

#Una observación sobre tus resultados

#La varianza acumulada de PCA fue del 51.72 %. Esto significa que la visualización bidimensional 
#conserva aproximadamente la mitad de la variabilidad original. Es útil para explorar los perfiles, 
#pero no debemos utilizarla como sustituto de las cuatro características originales para tomar decisiones.

# Objetivo:
# Integrar Random Forest, K-Means y PCA para analizar
# estudiantes y generar recomendaciones académicas.
# ============================================================

# ============================================================
# 1. IMPORTAR LIBRERÍAS
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd


# ============================================================
# 2. CONFIGURACIÓN
# ============================================================

DIRECTORIO_BASE = os.path.dirname(
    os.path.abspath(__file__)
)

DIRECTORIO_MODELOS = os.path.join(
    DIRECTORIO_BASE,
    "resultados_p10",
    "modelos"
)

ARCHIVO_RANDOM_FOREST = os.path.join(
    DIRECTORIO_MODELOS,
    "modelo_smartacademy.joblib"
)

ARCHIVO_KMEANS = os.path.join(
    DIRECTORIO_MODELOS,
    "modelo_kmeans_smartacademy.joblib"
)

ARCHIVO_PCA = os.path.join(
    DIRECTORIO_MODELOS,
    "modelo_pca_smartacademy.joblib"
)

DIRECTORIO_RESULTADOS = os.path.join(
    DIRECTORIO_BASE,
    "resultados_p10"
)

os.makedirs(
    DIRECTORIO_RESULTADOS,
    exist_ok=True
)

CARACTERISTICAS = [
    "horas_estudio",
    "asistencia",
    "actividades",
    "participacion"
]


# ============================================================
# 3. CARGAR LOS TRES MODELOS
# ============================================================

def cargar_modelos():

    print("\nCARGANDO MODELOS INTELIGENTES...")

    # Comprobar que existan los archivos.
    archivos = [
        ARCHIVO_RANDOM_FOREST,
        ARCHIVO_KMEANS,
        ARCHIVO_PCA
    ]

    for archivo in archivos:

        if not os.path.exists(archivo):

            raise FileNotFoundError(
                f"No se encontró el modelo: {archivo}"
            )

    # Recuperar los modelos.
    paquete_supervisado = joblib.load(
        ARCHIVO_RANDOM_FOREST
    )

    paquete_kmeans = joblib.load(
        ARCHIVO_KMEANS
    )

    paquete_pca = joblib.load(
        ARCHIVO_PCA
    )

    # Recuperar los objetos necesarios.
    modelo_supervisado = paquete_supervisado["modelo"]

    modelo_kmeans = paquete_kmeans["modelo"]

    escalador_kmeans = paquete_kmeans["escalador"]

    mapa_perfiles = paquete_kmeans["mapa_perfiles"]

    modelo_pca = paquete_pca["modelo_pca"]

    escalador_pca = paquete_pca["escalador"]

    print("Random Forest cargado.")
    print("K-Means cargado.")
    print("PCA cargado.")

    return {
        "supervisado": modelo_supervisado,
        "kmeans": modelo_kmeans,
        "escalador_kmeans": escalador_kmeans,
        "mapa_perfiles": mapa_perfiles,
        "pca": modelo_pca,
        "escalador_pca": escalador_pca
    }


# ============================================================
# 4. CAPTURAR LOS DATOS DEL ESTUDIANTE
# ============================================================

def capturar_estudiante():

    print("\nDATOS DEL ESTUDIANTE")

    print("-" * 45)

    # Solicitar información al usuario.
    horas = float(
        input("Horas de estudio(0-24)): ")
    )

    asistencia = float(
        input("Porcentaje de asistencia: ")
    )

    actividades = float(
        input("Número de actividades: ")
    )

    participacion = float(
        input("Nivel de participación (0-100): ")
    )

    # Validar los valores.
    if not np.isfinite([
        horas,
        asistencia,
        actividades,
        participacion
    ]).all():

        raise ValueError(
            "Los valores deben ser números finitos."
        )

    if not 0 <= horas <= 24:
        raise ValueError(
            "Las horas deben estar entre 0 y 24."
        )

    if not 0 <= asistencia <= 100:
        raise ValueError(
            "La asistencia debe estar entre 0 y 100."
        )

    if not 0 <= actividades <= 30:
        raise ValueError(
            "Las actividades deben estar entre 0 y 30."
        )

    if not 0 <= participacion <= 100:
        raise ValueError(
            "La participación debe estar entre 0 y 100."
        )

    # Crear DataFrame con el orden de columnas correcto.
    estudiante = pd.DataFrame([{
        "horas_estudio": horas,
        "asistencia": asistencia,
        "actividades": actividades,
        "participacion": participacion
    }])

    return estudiante


# ============================================================
# 5. REALIZAR PREDICCIÓN SUPERVISADA
# ============================================================

def predecir_aprobacion(estudiante, modelo):

    print("\nANÁLISIS DE APRENDIZAJE SUPERVISADO")

    # Obtener la predicción.
    prediccion = modelo.predict(estudiante)[0]

    # Obtener las probabilidades estimadas.
    probabilidades = modelo.predict_proba(
        estudiante
    )[0]

    # Identificar correctamente cada clase.
    clases = list(modelo.classes_)

    probabilidades_por_clase = {
        int(clase): float(probabilidad)
        for clase, probabilidad in zip(
            clases,
            probabilidades
        )
    }

    prob_no_aprueba = probabilidades_por_clase.get(
        0, 0.0
    )

    prob_aprueba = probabilidades_por_clase.get(
        1, 0.0
    )

    resultado = (
        "APROBACIÓN ESTIMADA"
        if prediccion == 1
        else "RIESGO DE NO APROBACIÓN ESTIMADO"
    )

    print("Resultado:", resultado)

    print(
        f"Probabilidad estimada de aprobar: "
        f"{prob_aprueba * 100:.2f}%"
    )

    print(
        f"Probabilidad estimada de no aprobar: "
        f"{prob_no_aprueba * 100:.2f}%"
    )

    return {
        "prediccion": int(prediccion),
        "resultado": resultado,
        "probabilidad_aprobar": prob_aprueba,
        "probabilidad_no_aprobar": prob_no_aprueba
    }


# ============================================================
# 6. IDENTIFICAR EL PERFIL CON K-MEANS
# ============================================================

def identificar_perfil(
    estudiante,
    modelo,
    escalador,
    mapa_perfiles
):

    print("\nANÁLISIS DE APRENDIZAJE NO SUPERVISADO")

    # Transformar los datos utilizando el escalador
    # que fue entrenado durante la etapa K-Means.
    datos_escalados = escalador.transform(
        estudiante
    )

    # Obtener el grupo al que pertenece el estudiante.
    cluster = int(
        modelo.predict(datos_escalados)[0]
    )

    # Recuperar su nombre descriptivo.
    perfil = mapa_perfiles[cluster]

    print("Cluster identificado:", cluster)

    print("Perfil académico:", perfil)

    return {
        "cluster": cluster,
        "perfil": perfil
    }


# ============================================================
# 7. PROYECTAR EL ESTUDIANTE MEDIANTE PCA
# ============================================================

def proyectar_pca(
    estudiante,
    modelo_pca,
    escalador
):

    print("\nANÁLISIS PCA")

    # Aplicar el escalador utilizado por PCA.
    datos_escalados = escalador.transform(
        estudiante
    )

    # Obtener PC1 y PC2.
    componentes = modelo_pca.transform(
        datos_escalados
    )

    pc1 = float(componentes[0, 0])

    pc2 = float(componentes[0, 1])

    print("Componente PC1:", round(pc1, 4))

    print("Componente PC2:", round(pc2, 4))

    return {
        "PC1": pc1,
        "PC2": pc2
    }


# ============================================================
# 8. MOTOR DE REGLAS DE RECOMENDACIÓN
# ============================================================

def generar_recomendaciones(
    estudiante,
    prediccion,
    perfil
):

    print("\nMOTOR DE RECOMENDACIONES")

    recomendaciones = []

    # Recuperar valores del estudiante.
    horas = float(
        estudiante["horas_estudio"].iloc[0]
    )

    asistencia = float(
        estudiante["asistencia"].iloc[0]
    )

    actividades = float(
        estudiante["actividades"].iloc[0]
    )

    participacion = float(
        estudiante["participacion"].iloc[0]
    )

    # --------------------------------------------------------
    # REGLA 1: HORAS DE ESTUDIO
    # --------------------------------------------------------

    if horas < 3:

        recomendaciones.append(
            "Establecer una rutina gradual de estudio "
            "y distribuir el tiempo durante la semana."
        )

    elif horas < 6:

        recomendaciones.append(
            "Incrementar progresivamente las horas "
            "de estudio y utilizar técnicas de repaso."
        )

    else:

        recomendaciones.append(
            "Mantener los hábitos de estudio y "
            "verificar que el tiempo se aproveche eficazmente."
        )

    # --------------------------------------------------------
    # REGLA 2: ASISTENCIA
    # --------------------------------------------------------

    if asistencia < 70:

        recomendaciones.append(
            "Revisar las causas de inasistencia y "
            "establecer un plan de seguimiento."
        )

    elif asistencia < 85:

        recomendaciones.append(
            "Mejorar la constancia de asistencia "
            "a las sesiones académicas."
        )

    else:

        recomendaciones.append(
            "Mantener el nivel de asistencia alcanzado."
        )

    # --------------------------------------------------------
    # REGLA 3: ACTIVIDADES
    # --------------------------------------------------------

    if actividades < 10:

        recomendaciones.append(
            "Priorizar la entrega de actividades "
            "y establecer objetivos semanales."
        )

    elif actividades < 20:

        recomendaciones.append(
            "Aumentar la regularidad en la realización "
            "de actividades académicas."
        )

    else:

        recomendaciones.append(
            "Mantener la constancia en la realización "
            "de actividades."
        )

    # --------------------------------------------------------
    # REGLA 4: PARTICIPACIÓN
    # --------------------------------------------------------

    if participacion < 40:

        recomendaciones.append(
            "Promover la participación mediante "
            "preguntas, ejercicios y trabajo colaborativo."
        )

    elif participacion < 70:

        recomendaciones.append(
            "Fortalecer la participación en clase "
            "y en actividades colaborativas."
        )

    else:

        recomendaciones.append(
            "Conservar la participación activa "
            "en las actividades académicas."
        )

    # --------------------------------------------------------
    # REGLA 5: PREDICCIÓN DEL MODELO
    # --------------------------------------------------------

    if prediccion["prediccion"] == 0:

        recomendaciones.append(
            "La predicción del modelo indica una posible "
            "dificultad académica. Se recomienda una "
            "revisión docente individual."
        )

    else:

        recomendaciones.append(
            "La predicción del modelo es favorable. "
            "Se recomienda mantener el seguimiento académico."
        )

    # --------------------------------------------------------
    # REGLA 6: PERFIL K-MEANS
    # --------------------------------------------------------

    recomendaciones.append(
        f"Considerar el perfil identificado por K-Means: "
        f"{perfil['perfil']}."
    )

    print("\nRECOMENDACIONES GENERADAS:")

    for numero, recomendacion in enumerate(
        recomendaciones,
        start=1
    ):

        print(f"{numero}. {recomendacion}")

    return recomendaciones


# ============================================================
# 9. GENERAR REPORTE DEL ESTUDIANTE
# ============================================================

def generar_reporte(
    estudiante,
    prediccion,
    perfil,
    pca,
    recomendaciones
):

    print("\nGENERANDO REPORTE...")

    reporte = {

        "horas_estudio": float(
            estudiante["horas_estudio"].iloc[0]
        ),

        "asistencia": float(
            estudiante["asistencia"].iloc[0]
        ),

        "actividades": float(
            estudiante["actividades"].iloc[0]
        ),

        "participacion": float(
            estudiante["participacion"].iloc[0]
        ),

        "prediccion": prediccion["prediccion"],

        "resultado": prediccion["resultado"],

        "probabilidad_aprobar":
            prediccion["probabilidad_aprobar"],

        "probabilidad_no_aprobar":
            prediccion["probabilidad_no_aprobar"],

        "cluster": perfil["cluster"],

        "perfil": perfil["perfil"],

        "PC1": pca["PC1"],

        "PC2": pca["PC2"],

        "recomendaciones": recomendaciones

    }

    # Convertir el reporte a DataFrame.
    reporte_df = pd.DataFrame([reporte])

    # Exportar a CSV.
    archivo = os.path.join(
        DIRECTORIO_RESULTADOS,
        "reporte_estudiante.csv"
    )

    reporte_df.to_csv(
        archivo,
        index=False
    )

    print("Reporte exportado:", archivo)

    return reporte


# ============================================================
# 10. MOSTRAR RESULTADO INTEGRAL
# ============================================================

def mostrar_resultado(reporte):

    print("\n" + "=" * 60)

    print("REPORTE INTELIGENTE SMARTACADEMY")

    print("=" * 60)

    print("\nRESULTADO DEL MODELO:")

    print(reporte["resultado"])

    print(
        "Probabilidad estimada de aprobar:",
        f"{reporte['probabilidad_aprobar'] * 100:.2f}%"
    )

    print("\nPERFIL ACADÉMICO:")

    print(reporte["perfil"])

    print("\nCOMPONENTES PCA:")

    print("PC1:", round(reporte["PC1"], 4))

    print("PC2:", round(reporte["PC2"], 4))

    print("\nRECOMENDACIONES:")

    for i, recomendacion in enumerate(
        reporte["recomendaciones"],
        start=1
    ):

        print(f"{i}. {recomendacion}")

    print("\n" + "=" * 60)

    print("ANÁLISIS FINALIZADO")

    print("=" * 60)

    print(
        "\nIMPORTANTE: los resultados son estimaciones "
        "y apoyos para el análisis docente. No sustituyen "
        "una evaluación académica individual."
    )


# ============================================================
# 11. FUNCIÓN PRINCIPAL
# ============================================================

def main():

    print("\n" + "=" * 60)

    print("SMARTACADEMY")

    print("ETAPA 5 - MOTOR DE RECOMENDACIONES")

    print("=" * 60)

    # Paso 1: cargar modelos.
    modelos = cargar_modelos()

    # Paso 2: capturar estudiante.
    estudiante = capturar_estudiante()

    # Paso 3: predecir aprobación.
    prediccion = predecir_aprobacion(
        estudiante,
        modelos["supervisado"]
    )

    # Paso 4: identificar perfil.
    perfil = identificar_perfil(
        estudiante,
        modelos["kmeans"],
        modelos["escalador_kmeans"],
        modelos["mapa_perfiles"]
    )

    # Paso 5: proyectar mediante PCA.
    pca = proyectar_pca(
        estudiante,
        modelos["pca"],
        modelos["escalador_pca"]
    )

    # Paso 6: generar recomendaciones.
    recomendaciones = generar_recomendaciones(
        estudiante,
        prediccion,
        perfil
    )

    # Paso 7: crear reporte.
    reporte = generar_reporte(
        estudiante,
        prediccion,
        perfil,
        pca,
        recomendaciones
    )

    # Paso 8: mostrar resultados.
    mostrar_resultado(reporte)


# ============================================================
# 12. EJECUTAR PROGRAMA
# ============================================================

if __name__ == "__main__":

    main()
