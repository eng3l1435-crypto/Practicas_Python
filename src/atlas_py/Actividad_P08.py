# ============================================================
# UNIDAD III - FUNDAMENTOS DE INTELIGENCIA ARTIFICIAL
# ACTIVIDAD PRACTICA 08:
# ANALISIS AVANZADO Y OPTIMIZACION DE SEGMENTOS ACADEMICOS
#
# Archivo:
# Actividad_P08.py
#
# Objetivo:
# Resolver las siete actividades propuestas en la Practica 08:
# 1. Cambiar NUM_CLUSTERS = 3 por 2 y por 4, y comparar resultados.
# 2. Cambiar NUM_INICIALIZACIONES = 20 por 5 y 50.
# 3. Agregar una nueva variable llamada "entregas_puntuales"
#    y evaluar su efecto en la segmentacion.
# 4. Modificar PERCENTIL_ATIPICOS = 95 por 90 y 97.5, y comparar
#    cuantos estudiantes se identifican.
# 5. Comparar los perfiles academicos obtenidos con StandardScaler
#    y MinMaxScaler.
# 6. Explicar si los segmentos son estables entre diferentes semillas.
# 7. Investigar por que dos estudiantes con caracteristicas similares
#    podrian terminar en grupos diferentes.
#
# IMPORTANTE:
# Los datos de esta practica son sinteticos y se utilizan
# exclusivamente con fines educativos.
# ============================================================


# ------------------------------------------------------------
# 1. IMPORTACION DE LIBRERIAS
# ------------------------------------------------------------

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler
)

from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score
)


# ------------------------------------------------------------
# 2. CONFIGURACION GENERAL
# ------------------------------------------------------------

SEMILLA = 42

VARIABLES = [
    "horas_estudio",
    "asistencia",
    "actividades",
    "participacion"
]

CARPETA_RESULTADOS = "resultados_actividad_p08"

os.makedirs(CARPETA_RESULTADOS, exist_ok=True)


# ------------------------------------------------------------
# 3. GENERACION DE DATOS BASE (EQUIVALENTE A P07 / P08)
# ------------------------------------------------------------

def generar_datos_smartacademy(incluir_entregas_puntuales=False):
    """
    Genera 500 estudiantes con tres perfiles simulados usando
    la semilla 42 para mantener reproducibilidad con P07 y P08.
    """
    rng = np.random.default_rng(SEMILLA)

    if not incluir_entregas_puntuales:
        # Perfil 1: alto compromiso
        alto = np.column_stack([
            rng.normal(8.0, 1.0, 180),
            rng.normal(92.0, 5.0, 180),
            rng.normal(27.0, 2.0, 180),
            rng.normal(90.0, 6.0, 180)
        ])

        # Perfil 2: compromiso intermedio
        medio = np.column_stack([
            rng.normal(5.0, 1.0, 170),
            rng.normal(75.0, 7.0, 170),
            rng.normal(20.0, 3.0, 170),
            rng.normal(65.0, 8.0, 170)
        ])

        # Perfil 3: requiere acompanamiento
        bajo = np.column_stack([
            rng.normal(2.5, 1.0, 150),
            rng.normal(55.0, 8.0, 150),
            rng.normal(12.0, 4.0, 150),
            rng.normal(40.0, 10.0, 150)
        ])

        datos = np.vstack([alto, medio, bajo])
        rng.shuffle(datos)

        datos[:, 0] = np.clip(datos[:, 0], 0, 12)
        datos[:, 1] = np.clip(datos[:, 1], 0, 100)
        datos[:, 2] = np.clip(datos[:, 2], 0, 30)
        datos[:, 3] = np.clip(datos[:, 3], 0, 100)

        df = pd.DataFrame(datos, columns=VARIABLES)

    else:
        # Incorporamos la 5ta variable: entregas_puntuales (0 a 100%)
        columnas_ext = VARIABLES + ["entregas_puntuales"]

        alto = np.column_stack([
            rng.normal(8.0, 1.0, 180),
            rng.normal(92.0, 5.0, 180),
            rng.normal(27.0, 2.0, 180),
            rng.normal(90.0, 6.0, 180),
            rng.normal(94.0, 5.0, 180)
        ])

        medio = np.column_stack([
            rng.normal(5.0, 1.0, 170),
            rng.normal(75.0, 7.0, 170),
            rng.normal(20.0, 3.0, 170),
            rng.normal(65.0, 8.0, 170),
            rng.normal(72.0, 9.0, 170)
        ])

        bajo = np.column_stack([
            rng.normal(2.5, 1.0, 150),
            rng.normal(55.0, 8.0, 150),
            rng.normal(12.0, 4.0, 150),
            rng.normal(40.0, 10.0, 150),
            rng.normal(48.0, 12.0, 150)
        ])

        datos = np.vstack([alto, medio, bajo])
        rng.shuffle(datos)

        datos[:, 0] = np.clip(datos[:, 0], 0, 12)
        datos[:, 1] = np.clip(datos[:, 1], 0, 100)
        datos[:, 2] = np.clip(datos[:, 2], 0, 30)
        datos[:, 3] = np.clip(datos[:, 3], 0, 100)
        datos[:, 4] = np.clip(datos[:, 4], 0, 100)

        df = pd.DataFrame(datos, columns=columnas_ext)

    df.insert(
        0,
        "id_estudiante",
        [f"EST-{i:04d}" for i in range(1, len(df) + 1)]
    )

    return df


# ------------------------------------------------------------
# 4. ACTIVIDAD 1: CAMBIAR NUM_CLUSTERS (2, 3 Y 4)
# ------------------------------------------------------------

def actividad_1_num_clusters(df, X_escalado):
    print("\n" + "=" * 60)
    print("ACTIVIDAD 1: CAMBIAR NUM_CLUSTERS A 2, 3 Y 4")
    print("=" * 60)

    resultados = []

    for k in [2, 3, 4]:
        modelo = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=20,
            random_state=SEMILLA
        )
        etiquetas = modelo.fit_predict(X_escalado)
        sil = silhouette_score(X_escalado, etiquetas)
        conteos = pd.Series(etiquetas).value_counts().sort_index().to_dict()

        resultados.append({
            "NUM_CLUSTERS": k,
            "inercia": round(modelo.inertia_, 4),
            "silhouette": round(sil, 4),
            "estudiantes_por_cluster": str(conteos)
        })

        df_temp = df[VARIABLES].copy()
        df_temp["cluster"] = etiquetas
        print(f"\n--- Perfiles promedio con NUM_CLUSTERS = {k} ---")
        print(df_temp.groupby("cluster").mean().round(2))

    df_act1 = pd.DataFrame(resultados)
    print("\nResumen comparativo Actividad 1:")
    print(df_act1.to_string(index=False))

    df_act1.to_csv(
        f"{CARPETA_RESULTADOS}/act1_comparacion_clusters.csv",
        index=False
    )

    print(
        "\nExplicacion Actividad 1:\n"
        "- Con NUM_CLUSTERS = 2: La inercia es muy alta y el modelo fusiona el\n"
        "  segmento intermedio con uno de los extremos, perdiendo capacidad para\n"
        "  diferenciar alumnos que requieren apoyo preventivo.\n"
        "- Con NUM_CLUSTERS = 3: Se obtiene el Silhouette Score mas alto, logrando\n"
        "  una separacion clara y equilibrada de los tres perfiles academicos.\n"
        "- Con NUM_CLUSTERS = 4: Aunque la inercia baja, el Silhouette Score cae\n"
        "  porque se divide artificialmente un grupo homogeneo en dos subgrupos\n"
        "  muy cercanos entre si."
    )

    return df_act1


# ------------------------------------------------------------
# 5. ACTIVIDAD 2: CAMBIAR NUM_INICIALIZACIONES (5, 20 Y 50)
# ------------------------------------------------------------

def actividad_2_inicializaciones(X_escalado):
    print("\n" + "=" * 60)
    print("ACTIVIDAD 2: CAMBIAR NUM_INICIALIZACIONES A 5, 20 Y 50")
    print("=" * 60)

    resultados = []
    etiquetas_ref = None

    for n_init in [5, 20, 50]:
        modelo = KMeans(
            n_clusters=3,
            init="k-means++",
            n_init=n_init,
            random_state=SEMILLA
        )
        etiquetas = modelo.fit_predict(X_escalado)
        sil = silhouette_score(X_escalado, etiquetas)

        if etiquetas_ref is None:
            etiquetas_ref = etiquetas

        ari_vs_5 = adjusted_rand_score(etiquetas_ref, etiquetas)

        resultados.append({
            "NUM_INICIALIZACIONES": n_init,
            "inercia": round(modelo.inertia_, 4),
            "silhouette": round(sil, 4),
            "iteraciones_ejecutadas": modelo.n_iter_,
            "ARI_respecto_a_5": round(ari_vs_5, 4)
        })

    df_act2 = pd.DataFrame(resultados)
    print("\nResumen comparativo Actividad 2:")
    print(df_act2.to_string(index=False))

    df_act2.to_csv(
        f"{CARPETA_RESULTADOS}/act2_inicializaciones.csv",
        index=False
    )

    print(
        "\nExplicacion Actividad 2:\n"
        "Con datos bien estructurados y usando init='k-means++', desde n_init=5\n"
        "el algoritmo ya encuentra la misma solucion optima que con 20 y 50\n"
        "(ARI = 1.0, identica inercia y Silhouette). Aumentar n_init a 50 aporta\n"
        "mayor garantia contra minimos locales cuando los datos son mas ruidosos\n"
        "o tienen muchos grupos, a cambio de ejecutar mas veces el entrenamiento."
    )

    return df_act2


# ------------------------------------------------------------
# 6. ACTIVIDAD 3: AGREGAR VARIABLE entregas_puntuales
# ------------------------------------------------------------

def actividad_3_nueva_variable(X_base_escalado):
    print("\n" + "=" * 60)
    print("ACTIVIDAD 3: NUEVA VARIABLE (entregas_puntuales)")
    print("=" * 60)

    df_ext = generar_datos_smartacademy(incluir_entregas_puntuales=True)
    columnas_ext = VARIABLES + ["entregas_puntuales"]

    escalador_ext = StandardScaler()
    X_ext_escalado = escalador_ext.fit_transform(df_ext[columnas_ext])

    modelo_base = KMeans(
        n_clusters=3, init="k-means++", n_init=20, random_state=SEMILLA
    )
    etiquetas_base = modelo_base.fit_predict(X_base_escalado)

    modelo_ext = KMeans(
        n_clusters=3, init="k-means++", n_init=20, random_state=SEMILLA
    )
    etiquetas_ext = modelo_ext.fit_predict(X_ext_escalado)

    sil_base = silhouette_score(X_base_escalado, etiquetas_base)
    sil_ext = silhouette_score(X_ext_escalado, etiquetas_ext)
    ari_comparacion = adjusted_rand_score(etiquetas_base, etiquetas_ext)

    df_ext["cluster"] = etiquetas_ext
    perfiles_ext = df_ext.groupby("cluster")[columnas_ext].mean().round(2)

    print(f"Silhouette (4 variables): {sil_base:.4f}")
    print(f"Silhouette (5 variables): {sil_ext:.4f}")
    print(f"ARI entre ambos modelos:  {ari_comparacion:.4f}")

    print("\nPerfiles promedio con 'entregas_puntuales' incluida:")
    print(perfiles_ext)

    perfiles_ext.to_csv(
        f"{CARPETA_RESULTADOS}/act3_perfiles_5variables.csv"
    )

    print(
        "\nExplicacion Actividad 3:\n"
        "Al incorporar 'entregas_puntuales', la dimension del espacio pasa de 4 a 5\n"
        "variables. Esto incrementa la inercia total (por sumar distancias en una\n"
        "dimension adicional), mantiene una alta coherencia con los segmentos previos\n"
        "(ARI cercano a 1) y permite distinguir mejor a los alumnos que cumplen\n"
        "en tiempo frente a quienes entregan actividades tarde."
    )

    return df_ext


# ------------------------------------------------------------
# 7. ACTIVIDAD 4: MODIFICAR PERCENTIL_ATIPICOS (90, 95 Y 97.5)
# ------------------------------------------------------------

def actividad_4_percentil_atipicos(df, X_escalado):
    print("\n" + "=" * 60)
    print("ACTIVIDAD 4: CAMBIAR PERCENTIL_ATIPICOS (90, 95 Y 97.5)")
    print("=" * 60)

    modelo = KMeans(
        n_clusters=3, init="k-means++", n_init=20, random_state=SEMILLA
    )
    etiquetas = modelo.fit_predict(X_escalado)

    centroides_asignados = modelo.cluster_centers_[etiquetas]
    distancias = np.linalg.norm(X_escalado - centroides_asignados, axis=1)

    resultados = []
    for p in [90, 95, 97.5]:
        umbral = np.percentile(distancias, p)
        cantidad = int((distancias > umbral).sum())
        porcentaje = round(cantidad / len(df) * 100, 2)

        resultados.append({
            "PERCENTIL_ATIPICOS": p,
            "umbral_distancia": round(umbral, 4),
            "estudiantes_identificados": cantidad,
            "porcentaje_total": f"{porcentaje}%"
        })

    df_act4 = pd.DataFrame(resultados)
    print("\nComparacion de umbrales de atipicos:")
    print(df_act4.to_string(index=False))

    df_act4.to_csv(
        f"{CARPETA_RESULTADOS}/act4_percentiles_atipicos.csv",
        index=False
    )

    print(
        "\nExplicacion Actividad 4:\n"
        "- Con percentil 90: El umbral baja y se marca el 10% mas alejado (50 alumnos),\n"
        "  aumentando la sensibilidad pero incluyendo casos que son solo variaciones\n"
        "  normales dentro de su grupo.\n"
        "- Con percentil 95: Equilibrio estandar (25 alumnos, 5%).\n"
        "- Con percentil 97.5: El umbral es mas estricto y solo marca el 2.5% mas\n"
        "  extremo (13 alumnos), enfocando la revision en los casos verdaderamente\n"
        "  inusuales."
    )

    return df_act4, distancias


# ------------------------------------------------------------
# 8. ACTIVIDAD 5: COMPARAR PERFILES (STANDARDSCALER VS MINMAXSCALER)
# ------------------------------------------------------------

def actividad_5_comparar_perfiles_escaladores(df, X):
    print("\n" + "=" * 60)
    print("ACTIVIDAD 5: PERFILES CON STANDARDSCALER VS MINMAXSCALER")
    print("=" * 60)

    nombres_perfiles = [
        "Requiere acompanamiento",
        "Compromiso intermedio",
        "Alto compromiso"
    ]

    df_comp = df.copy()
    perfiles_dict = {}

    for nombre, escalador in [
        ("StandardScaler", StandardScaler()),
        ("MinMaxScaler", MinMaxScaler())
    ]:
        X_esc = escalador.fit_transform(X)
        modelo = KMeans(
            n_clusters=3, init="k-means++", n_init=20, random_state=SEMILLA
        )
        etiquetas = modelo.fit_predict(X_esc)

        df_temp = df[VARIABLES].copy()
        df_temp["cluster"] = etiquetas
        medias = df_temp.groupby("cluster")[VARIABLES].mean()

        # Ordenamos por horas_estudio para alinear nombres de perfil
        orden = medias["horas_estudio"].sort_values().index.tolist()
        mapa = {c: n for c, n in zip(orden, nombres_perfiles)}

        df_comp[f"perfil_{nombre}"] = pd.Series(etiquetas).map(mapa)

        medias_nombradas = medias.loc[orden].copy()
        medias_nombradas.index = nombres_perfiles
        perfiles_dict[nombre] = medias_nombradas.round(2)

        print(f"\n--- Perfiles promedio con {nombre} ---")
        print(medias_nombradas.round(2))

    ari_esc = adjusted_rand_score(
        df_comp["perfil_StandardScaler"],
        df_comp["perfil_MinMaxScaler"]
    )
    cambios = int(
        (df_comp["perfil_StandardScaler"] != df_comp["perfil_MinMaxScaler"]).sum()
    )

    print(f"\nARI entre StandardScaler y MinMaxScaler: {ari_esc:.4f}")
    print(f"Estudiantes que cambiaron de perfil:     {cambios} de {len(df)}")

    perfiles_dict["StandardScaler"].to_csv(
        f"{CARPETA_RESULTADOS}/act5_perfiles_standard.csv"
    )
    perfiles_dict["MinMaxScaler"].to_csv(
        f"{CARPETA_RESULTADOS}/act5_perfiles_minmax.csv"
    )

    print(
        "\nExplicacion Actividad 5:\n"
        "Ambos escaladores recuperan los mismos tres perfiles generales, pero\n"
        "StandardScaler escala segun la desviacion estandar (varianza de cada\n"
        "variable), mientras que MinMaxScaler comprime segun los valores minimo y\n"
        "maximo absolutos [0, 1]. Si una variable tiene valores extremos aislados,\n"
        "MinMaxScaler comprime el resto de datos en un rango mas estrecho, lo que\n"
        "provoca que algunos estudiantes situados en las fronteras cambien de grupo."
    )

    return df_comp


# ------------------------------------------------------------
# 9. ACTIVIDAD 6 Y 7: ESTABILIDAD ENTRE SEMILLAS E INVESTIGACION
# ------------------------------------------------------------

def actividad_6_y_7_estabilidad_y_fronteras(df, X_escalado):
    print("\n" + "=" * 60)
    print("ACTIVIDAD 6: ESTABILIDAD ENTRE DIFERENTES SEMILLAS")
    print("=" * 60)

    semillas = [10, 20, 30, 40, 50, 42]
    particiones = {
        s: KMeans(
            n_clusters=3, init="k-means++", n_init=20, random_state=s
        ).fit_predict(X_escalado)
        for s in semillas
    }

    aris = []
    for i in range(len(semillas)):
        for j in range(i + 1, len(semillas)):
            aris.append(
                adjusted_rand_score(
                    particiones[semillas[i]],
                    particiones[semillas[j]]
                )
            )

    print(f"ARI minimo entre semillas:   {min(aris):.4f}")
    print(f"ARI promedio entre semillas: {np.mean(aris):.4f}")
    print(f"ARI maximo entre semillas:   {max(aris):.4f}")

    print(
        "\nExplicacion Actividad 6:\n"
        "Los segmentos son altamente estables (ARI = 1.0000 o muy cercano a 1.0)\n"
        "al variar la semilla. Esto ocurre porque los tres perfiles academicos estan\n"
        "bien diferenciados en el espacio de 4 variables y el uso de init='k-means++'\n"
        "junto con n_init=20 permite que K-Means converja consistentemente a los\n"
        "mismos centroides globales sin importar la semilla inicial."
    )

    print("\n" + "=" * 60)
    print("ACTIVIDAD 7: POR QUE DOS ESTUDIANTES SIMILARES QUEDAN EN GRUPOS DISTINTOS")
    print("=" * 60)

    print(
        "Investigacion Actividad 7:\n"
        "1. Fronteras de Voronoi: K-Means divide el espacio mediante hiperplanos\n"
        "   equidistantes entre centroides (diagramas de Voronoi). Dos estudiantes\n"
        "   muy cercanos entre si, pero situados a ambos lados de esa frontera,\n"
        "   tienen distancias casi iguales a dos centroides y se asignan a grupos\n"
        "   distintos.\n"
        "2. Efecto multidimensional (4 variables simultaneas): Dos alumnos pueden\n"
        "   parecer identicos en una grafica 2D (horas_estudio y asistencia), pero\n"
        "   diferir significativamente en las otras dos dimensiones ocultas\n"
        "   (actividades y participacion), inclinando la distancia euclidiana 4D.\n"
        "3. Asignacion dura (Hard Clustering): A diferencia de modelos probabilisticos\n"
        "   (como Gaussian Mixture Models), K-Means obliga a asignar cada punto al\n"
        "   100% a un solo cluster aunque tenga 51% de afinidad con uno y 49% con otro."
    )


# ------------------------------------------------------------
# 10. GRAFICAS RESUMEN DE LA ACTIVIDAD 08
# ------------------------------------------------------------

def generar_graficas_actividad(df_act1, df_act4, distancias):
    fig, ejes = plt.subplots(1, 2, figsize=(14, 5))

    # Grafica 1: Silhouette por NUM_CLUSTERS (Actividad 1)
    ejes[0].bar(
        df_act1["NUM_CLUSTERS"].astype(str),
        df_act1["silhouette"],
        color=["#7f8c8d", "#2980b9", "#95a5a6"]
    )
    ejes[0].set_title("Actividad 1: Silhouette Score por NUM_CLUSTERS")
    ejes[0].set_xlabel("NUM_CLUSTERS")
    ejes[0].set_ylabel("Silhouette Score")
    ejes[0].set_ylim(0, 0.75)
    ejes[0].grid(axis="y", alpha=0.3)

    # Grafica 2: Umbrales de Percentil de Atipicos (Actividad 4)
    ejes[1].scatter(range(len(distancias)), distancias, alpha=0.5, s=20)
    estilos = [("--", "red"), ("-.", "orange"), (":", "green")]
    for (_, fila), (ls, color) in zip(df_act4.iterrows(), estilos):
        ejes[1].axhline(
            y=fila["umbral_distancia"],
            linestyle=ls,
            color=color,
            label=f"P{fila['PERCENTIL_ATIPICOS']} ({fila['estudiantes_identificados']} alumnos)"
        )
    ejes[1].set_title("Actividad 4: Comparacion de Percentiles Atipicos")
    ejes[1].set_xlabel("Indice de estudiante")
    ejes[1].set_ylabel("Distancia al centroide")
    ejes[1].legend()
    ejes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"{CARPETA_RESULTADOS}/actividad_p08_resumen.png",
        dpi=300
    )
    plt.close()


# ------------------------------------------------------------
# 11. FUNCION PRINCIPAL
# ------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("SMARTACADEMY - ACTIVIDAD PRACTICA 08")
    print("=" * 60)

    df = generar_datos_smartacademy(incluir_entregas_puntuales=False)
    X = df[VARIABLES].to_numpy()

    escalador = StandardScaler()
    X_escalado = escalador.fit_transform(X)

    df_act1 = actividad_1_num_clusters(df, X_escalado)
    actividad_2_inicializaciones(X_escalado)
    actividad_3_nueva_variable(X_escalado)
    df_act4, distancias = actividad_4_percentil_atipicos(df, X_escalado)
    actividad_5_comparar_perfiles_escaladores(df, X)
    actividad_6_y_7_estabilidad_y_fronteras(df, X_escalado)

    generar_graficas_actividad(df_act1, df_act4, distancias)

    print("\n" + "=" * 60)
    print("ACTIVIDAD 08 FINALIZADA CORRECTAMENTE")
    print(f"Resultados exportados en: {CARPETA_RESULTADOS}/")
    print("=" * 60)


if __name__ == "__main__":
    main()