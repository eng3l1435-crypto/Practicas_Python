# ================================================================
# PROYECTO: SMARTACADEMY
# ACTIVIDAD PRACTICA 07: SEGMENTACION DE ESTUDIANTES CON K-MEANS
#
# Archivo:
# Actividad_P07.py
#
# Objetivo:
# Resolver las cuatro actividades propuestas en la Practica 07:
# - Actividad 1: Cambiar el numero de grupos (K = 2, 3, 4 y 5),
#                comparar las agrupaciones y explicar los cambios.
# - Actividad 2: Incorporar una nueva variable ("calificacion_parcial")
#                en la generacion de datos, el DataFrame, la seleccion
#                de variables, los centroides y la exportacion.
# - Actividad 3: Comparar K-Means con datos originales sin escalar
#                frente a datos estandarizados con StandardScaler.
# - Actividad 4: Analizar e investigar el efecto de los valores
#                atipicos (outliers) en la ubicacion de los centroides.
#
# IMPORTANTE:
# Los datos de esta practica son sinteticos y se utilizan
# exclusivamente con fines educativos.
# ================================================================


# ================================================================
# 1. IMPORTAR LAS BIBLIOTECAS
# ================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# StandardScaler permite estandarizar las variables.
from sklearn.preprocessing import StandardScaler

# KMeans es el algoritmo de agrupamiento.
from sklearn.cluster import KMeans

# silhouette_score permite evaluar la calidad de los grupos.
from sklearn.metrics import silhouette_score


# ================================================================
# 2. CONFIGURACION GENERAL Y DATOS BASE (IGUAL A P07)
# ================================================================

# Utilizamos la misma semilla de la Practica 07 para reproducibilidad.
rng = np.random.default_rng(42)

# Cantidad de estudiantes por perfil simulado.
cantidad_estudiantes = [180, 170, 150]

# Perfil simulado 1: estudiantes con alto compromiso
perfil_alto = np.column_stack([
    rng.normal(8.0, 1.0, cantidad_estudiantes[0]),
    rng.normal(92, 5, cantidad_estudiantes[0]),
    rng.normal(27, 2, cantidad_estudiantes[0]),
    rng.normal(90, 6, cantidad_estudiantes[0])
])

# Perfil simulado 2: estudiantes con compromiso medio
perfil_medio = np.column_stack([
    rng.normal(5.0, 1.0, cantidad_estudiantes[1]),
    rng.normal(75, 7, cantidad_estudiantes[1]),
    rng.normal(20, 3, cantidad_estudiantes[1]),
    rng.normal(65, 8, cantidad_estudiantes[1])
])

# Perfil simulado 3: estudiantes que requieren apoyo
perfil_bajo = np.column_stack([
    rng.normal(2.5, 1.0, cantidad_estudiantes[2]),
    rng.normal(55, 8, cantidad_estudiantes[2]),
    rng.normal(12, 4, cantidad_estudiantes[2]),
    rng.normal(40, 10, cantidad_estudiantes[2])
])

# Unificamos y mezclamos los registros.
datos_base = np.vstack([
    perfil_alto,
    perfil_medio,
    perfil_bajo
])
rng.shuffle(datos_base)

# Creamos el DataFrame base de 4 variables.
df_base = pd.DataFrame(
    datos_base,
    columns=[
        "horas_estudio",
        "asistencia",
        "actividades",
        "participacion"
    ]
)

# Ajustamos a rangos realistas igual que en P07.
df_base["horas_estudio"] = df_base["horas_estudio"].clip(0, 12)
df_base["asistencia"] = df_base["asistencia"].clip(0, 100)
df_base["actividades"] = df_base["actividades"].clip(0, 30)
df_base["participacion"] = df_base["participacion"].clip(0, 100)

# Escalamos los datos base para las Actividades 1 y 3.
X_base = df_base[
    ["horas_estudio", "asistencia", "actividades", "participacion"]
]
escalador_base = StandardScaler()
X_base_escalado = escalador_base.fit_transform(X_base)

print("\n==========================================")
print("SMARTACADEMY")
print("ACTIVIDAD P07: SEGMENTACION CON K-MEANS")
print("==========================================")


# ================================================================
# 3. ACTIVIDAD 1: CAMBIAR EL NUMERO DE GRUPOS (K)
# ================================================================

print("\n==========================================")
print("ACTIVIDAD 1: COMPARACION CON DIFERENTES K")
print("==========================================")

valores_k_prueba = [2, 3, 4, 5]
resumen_k = []

for k in valores_k_prueba:

    modelo_k = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    etiquetas_k = modelo_k.fit_predict(X_base_escalado)
    inercia_k = modelo_k.inertia_
    sil_k = silhouette_score(X_base_escalado, etiquetas_k)

    # Contamos cuantos estudiantes quedaron en cada cluster.
    conteos = pd.Series(etiquetas_k).value_counts().sort_index().to_dict()

    resumen_k.append({
        "K": k,
        "inercia": round(inercia_k, 2),
        "silhouette_score": round(sil_k, 4),
        "distribucion_estudiantes": str(conteos)
    })

    # Mostramos los promedios por grupo para analizar como se dividen.
    df_temp = df_base.copy()
    df_temp["cluster"] = etiquetas_k
    promedios_k = df_temp.groupby("cluster").mean().round(2)

    print(f"\n--- Configuracion con K = {k} ---")
    print(f"Inercia: {inercia_k:.2f} | Silhouette Score: {sil_k:.4f}")
    print(promedios_k)

df_resumen_k = pd.DataFrame(resumen_k)

print("\nTabla resumen Actividad 1:")
print(df_resumen_k.to_string(index=False))

print(
    "\nExplicacion Actividad 1:\n"
    "- Con K=2: El algoritmo fusiona dos perfiles en uno solo (generalmente junta\n"
    "  a los estudiantes de compromiso intermedio con uno de los extremos), obteniendo\n"
    "  una inercia alta y perdiendo detalle para intervenir de forma diferenciada.\n"
    "- Con K=3: Coincide con la estructura real de los datos y alcanza el Silhouette\n"
    "  Score mas alto, separando claramente los perfiles bajo, intermedio y alto.\n"
    "- Con K=4 y K=5: La inercia sigue bajando, pero el Silhouette Score disminuye\n"
    "  porque K-Means comienza a subdividir artificialmente grupos que ya eran\n"
    "  cohesivos en subgrupos muy cercanos entre si."
)


# ================================================================
# 4. ACTIVIDAD 2: INCORPORAR UNA NUEVA VARIABLE
# ================================================================

print("\n==========================================")
print("ACTIVIDAD 2: NUEVA VARIABLE (calificacion_parcial)")
print("==========================================")

# Generamos nuevamente los datos incorporando la 5ta variable:
# calificacion_parcial (escala de 0 a 100).
rng_act2 = np.random.default_rng(42)

perfil_alto_ext = np.column_stack([
    rng_act2.normal(8.0, 1.0, cantidad_estudiantes[0]),
    rng_act2.normal(92, 5, cantidad_estudiantes[0]),
    rng_act2.normal(27, 2, cantidad_estudiantes[0]),
    rng_act2.normal(90, 6, cantidad_estudiantes[0]),
    rng_act2.normal(91, 5, cantidad_estudiantes[0])   # Nueva variable
])

perfil_medio_ext = np.column_stack([
    rng_act2.normal(5.0, 1.0, cantidad_estudiantes[1]),
    rng_act2.normal(75, 7, cantidad_estudiantes[1]),
    rng_act2.normal(20, 3, cantidad_estudiantes[1]),
    rng_act2.normal(65, 8, cantidad_estudiantes[1]),
    rng_act2.normal(74, 7, cantidad_estudiantes[1])   # Nueva variable
])

perfil_bajo_ext = np.column_stack([
    rng_act2.normal(2.5, 1.0, cantidad_estudiantes[2]),
    rng_act2.normal(55, 8, cantidad_estudiantes[2]),
    rng_act2.normal(12, 4, cantidad_estudiantes[2]),
    rng_act2.normal(40, 10, cantidad_estudiantes[2]),
    rng_act2.normal(52, 9, cantidad_estudiantes[2])   # Nueva variable
])

datos_ext = np.vstack([
    perfil_alto_ext,
    perfil_medio_ext,
    perfil_bajo_ext
])
rng_act2.shuffle(datos_ext)

# 1) Agregamos la nueva caracteristica al DataFrame.
columnas_ext = [
    "horas_estudio",
    "asistencia",
    "actividades",
    "participacion",
    "calificacion_parcial"
]

df_ext = pd.DataFrame(datos_ext, columns=columnas_ext)

# Ajustamos rangos realistas.
df_ext["horas_estudio"] = df_ext["horas_estudio"].clip(0, 12)
df_ext["asistencia"] = df_ext["asistencia"].clip(0, 100)
df_ext["actividades"] = df_ext["actividades"].clip(0, 30)
df_ext["participacion"] = df_ext["participacion"].clip(0, 100)
df_ext["calificacion_parcial"] = df_ext["calificacion_parcial"].clip(0, 100)

# 2) Seleccion de variables (incluyendo la nueva).
X_ext = df_ext[columnas_ext]

escalador_ext = StandardScaler()
X_ext_escalado = escalador_ext.fit_transform(X_ext)

modelo_ext = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=20
)

df_ext["cluster"] = modelo_ext.fit_predict(X_ext_escalado)

# Asignamos nombres descriptivos ordenando por horas de estudio.
perfiles_ext = df_ext.groupby("cluster")[columnas_ext].mean().round(2)
orden_ext = perfiles_ext["horas_estudio"].sort_values().index.tolist()
nombres_perfiles = [
    "Requiere acompanamiento",
    "Compromiso intermedio",
    "Alto compromiso"
]
mapa_ext = {c: n for c, n in zip(orden_ext, nombres_perfiles)}
df_ext["perfil"] = df_ext["cluster"].map(mapa_ext)

# 3) Recuperamos los centroides en escala original (5 variables).
centroides_ext_orig = escalador_ext.inverse_transform(
    modelo_ext.cluster_centers_
)
centroides_ext_df = pd.DataFrame(
    centroides_ext_orig,
    columns=columnas_ext
).round(2)

print("\nPerfiles promedio con 5 variables:")
print(perfiles_ext)

print("\nCentroides en escala original (5 variables):")
print(centroides_ext_df)

print(
    f"\nSilhouette Score con 5 variables: "
    f"{silhouette_score(X_ext_escalado, df_ext['cluster']):.4f}"
)

# 4) Exportacion de resultados con la nueva variable.
df_ext.to_csv(
    "actividad_p07_estudiantes_5vars.csv",
    index=False,
    encoding="utf-8-sig"
)
centroides_ext_df.to_csv(
    "actividad_p07_centroides_5vars.csv",
    index=False,
    encoding="utf-8-sig"
)


# ================================================================
# 5. ACTIVIDAD 3: DATOS SIN ESCALAR VS DATOS ESTANDARIZADOS
# ================================================================

print("\n==========================================")
print("ACTIVIDAD 3: SIN ESCALAR VS STANDARDSCALER")
print("==========================================")

# Modelo A: Entrenado directamente con X_base (sin escalar).
modelo_sin_escalar = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=20
)
etiquetas_sin_escalar = modelo_sin_escalar.fit_predict(X_base)

# Modelo B: Entrenado con X_base_escalado (StandardScaler).
modelo_con_escalar = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=20
)
etiquetas_con_escalar = modelo_con_escalar.fit_predict(X_base_escalado)

# Evaluamos cada modelo en su propio espacio de caracteristicas.
sil_sin = silhouette_score(X_base, etiquetas_sin_escalar)
sil_con = silhouette_score(X_base_escalado, etiquetas_con_escalar)

# Tabla cruzada para ver cuantos estudiantes cambian de asignacion.
df_comp_escalado = df_base.copy()
df_comp_escalado["cluster_sin_escalar"] = etiquetas_sin_escalar
df_comp_escalado["cluster_con_escalar"] = etiquetas_con_escalar

# Estandarizamos nombres de grupo en ambos modelos para comparar justo.
ord_sin = (
    df_comp_escalado.groupby("cluster_sin_escalar")["horas_estudio"]
    .mean().sort_values().index.tolist()
)
ord_con = (
    df_comp_escalado.groupby("cluster_con_escalar")["horas_estudio"]
    .mean().sort_values().index.tolist()
)

df_comp_escalado["perfil_sin_escalar"] = df_comp_escalado[
    "cluster_sin_escalar"
].map({c: n for c, n in zip(ord_sin, nombres_perfiles)})

df_comp_escalado["perfil_con_escalar"] = df_comp_escalado[
    "cluster_con_escalar"
].map({c: n for c, n in zip(ord_con, nombres_perfiles)})

coincidencias = (
    df_comp_escalado["perfil_sin_escalar"]
    == df_comp_escalado["perfil_con_escalar"]
).sum()
diferencias = len(df_comp_escalado) - coincidencias

print(f"Silhouette Score (Sin escalar):   {sil_sin:.4f}")
print(f"Silhouette Score (Con escalar):   {sil_con:.4f}")
print(f"Estudiantes en el mismo perfil:   {coincidencias} de {len(df_comp_escalado)}")
print(f"Estudiantes que cambiaron grupo:  {diferencias}")

print("\nMatriz de comparacion (Sin escalar vs Con escalar):")
print(
    pd.crosstab(
        df_comp_escalado["perfil_sin_escalar"],
        df_comp_escalado["perfil_con_escalar"]
    )
)

print(
    "\nExplicacion Actividad 3:\n"
    "K-Means agrupa registros calculando distancias euclidianas. Sin escalar,\n"
    "'asistencia' y 'participacion' (rango 0-100) dominan casi por completo el\n"
    "calculo de distancia, mientras que 'horas_estudio' (rango 0-12) aporta muy poco\n"
    "peso numerico. Al aplicar StandardScaler, todas las variables tienen media 0\n"
    "y desviacion estandar 1, logrando que las cuatro variables contribuyan de\n"
    "forma equitativa y reclasificando a los estudiantes en los limites de grupo."
)


# ================================================================
# 6. ACTIVIDAD 4: DEMOSTRACION E INVESTIGACION DE OUTLIERS
# ================================================================

print("\n==========================================")
print("ACTIVIDAD 4: EFECTO DE VALORES ATIPICOS (OUTLIERS)")
print("==========================================")

# Creamos una copia con 5 registros atipicos extremos (outliers)
# para observar como se desplazan los centroides.
df_outliers = df_base.copy()

registros_atipicos = pd.DataFrame({
    "horas_estudio": [12.0, 12.0, 0.0, 12.0, 11.5],
    "asistencia": [10.0, 15.0, 100.0, 5.0, 12.0],
    "actividades": [1.0, 2.0, 30.0, 0.0, 2.0],
    "participacion": [5.0, 8.0, 100.0, 3.0, 6.0]
})

df_outliers = pd.concat([df_outliers, registros_atipicos], ignore_index=True)

esc_out = StandardScaler()
X_out_esc = esc_out.fit_transform(df_outliers)

modelo_out = KMeans(n_clusters=3, random_state=42, n_init=20)
modelo_out.fit(X_out_esc)

centroides_limpios = pd.DataFrame(
    escalador_base.inverse_transform(
        modelo_con_escalar.cluster_centers_
    ),
    columns=X_base.columns
).sort_values("horas_estudio").reset_index(drop=True).round(2)

centroides_con_out = pd.DataFrame(
    esc_out.inverse_transform(
        modelo_out.cluster_centers_
    ),
    columns=X_base.columns
).sort_values("horas_estudio").reset_index(drop=True).round(2)

print("\nCentroides SIN outliers:")
print(centroides_limpios)

print("\nCentroides CON outliers extremos:")
print(centroides_con_out)

print(
    "\nInvestigacion Actividad 4 (Efecto de los outliers en K-Means):\n"
    "1. Sensibilidad de la media: K-Means calcula cada centroide como el promedio\n"
    "   aritmetico de los puntos asignados al cluster. Como la media no es robusta,\n"
    "   pocos valores extremos 'jalan' al centroide fuera del centro real del grupo.\n"
    "2. Distorsion de fronteras: Al desplazarse el centroide, estudiantes normales\n"
    "   cercanos al borde pueden quedar mal asignados a otro cluster.\n"
    "3. Aumento de inercia y falsos clusters: Si se elige un K mayor, K-Means puede\n"
    "   dedicar un cluster exclusivo para aislar a los outliers en lugar de segmentar\n"
    "   perfiles academicos utiles.\n"
    "4. Soluciones habituales: Detectar y tratar outliers antes de entrenar (rango\n"
    "   intercuartilico IQR o Z-score), usar RobustScaler o emplear K-Medoids (PAM),\n"
    "   que usa registros reales (medoides) en lugar de promedios."
)


# ================================================================
# 7. GRAFICA COMPARATIVA DE LAS ACTIVIDADES
# ================================================================

fig, ejes = plt.subplots(1, 2, figsize=(14, 5))

# --- Subgrafica 1: Efecto de K (Actividad 1) ---
ejes[0].plot(
    df_resumen_k["K"],
    df_resumen_k["silhouette_score"],
    marker="o",
    linewidth=2,
    color="tab:blue"
)
ejes[0].set_title("Actividad 1: Silhouette Score segun K")
ejes[0].set_xlabel("Numero de grupos (K)")
ejes[0].set_ylabel("Silhouette Score")
ejes[0].set_xticks(valores_k_prueba)
ejes[0].grid(True, linestyle="--", alpha=0.4)

# --- Subgrafica 2: Sin escalar vs Con escalar (Actividad 3) ---
for perfil in nombres_perfiles:
    sub = df_comp_escalado[df_comp_escalado["perfil_con_escalar"] == perfil]
    ejes[1].scatter(
        sub["horas_estudio"],
        sub["asistencia"],
        label=perfil,
        alpha=0.65
    )

ejes[1].set_title("Actividad 3: Segmentacion Estandarizada (K=3)")
ejes[1].set_xlabel("Horas de estudio")
ejes[1].set_ylabel("Asistencia (%)")
ejes[1].legend()
ejes[1].grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("actividad_p07_grafica_resultados.png", dpi=300)
plt.show()


# ================================================================
# 8. EXPORTAR RESUMENES Y MENSAJE FINAL
# ================================================================

df_resumen_k.to_csv(
    "actividad_p07_comparacion_k.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n==========================================")
print("ARCHIVOS GENERADOS")
print("==========================================")
print("1. actividad_p07_comparacion_k.csv")
print("2. actividad_p07_estudiantes_5vars.csv")
print("3. actividad_p07_centroides_5vars.csv")
print("4. actividad_p07_grafica_resultados.png")
print("\nACTIVIDAD P07 FINALIZADA.")