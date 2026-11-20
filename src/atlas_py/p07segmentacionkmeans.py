
# ================================================================
# PROYECTO: SMARTACADEMY
# PRACTICA 07: SEGMENTACION DE ESTUDIANTES MEDIANTE K-MEANS
# ARCHIVO: p07segmentacion_kmeans.py
#
# OBJETIVO:
# Implementar un algoritmo K-Means para identificar grupos de estudiantes 
# con características académicas similares, utilizando técnicas de 
# agrupamiento (clustering), escalamiento de variables, selección del 
# número de grupos e interpretación de resultados.

#Caso de estudio
#SmartAcademy ha recopilado información de 500 estudiantes. 
#La institución desea identificar perfiles académicos para diseñar estrategias de acompañamiento.

#Sin embargo, existe un problema: no sabemos previamente cuántos perfiles de estudiantes existen.

#Queremos que el algoritmo encuentre agrupaciones como:
#- Estudiantes con alto rendimiento académico.
#- Estudiantes con rendimiento intermedio.
#- Estudiantes que necesitan acompañamiento académico.

#Una diferencia fundamental: no vamos a utilizar la columna aprueba para formar los grupos. 
#K-Means buscará similitudes a partir de las características de los estudiantes.

#Fundamentos teóricos: ¿qué es K-Means?
#K-Means es un algoritmo de aprendizaje no supervisado que agrupa registros de acuerdo 
#con la similitud de sus características.

#La letra K representa el número de grupos que queremos encontrar.

#La palabra Means hace referencia a la media o promedio que se utiliza para calcular 
#el centro de cada grupo, conocido como centroide.

# El algoritmo trabaja mediante cuatro etapas:
# 1. Inicialización: selecciona K centroides iniciales.
# 2. Asignación: asigna cada estudiante al centroide más cercano.
# 3. Actualización: calcula nuevamente el centro de cada grupo utilizando el promedio de sus integrantes.
# 4. Convergencia: repite la asignación y actualización hasta que los grupos dejan de 
#                   cambiar significativamente o se alcanza el límite de iteraciones.

#Ejemplo sencillo

#Imaginemos que tenemos tres estudiantes:
#Estudiante	Horas de estudio	Asistencia
#   A	            2	           50
#   B	            3	           55
#   C	            8	           95

#Es probable que A y B sean agrupados juntos porque tienen características similares, 
#mientras que C pertenezca a otro grupo.

#K-Means no conoce previamente los nombres de los grupos ni sabe si un estudiante aprueba. 
#Solamente calcula distancias y busca agrupaciones.

#   Concepto	            Descripción

#   Clustering	            Proceso de agrupar registros similares.
#   K	                    Número de grupos que se desean encontrar.
#   Centroide	            Centro matemático de un grupo.
#   Distancia euclidiana	Medida de distancia entre registros y centroides.
#   WCSS	                Suma de las distancias cuadráticas de los registros a su centroide.
#   Inercia	                Valor WCSS que calcula K-Means.
#   Silhouette Score	    Métrica que evalúa qué tan bien separados y cohesionados están los grupos.

#¿Por qué debemos escalar los datos?
#En nuestro caso tenemos variables con diferentes escalas:
#- Horas de estudio: aproximadamente de 0 a 12.
#- Asistencia: de 0 a 100.
#- Actividades: de 0 a 30.
#- Participación: de 0 a 100.

#Si calculamos distancias directamente, las variables con valores numéricos más grandes 
#podrían tener demasiada influencia.

#Por ello utilizaremos StandardScaler, que transforma las variables para que tengan 
#aproximadamente media 0 y desviación estándar 1.

# IMPORTANTE:
# No utilizaremos una variable objetivo como "aprueba".
# El algoritmo descubrira los grupos utilizando las caracteristicas.

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
# 2. CONFIGURAR LA GENERACION DE DATOS
# ================================================================

# Utilizamos una semilla para que los resultados sean reproducibles.
rng = np.random.default_rng(42)

# Definimos la cantidad de estudiantes por perfil simulado.
# Estos perfiles solo se utilizan para generar datos de ejemplo.
# NO se entregan al algoritmo K-Means.

cantidad_estudiantes = [180, 170, 150]

# ------------------------------------------------
# Perfil simulado 1: estudiantes con alto compromiso
# ------------------------------------------------

perfil_alto = np.column_stack([

    rng.normal(8.0, 1.0, cantidad_estudiantes[0]),

    rng.normal(92, 5, cantidad_estudiantes[0]),

    rng.normal(27, 2, cantidad_estudiantes[0]),

    rng.normal(90, 6, cantidad_estudiantes[0])

])

# ------------------------------------------------
# Perfil simulado 2: estudiantes con compromiso medio
# ------------------------------------------------

perfil_medio = np.column_stack([

    rng.normal(5.0, 1.0, cantidad_estudiantes[1]),

    rng.normal(75, 7, cantidad_estudiantes[1]),

    rng.normal(20, 3, cantidad_estudiantes[1]),

    rng.normal(65, 8, cantidad_estudiantes[1])

])

# ------------------------------------------------
# Perfil simulado 3: estudiantes que requieren apoyo
# ------------------------------------------------

perfil_bajo = np.column_stack([

    rng.normal(2.5, 1.0, cantidad_estudiantes[2]),

    rng.normal(55, 8, cantidad_estudiantes[2]),

    rng.normal(12, 4, cantidad_estudiantes[2]),

    rng.normal(40, 10, cantidad_estudiantes[2])

])


# ================================================================
# 3. UNIFICAR LOS REGISTROS
# ================================================================

# Juntamos los tres conjuntos de datos.
# El algoritmo no recibe ninguna etiqueta de perfil.

datos = np.vstack([

    perfil_alto,

    perfil_medio,

    perfil_bajo

])

# Mezclamos los registros para evitar que aparezcan ordenados
# de acuerdo con los perfiles utilizados para generarlos.

rng.shuffle(datos)


# ================================================================
# 4. CREAR EL DATAFRAME
# ================================================================

# Construimos una tabla con las caracteristicas academicas.

df = pd.DataFrame(

    datos,

    columns=[

        "horas_estudio",

        "asistencia",

        "actividades",

        "participacion"

    ]

)

# ------------------------------------------------
# 4.1 Ajustar los valores a rangos realistas
# ------------------------------------------------

# Evitamos valores negativos en horas de estudio.

df["horas_estudio"] = df["horas_estudio"].clip(0, 12)

# La asistencia se expresa de 0 a 100.

df["asistencia"] = df["asistencia"].clip(0, 100)

# Las actividades se expresan de 0 a 30.

df["actividades"] = df["actividades"].clip(0, 30)

# La participacion se expresa de 0 a 100.

df["participacion"] = df["participacion"].clip(0, 100)


# ================================================================
# 5. EXPLORAR LOS DATOS
# ================================================================

print("\n==========================================")
print("1. DATOS GENERADOS")
print("==========================================")

print(df.head(10))

print("\nCantidad de estudiantes:")

print(len(df))

print("\nInformacion estadistica:")

print(df.describe().round(2))


# ================================================================
# 6. SELECCIONAR LAS VARIABLES
# ================================================================

# Seleccionamos unicamente las caracteristicas que utilizaran
# los algoritmos de agrupamiento.

X = df[

    [

        "horas_estudio",

        "asistencia",

        "actividades",

        "participacion"

    ]

]


# ================================================================
# 7. ESCALAR LOS DATOS
# ================================================================

# Creamos el objeto StandardScaler.

escalador = StandardScaler()

# Calculamos la media y desviacion estandar de cada variable
# y transformamos los registros.

X_escalado = escalador.fit_transform(X)

print("\n==========================================")
print("2. DATOS ESCALADOS")
print("==========================================")

print("Media aproximada de las variables:")

print(X_escalado.mean(axis=0).round(2))

print("\nDesviacion estandar aproximada:")

print(X_escalado.std(axis=0).round(2))


# ================================================================
# 8. METODO DEL CODO (ELBOW METHOD)
# ================================================================

# El metodo del codo ayuda a explorar cuantos grupos podemos
# utilizar en nuestro problema.

# Calcularemos la inercia para diferentes valores de K.

valores_k = range(1, 9)

inercias = []

for k in valores_k:

    # Creamos un modelo K-Means con el numero de grupos actual.

    modelo = KMeans(

        n_clusters=k,

        random_state=42,

        n_init=20

    )

    # Entrenamos el modelo.

    modelo.fit(X_escalado)

    # Guardamos la inercia.

    inercias.append(modelo.inertia_)

    print(f"K={k} | Inercia={modelo.inertia_:.2f}")


# ------------------------------------------------
# 8.1 Graficar el metodo del codo
# ------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(

    list(valores_k),

    inercias,

    marker="o",

    linewidth=2

)

plt.title("SmartAcademy - Metodo del codo")

plt.xlabel("Cantidad de grupos (K)")

plt.ylabel("Inercia (WCSS)")

plt.xticks(list(valores_k))

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig("p07_metodo_del_codo.png", dpi=300)

plt.show()


# ================================================================
# 9. EVALUAR LOS GRUPOS CON SILHOUETTE SCORE
# ================================================================

# El coeficiente silhouette mide que tan bien se encuentra
# cada registro dentro de su grupo y separado de los demas.

# Sus valores van aproximadamente de -1 a 1.

# Valores cercanos a 1 indican grupos bien separados.
# Valores cercanos a 0 indican grupos que se traslapan.
# Valores negativos pueden indicar asignaciones inadecuadas.

resultados_silhouette = []

for k in range(2, 9):

    modelo = KMeans(

        n_clusters=k,

        random_state=42,

        n_init=20

    )

    etiquetas = modelo.fit_predict(X_escalado)

    puntuacion = silhouette_score(

        X_escalado,

        etiquetas

    )

    resultados_silhouette.append(puntuacion)

    print(

        f"K={k} | Silhouette Score={puntuacion:.4f}"

    )


# ------------------------------------------------
# 9.1 Graficar los resultados
# ------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(

    list(range(2, 9)),

    resultados_silhouette,

    marker="o",

    linewidth=2

)

plt.title("SmartAcademy - Evaluacion Silhouette")

plt.xlabel("Cantidad de grupos (K)")

plt.ylabel("Silhouette Score")

plt.xticks(list(range(2, 9)))

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig("p07_silhouette.png", dpi=300)

plt.show()


# ================================================================
# 10. ENTRENAR EL MODELO FINAL
# ================================================================

# Para esta practica utilizaremos tres grupos como hipotesis
# inicial del caso de estudio.
#
# La eleccion debe justificarse observando el metodo del codo,
# Silhouette Score y el conocimiento del problema.

K_FINAL = 3

modelo_final = KMeans(

    n_clusters=K_FINAL,

    random_state=42,

    n_init=20

)

# fit_predict entrena el modelo y devuelve el grupo asignado
# a cada estudiante.

df["cluster"] = modelo_final.fit_predict(X_escalado)


# ================================================================
# 11. INTERPRETAR LOS GRUPOS
# ================================================================

print("\n==========================================")
print("3. CANTIDAD DE ESTUDIANTES POR GRUPO")
print("==========================================")

print(df["cluster"].value_counts().sort_index())


# ------------------------------------------------
# 11.1 Calcular los perfiles promedio
# ------------------------------------------------

# Agrupamos por cluster y calculamos el promedio
# de cada caracteristica.

perfiles = df.groupby("cluster")[

    [

        "horas_estudio",

        "asistencia",

        "actividades",

        "participacion"

    ]

].mean().round(2)

print("\n==========================================")
print("4. PERFIL ACADEMICO DE CADA GRUPO")
print("==========================================")

print(perfiles)


# ------------------------------------------------
# 11.2 Asignar nombres descriptivos a los grupos
# ------------------------------------------------

# Los numeros de cluster son etiquetas arbitrarias.
# Cluster 0 no significa necesariamente rendimiento bajo.
#
# Ordenamos los grupos de acuerdo con sus horas promedio
# de estudio para asignarles una descripcion orientativa.

orden = perfiles["horas_estudio"].sort_values().index.tolist()

nombres = [

    "Requiere acompanamiento",

    "Compromiso intermedio",

    "Alto compromiso"

]

mapa_nombres = {

    cluster: nombre

    for cluster, nombre in zip(orden, nombres)

}

df["perfil"] = df["cluster"].map(mapa_nombres)

print("\n==========================================")
print("5. INTERPRETACION DE LOS PERFILES")
print("==========================================")

print(

    df[

        [

            "horas_estudio",

            "asistencia",

            "actividades",

            "participacion",

            "cluster",

            "perfil"

        ]

    ].head(15)

)


# ================================================================
# 12. VISUALIZAR LOS GRUPOS
# ================================================================

# Representaremos las horas de estudio y la asistencia.
# Aunque K-Means utiliza cuatro variables, podemos mostrar
# dos de ellas para facilitar la interpretacion visual.

plt.figure(figsize=(10, 6))

for cluster in sorted(df["cluster"].unique()):

    datos_cluster = df[df["cluster"] == cluster]

    plt.scatter(

        datos_cluster["horas_estudio"],

        datos_cluster["asistencia"],

        label=mapa_nombres[cluster],

        alpha=0.65

    )


# ------------------------------------------------
# 12.1 Recuperar los centroides a escala original
# ------------------------------------------------

# Los centroides del modelo estan estandarizados.
# inverse_transform los devuelve a las unidades originales.

centroides_originales = escalador.inverse_transform(

    modelo_final.cluster_centers_

)

# Convertimos los centroides a un DataFrame.

centroides_df = pd.DataFrame(

    centroides_originales,

    columns=X.columns

)

# ------------------------------------------------
# 12.2 Mostrar los centroides en la grafica
# ------------------------------------------------

plt.scatter(

    centroides_df["horas_estudio"],

    centroides_df["asistencia"],

    marker="X",

    s=250,

    edgecolor="black",

    linewidth=1.5,

    label="Centroides"

)

plt.title("SmartAcademy - Segmentacion de estudiantes")

plt.xlabel("Horas de estudio")

plt.ylabel("Asistencia (%)")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig("p07_segmentacion_estudiantes.png", dpi=300)

plt.show()


# ================================================================
# 13. MOSTRAR LOS CENTROIDES
# ================================================================

print("\n==========================================")
print("6. CENTROIDES EN ESCALA ORIGINAL")
print("==========================================")

print(centroides_df.round(2))


# ================================================================
# 14. EVALUAR EL MODELO FINAL
# ================================================================

silhouette_final = silhouette_score(

    X_escalado,

    df["cluster"]

)

print("\n==========================================")
print("7. EVALUACION FINAL DEL MODELO")
print("==========================================")

print("Numero de grupos:", K_FINAL)

print("Inercia final:", round(modelo_final.inertia_, 2))

print("Silhouette Score:", round(silhouette_final, 4))


# ================================================================
# 15. EXPORTAR LOS RESULTADOS
# ================================================================

# Guardamos los registros con su cluster y perfil asignado.

df.to_csv(

    "p07_estudiantes_segmentados.csv",

    index=False,

    encoding="utf-8-sig"

)

# Exportamos los promedios de los perfiles.

perfiles.to_csv(

    "p07_perfiles_academicos.csv",

    encoding="utf-8-sig"

)


# ================================================================
# 16. MENSAJE FINAL
# ================================================================

print("\n==========================================")
print("PRACTICA 07 FINALIZADA")
print("==========================================")

print("Archivos generados:")

print("1. p07_metodo_del_codo.png")

print("2. p07_silhouette.png")

print("3. p07_segmentacion_estudiantes.png")

print("4. p07_estudiantes_segmentados.csv")

print("5. p07_perfiles_academicos.csv")

print("\nSmartAcademy ha segmentado a sus estudiantes.")

print("El proceso se realizo mediante aprendizaje no supervisado.")

print("==========================================")


#A. Método del codo

#El método del codo representa la inercia para diferentes valores de K.

#La inercia mide la suma de las distancias cuadráticas entre los registros y sus centroides.
#- Una inercia menor indica que los registros están más cerca de sus centroides.
#- Al incrementar K, la inercia disminuye.
#- Buscamos un punto donde la reducción comienza a ser menos significativa.
#Ese punto se conoce como el codo.

#Importante: no debemos elegir automáticamente el K que tenga la menor inercia, 
#porque aumentar el número de grupos normalmente reduce este valor. 
#Buscamos un equilibrio entre simplicidad y calidad.

#B. Silhouette Score

#Esta métrica permite evaluar qué tan bien agrupados están los registros.

#Valor aproximado	Interpretación

#Cercano a 1	    Grupos bien definidos y separados.
#Cercano a 0	    Existen grupos que se traslapan.
#Menor que 0	    Algunos registros podrían estar asignados a grupos inadecuados.

#No son límites absolutos para determinar si un modelo es bueno o malo; deben interpretarse 
#considerando el problema y los datos.

#C. Perfiles académicos
#La salida de perfiles mostrará los promedios de cada grupo.

#Por ejemplo, los resultados podrían tener esta estructura:

#   Perfil	                Horas de estudio	Asistencia	 Actividades	Participación
#   Requiere acompañamiento	    2.5	                55	         12	             40
#   Compromiso intermedio	    5.0	                75	         20	             65
#   Alto compromiso	            8.0	                92	         27	             90

#Son valores ilustrativos, no resultados reales de la ejecución.

#Con esta información, SmartAcademy podría proponer estrategias como:

#- Ofrecer tutorías y seguimiento a estudiantes que requieren acompañamiento.
#- Diseñar actividades de reforzamiento para estudiantes con compromiso intermedio.
#- Proponer actividades de profundización para estudiantes con alto compromiso.
#Es importante aclarar que estos perfiles describen patrones de comportamiento académico, 
#no determinan por sí mismos si un estudiante aprobará o reprobará.


#Actividades para los alumnos

#Actividad 1. Cambiar el número de grupos, Comparar las agrupaciones y explicar qué cambios se producen.

#Actividad 2. Modificar las características, Incorporar una nueva variable.
#            Modificar la generación de datos y agregar la característica a:
#                   - El DataFrame.
#                   - La selección de variables.
#                   - Los centroides.
#                   - La exportación de resultados.

#Actividad 3. Comparar los resultados

#Ejecutar K-Means utilizando:
#- Datos originales sin escalar.
#- Datos estandarizados con StandardScaler.
#Comparar las agrupaciones y explicar por qué el escalamiento puede modificar los resultados.

#Actividad 4. Investigar

#Investigar qué sucede cuando existen valores atípicos (outliers) y cómo pueden afectar la ubicación de los centroides.