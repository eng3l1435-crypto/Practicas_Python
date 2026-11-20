
# ================================================================
# PROYECTO: SMARTACADEMY
# PRACTICA 06: OPTIMIZACION DE HIPERPARAMETROS

# Archivo:
# p06optimizacion_hiperparametros.py

# Objetivo:
# Encontrar las mejores configuraciones de los algoritmos
# Decision Tree, Random Forest y KNN que pruebe distintas configuraciones 
# de los algoritmos de Machine Learning para encontrar aquella que consiga 
# el mejor rendimiento de clasificación.

# Parámetros: valores que el algoritmo aprende durante el entrenamiento.

# Hiperparámetros: es un valor que configuramos antes de entrenar el modelo para controlar como aprende.

# Optimización: proceso de buscar la configuración que obtiene mejores resultados.

# Validación cruzada: evaluación del modelo utilizando diferentes particiones de los datos.

# Que se va a optimizar

# Decision Tree: Un árbol de decisión funciona mediante preguntas y condiciones. 
    #Cada respuesta permite avanzar por una rama hasta llegar a una decisión final.

# - max_depth: Establece la profundidad máxima que puede alcanzar el árbol de decisión, En otras palabras, 
#              determina cuántos niveles de preguntas puede realizar el algoritmo antes de llegar a una decisión.

# - min_samples_split: Establece la cantidad mínima de registros que debe tener un nodo para que el algoritmo 
#               pueda dividirlo en dos nodos hijos.

# Significa que un nodo necesita al menos 10 registros para poder dividirse.

#   Valor	Comportamiento

#   2	    Permite dividir nodos con muy pocos registros.
#   5	    Exige al menos cinco registros.
#   10	    Exige al menos diez registros.
#   20	    Limita más las divisiones del árbol.

        #Ejemplo SmartAcademy:

        #Imaginemos que el árbol llega a un grupo de estudiantes que tienen características similares.
        
        #- Si hay 15 estudiantes, el nodo puede dividirse cuando el mínimo es 10.
        #- Si hay 8 estudiantes, no puede dividirse cuando el mínimo es 10.
        #Evita que el árbol continúe creando divisiones con grupos demasiado pequeños, 
        #lo cual puede ayudar a reducir el sobreajuste.

# - min_samples_leaf: Establece la cantidad mínima de registros que debe contener cada hoja del árbol.
        #Recordemos que una hoja es un nodo final que representa una predicción, ayuda a evitar decisiones basadas 
        #en muy pocos ejemplos y puede mejorar la capacidad de generalización del modelo..

# - criterion: Define el criterio matemático que utiliza el árbol para evaluar qué tan 
        #buena es una división de los datos, En otras palabras, determina cómo decide el algoritmo 
        #cuál pregunta debe realizar primero.

#A) Gini

#Mide la impureza de un grupo.

#Un grupo es puro cuando sus registros pertenecen a una sola clase.

#Ejemplo:
#- 10 estudiantes que aprueban: grupo puro.

#- 5 estudiantes que aprueban y 5 que no aprueban: grupo mezclado.

#El algoritmo busca divisiones que reduzcan la impureza.

#B) Entropy

#Utiliza el concepto matemático de entropía para medir la incertidumbre de un grupo.

#Busca divisiones que permitan reducir esa incertidumbre.

# Criterio	 Descripción

#   gini	 Evalúa la impureza de los grupos.
#   entropy	 Evalúa la incertidumbre de los grupos.

#¿Cuál es mejor?

#No existe uno que sea siempre mejor. Depende de los datos y del problema. 
#Por eso podemos probar ambos mediante GridSearchCV.

# Random Forest: utiliza múltiples árboles de decisión para generar una predicción conjunta.

#Imaginemos que tenemos 100 árboles evaluando a un estudiante:

#- 75 árboles predicen que aprueba.
#- 25 árboles predicen que no aprueba.
#La predicción final será: Aprueba.

#Esto se conoce como votación mayoritaria para clasificación.

# - n_estimators: Define cuántos árboles de decisión se van a construir dentro del bosque aleatorio.

# - max_depth: Establece la profundidad máxima permitida para cada árbol que forma parte del bosque. 
         #Aunque Random Forest combina muchos árboles, cada uno puede sobreajustarse. 
         #Limitar su profundidad ayuda a controlar su complejidad.

# - min_samples_split: Establece la cantidad mínima de registros que necesita un nodo para 
         #poder dividirse, Cada árbol deberá tener al menos x registros en un nodo para permitir una división.

# - min_samples_leaf: Establece el número mínimo de registros que debe contener cada hoja de los árboles.

# KNN: algoritmo de los K vecinos más cercanos, en lugar de construir reglas o múltiples árboles, 
         #KNN busca los registros más similares al nuevo registro que queremos clasificar.

         #Tenemos un estudiante nuevo con estas características:
            #- Horas de estudio: 6.
            #- Asistencia: 85%.
            #- Actividades: 25.
            #- Participación: 80%.

#KNN compara sus características con las de los estudiantes registrados y busca los más cercanos.

            #Si configuramos n_neighbors=5, consulta a los cinco estudiantes más similares.
            #Si cuatro aprobaron y uno no aprobó, la predicción será: Aprueba.

#Sus hiperparámetros son:

# - n_neighbors: Define cuántos vecinos cercanos utilizará el algoritmo para generar una predicción.

# - weights:Define cómo se consideran los vecinos al momento de realizar una predicción.
            #A) uniform Todos los vecinos tienen la misma importancia.
            #B) distance Los vecinos más cercanos tienen mayor influencia en la predicción 
                #que los vecinos más alejados.

# - metric: Define la fórmula matemática que se utilizará para calcular la distancia entre los registros.
            #En nuestra práctica utilizamos:
            #- euclidean
            #- manhattan
            
            #A) Euclidean (distancia euclidiana: Calcula la distancia en línea recta entre dos puntos.
            #B) Manhattan (distancia Manhattan): Calcula la suma de las diferencias absolutas entre 
                                                #las características.

# Tecnicas:
# 1. GridSearchCV: es una herramienta de Scikit-learn que permite probar diferentes 
                #combinaciones de hiperparámetros para encontrar aquella que obtiene el mejor 
                #resultado según una métrica determinada, En lugar de elegir manualmente los valores, 
                #nosotros proporcionamos las opciones y el algoritmo evalúa todas las combinaciones posibles.

#   Argumento	Descripción

#   estimator	Modelo que queremos optimizar.
#   param_grid	Diccionario con los valores que queremos probar.
#   scoring	    Métrica utilizada para comparar las configuraciones.
#   cv	        Número de particiones para validación cruzada, o un objeto de particionado.
#   n_jobs	    Cantidad de procesos utilizados. -1 aprovecha los procesadores disponibles.

#¿Cuándo utilizar GridSearchCV?

#Cuando tenemos un espacio de búsqueda relativamente pequeño y queremos evaluar todas las combinaciones definidas.


# 2. RandomizedSearchCV: busca una buena combinación de hiperparámetros, 
                #pero a diferencia de GridSearchCV, no prueba todas las combinaciones posibles.
                #Selecciona aleatoriamente una cantidad determinada de configuraciones.
                #Esto permite explorar espacios de búsqueda grandes sin evaluar absolutamente todas las posibilidades.

#   Argumento	            Descripción

#   estimator	            Modelo que se desea optimizar.
#   param_distributions	    Valores o distribuciones de donde se seleccionan los parámetros.
#   n_iter	                Cantidad de configuraciones aleatorias que se evaluarán.
#   scoring	                Métrica para comparar los resultados.
#   cv	                    Estrategia de validación cruzada.
#   random_state	        Permite reproducir la selección aleatoria.
#   n_jobs	                Procesos utilizados para ejecutar la búsqueda.

# 3. StratifiedKFold: es una estrategia que divide un conjunto de datos en diferentes grupos, 
                #procurando conservar la proporción de las clases originales en cada partición.

#   Argumento	            Descripción

#   n_splits=5	            Divide los datos en cinco particiones.
#   shuffle=True	        Mezcla los registros antes de dividirlos.
#   random_state=42	        Permite reproducir la división.



# 4. Validacion cruzada (Cross Validation): La validación cruzada es una técnica que permite evaluar 
                #el rendimiento de un modelo utilizando diferentes particiones del conjunto de datos.
                
                #La finalidad es comprobar si el modelo obtiene resultados consistentes y no depende 
                #únicamente de una división particular de los datos, Una de las estrategias más 
                #utilizadas es K-Fold Cross Validation.

    #Supongamos que tenemos 100 estudiantes y queremos realizar una validación cruzada con cinco particiones.

        #El proceso será:
        
        #0. Proceso de limpieza.
        #1. Dividir los datos en cinco grupos.
        #2. Utilizar cuatro grupos para entrenar.
        #3. Utilizar el grupo restante para validar.
        #4. Repetir el proceso cinco veces, cambiando el grupo de validación.
        #5. Calcular el promedio de los resultados.
        #6. Presentar los resultados.
#
# Problema:
# Predecir si un estudiante aprueba o no.
#
# 0 = No aprueba
# 1 = Aprueba
# ================================================================


# ================================================================
# 1. IMPORTAR BIBLIOTECAS
# ================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Separacion de datos.
from sklearn.model_selection import train_test_split

# Busqueda exhaustiva de hiperparametros.
from sklearn.model_selection import GridSearchCV

# Busqueda aleatoria de hiperparametros.
from sklearn.model_selection import RandomizedSearchCV

# Validacion cruzada estratificada.
from sklearn.model_selection import StratifiedKFold

# Algoritmos de clasificacionb aprendizaje supervisado.
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Normalizacion de datos.
from sklearn.preprocessing import StandardScaler

# Pipeline para evitar fuga de informacion.
from sklearn.pipeline import Pipeline

# Metricas de evaluacion.
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Distribuciones para la busqueda aleatoria.
from scipy.stats import randint


# ================================================================
# 2. CONFIGURACION GENERAL
# ================================================================

SEMILLA = 42

rng = np.random.default_rng(SEMILLA)

CANTIDAD_ESTUDIANTES = 500

print("\n==========================================")
print("SMARTACADEMY")
print("OPTIMIZACION DE HIPERPARAMETROS")
print("==========================================")


# ================================================================
# 3. GENERAR DATOS SINTETICOS
# ================================================================

# Generamos las caracteristicas de los estudiantes.

horas_estudio = rng.uniform(
    0, 20, CANTIDAD_ESTUDIANTES
)

asistencia = rng.uniform(
    40, 100, CANTIDAD_ESTUDIANTES
)

actividades = rng.uniform(
    0, 100, CANTIDAD_ESTUDIANTES
)

participacion = rng.uniform(
    0, 10, CANTIDAD_ESTUDIANTES
)

# Construimos un indicador sintetico de rendimiento.

puntaje = (
    (horas_estudio / 20) * 0.30
    +
    (asistencia / 100) * 0.25
    +
    (actividades / 100) * 0.30
    +
    (participacion / 10) * 0.15
)

# Agregamos variabilidad a los resultados = ruido.

ruido = rng.normal(
    0, 0.12, CANTIDAD_ESTUDIANTES
)

puntaje_final = puntaje + ruido

# Definimos la variable objetivo.

aprueba = (
    puntaje_final >= 0.58
).astype(int)


# ================================================================
# 4. CREAR DATAFRAME
# ================================================================

df = pd.DataFrame({

    "horas_estudio": horas_estudio,

    "asistencia": asistencia,

    "actividades": actividades,

    "participacion": participacion,

    "aprueba": aprueba

})

print("\n1. DATOS GENERADOS")

print(df.head())

print("\nDistribucion de clases:")

print(df["aprueba"].value_counts())


# ================================================================
# 5. SEPARAR VARIABLES
# ================================================================

# Variables predictoras.

X = df[
    [
        "horas_estudio",
        "asistencia",
        "actividades",
        "participacion"
    ]
]

# Variable objetivo.

y = df["aprueba"]


# ================================================================
# 6. DIVIDIR DATOS EN ENTRENAMIENTO Y PRUEBA
# ================================================================

# Reservamos el 20% para la evaluacion final.
#
# El 80% restante se utilizara para optimizar
# los hiperparametros mediante validacion cruzada.

X_entrenamiento, X_prueba, \
y_entrenamiento, y_prueba = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=SEMILLA,

    stratify=y

)

print("\n2. DISTRIBUCION DE DATOS")

print("Entrenamiento:", len(X_entrenamiento))

print("Prueba:", len(X_prueba))


# ================================================================
# 7. CONFIGURAR VALIDACION CRUZADA
# ================================================================

# StratifiedKFold divide los datos en cinco particiones.
#
# Mantiene una proporcion semejante de estudiantes
# que aprueban y no aprueban en cada particion.

validacion_cruzada = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=SEMILLA

)


# ================================================================
# 8. DEFINIR LOS ALGORITMOS Y SUS HIPERPARAMETROS
# ================================================================

# ------------------------------------------------
# MODELO 1: DECISION TREE
# ------------------------------------------------

modelo_arbol = DecisionTreeClassifier(
    random_state=SEMILLA
)

parametros_arbol = {

    "max_depth": [2, 3, 5, 8, 10, None],

    "min_samples_split": [2, 5, 10],

    "min_samples_leaf": [1, 2, 4],

    "criterion": ["gini", "entropy"]

}


# ------------------------------------------------
# MODELO 2: RANDOM FOREST
# ------------------------------------------------

modelo_bosque = RandomForestClassifier(

    random_state=SEMILLA,

    n_jobs=-1

)

parametros_bosque = {

    "n_estimators": [50, 100, 200],

    "max_depth": [3, 5, 8, None],

    "min_samples_split": [2, 5, 10],

    "min_samples_leaf": [1, 2, 4]

}


# ------------------------------------------------
# MODELO 3: KNN
# ------------------------------------------------

# KNN utiliza un Pipeline para normalizar los datos.
#
# El escalador se ajusta exclusivamente con los
# datos de entrenamiento de cada particion.

modelo_knn = Pipeline([

    (
        "escalador",
        StandardScaler()
    ),

    (
        "clasificador",
        KNeighborsClassifier()
    )

])

# Los nombres de los parametros de un Pipeline
# utilizan el formato:
#
# nombre_paso__nombre_parametro

parametros_knn = {

    "clasificador__n_neighbors": [3, 5, 7, 9, 11, 15],

    "clasificador__weights": [
        "uniform",
        "distance"
    ],

    "clasificador__metric": [
        "euclidean",
        "manhattan"
    ]

}


# ================================================================
# 9. ORGANIZAR LOS MODELOS
# ================================================================

modelos = {

    "Decision Tree": (
        modelo_arbol,
        parametros_arbol
    ),

    "Random Forest": (
        modelo_bosque,
        parametros_bosque
    ),

    "KNN": (
        modelo_knn,
        parametros_knn
    )

}


# ================================================================
# 10. EJECUTAR GRIDSEARCHCV
# ================================================================

resultados_grid = []

mejores_modelos_grid = {}

print("\n==========================================")
print("3. OPTIMIZACION CON GRIDSEARCHCV")
print("==========================================")

# GridSearchCV prueba todas las combinaciones
# de hiperparametros que se definieron.

for nombre, configuracion in modelos.items():

    modelo, parametros = configuracion

    print("\n--------------------------------------")

    print("Optimizando:", nombre)

    print("--------------------------------------")

    busqueda = GridSearchCV(

        estimator=modelo,

        param_grid=parametros,

        # Metrica utilizada para comparar modelos.
        scoring="f1",

        # Cinco particiones de validacion cruzada.
        cv=validacion_cruzada,

        # Permite utilizar varios nucleos.
        n_jobs=-1,

        # Conserva resultados de entrenamiento.
        return_train_score=True,

        verbose=0

    )

    # Buscamos la mejor configuracion.
    busqueda.fit(
        X_entrenamiento,
        y_entrenamiento
    )

    # Guardamos el mejor modelo.
    mejores_modelos_grid[nombre] = (
        busqueda.best_estimator_
    )

    print("Mejor F1-score:",

          f"{busqueda.best_score_:.2%}")

    print("\nMejores hiperparametros:")

    print(busqueda.best_params_)

    # Guardamos los resultados.
    resultados_grid.append({

        "algoritmo": nombre,

        "mejor_f1": busqueda.best_score_,

        "hiperparametros": str(
            busqueda.best_params_
        ),

        "combinaciones": len(
            busqueda.cv_results_["params"]
        )

    })


# ================================================================
# 11. COMPARACION DE RESULTADOS GRIDSEARCHCV
# ================================================================

df_grid = pd.DataFrame(resultados_grid)

print("\n==========================================")

print("4. RESULTADOS GRIDSEARCHCV")

print("==========================================")

print(

    df_grid[
        [
            "algoritmo",
            "mejor_f1",
            "combinaciones"
        ]
    ].to_string(

        index=False,

        formatters={
            "mejor_f1": "{:.2%}".format
        }

    )

)


# ================================================================
# 12. RANDOMIZEDSEARCHCV
# ================================================================

# RandomizedSearchCV no prueba todas las combinaciones.
#
# Selecciona aleatoriamente una cantidad determinada
# de configuraciones del espacio de busqueda.
#
# Esto puede reducir el tiempo de procesamiento
# cuando existen muchos hiperparametros.

resultados_random = []

mejores_modelos_random = {}

print("\n==========================================")

print("5. OPTIMIZACION CON RANDOMIZEDSEARCHCV")

print("==========================================")

# Configuraciones de busqueda aleatoria.
#
# Para Decision Tree y Random Forest utilizamos
# distribuciones de valores enteros.
#
# Para KNN utilizamos listas de opciones.

parametros_random = {

    "Decision Tree": {

        "max_depth": randint(2, 15),

        "min_samples_split": randint(2, 15),

        "min_samples_leaf": randint(1, 8),

        "criterion": ["gini", "entropy"]

    },

    "Random Forest": {

        "n_estimators": randint(50, 301),

        "max_depth": [3, 5, 8, 12, None],

        "min_samples_split": randint(2, 12),

        "min_samples_leaf": randint(1, 6)

    },

    "KNN": {

        "clasificador__n_neighbors": [
            3, 5, 7, 9, 11, 15, 19
        ],

        "clasificador__weights": [
            "uniform",
            "distance"
        ],

        "clasificador__metric": [
            "euclidean",
            "manhattan"
        ]

    }

}

for nombre, configuracion in modelos.items():

    modelo, _ = configuracion

    print("\n--------------------------------------")

    print("Optimizando aleatoriamente:", nombre)

    print("--------------------------------------")

    busqueda_random = RandomizedSearchCV(

        estimator=modelo,

        param_distributions=parametros_random[nombre],

        # Numero de configuraciones que se probaran.
        n_iter=20,

        scoring="f1",

        cv=validacion_cruzada,

        n_jobs=-1,

        random_state=SEMILLA,

        verbose=0

    )

    # Ejecutamos la busqueda.
    busqueda_random.fit(

        X_entrenamiento,

        y_entrenamiento

    )

    # Guardamos el modelo.
    mejores_modelos_random[nombre] = (
        busqueda_random.best_estimator_
    )

    print("Mejor F1-score:",

          f"{busqueda_random.best_score_:.2%}")

    print("\nMejores hiperparametros:")

    print(busqueda_random.best_params_)

    resultados_random.append({

        "algoritmo": nombre,

        "mejor_f1": busqueda_random.best_score_,

        "hiperparametros": str(
            busqueda_random.best_params_
        )

    })


# ================================================================
# 13. COMPARAR GRIDSEARCH Y RANDOMIZEDSEARCH
# ================================================================

df_random = pd.DataFrame(resultados_random)

print("\n==========================================")

print("6. COMPARACION DE METODOS DE OPTIMIZACION")

print("==========================================")

comparacion = pd.merge(

    df_grid[
        [
            "algoritmo",
            "mejor_f1"
        ]
    ].rename(

        columns={
            "mejor_f1": "gridsearch"
        }

    ),

    df_random[
        [
            "algoritmo",
            "mejor_f1"
        ]
    ].rename(

        columns={
            "mejor_f1": "randomizedsearch"
        }

    ),

    on="algoritmo"

)

print(

    comparacion.to_string(

        index=False,

        formatters={
            "gridsearch": "{:.2%}".format,
            "randomizedsearch": "{:.2%}".format
        }

    )

)


# ================================================================
# 14. SELECCIONAR EL MEJOR ALGORITMO
# ================================================================

# Seleccionamos el algoritmo con el mayor F1-score
# obtenido mediante GridSearchCV.

mejor_indice = df_grid["mejor_f1"].idxmax()

mejor_nombre = df_grid.loc[
    mejor_indice,
    "algoritmo"
]

mejor_modelo = mejores_modelos_grid[
    mejor_nombre
]

print("\n==========================================")

print("7. MEJOR MODELO OPTIMIZADO")

print("==========================================")

print("Algoritmo:", mejor_nombre)

print(

    "F1-score en validacion cruzada:",

    f"{df_grid.loc[mejor_indice, 'mejor_f1']:.2%}"

)

print("\nConfiguracion seleccionada:")

print(mejor_modelo.get_params())


# ================================================================
# 15. EVALUACION FINAL
# ================================================================

# IMPORTANTE:
# El conjunto de prueba se utiliza hasta este momento.
#
# No participo en la optimizacion de hiperparametros.

predicciones = mejor_modelo.predict(
    X_prueba
)

accuracy = accuracy_score(
    y_prueba,
    predicciones
)

precision = precision_score(
    y_prueba,
    predicciones,
    zero_division=0
)

recall = recall_score(
    y_prueba,
    predicciones,
    zero_division=0
)

f1 = f1_score(
    y_prueba,
    predicciones,
    zero_division=0
)

print("\n==========================================")

print("8. EVALUACION CON DATOS DE PRUEBA")

print("==========================================")

print("Accuracy:", f"{accuracy:.2%}")

print("Precision:", f"{precision:.2%}")

print("Recall:", f"{recall:.2%}")

print("F1-score:", f"{f1:.2%}")


# ================================================================
# 16. REPORTE DE CLASIFICACION
# ================================================================

print("\n==========================================")

print("9. REPORTE DE CLASIFICACION")

print("==========================================")

print(

    classification_report(

        y_prueba,

        predicciones,

        labels=[0, 1],

        target_names=[
            "No aprueba",
            "Aprueba"
        ],

        zero_division=0

    )

)


# ================================================================
# 17. MATRIZ DE CONFUSION
# ================================================================

matriz = confusion_matrix(

    y_prueba,

    predicciones,

    labels=[0, 1]

)

print("\n==========================================")

print("10. MATRIZ DE CONFUSION")

print("==========================================")

print(matriz)

visualizacion = ConfusionMatrixDisplay(

    confusion_matrix=matriz,

    display_labels=[
        "No aprueba",
        "Aprueba"
    ]

)

visualizacion.plot(
    cmap="Blues",
    values_format="d"
)

plt.title(
    "Matriz de confusion - " + mejor_nombre
)

plt.tight_layout()

plt.savefig(
    "p06_matriz_confusion.png",
    dpi=300
)

plt.show()


# ================================================================
# 18. GRAFICA COMPARATIVA DE OPTIMIZACION
# ================================================================

posiciones = np.arange(
    len(comparacion)
)

ancho = 0.35

plt.figure(figsize=(10, 6))

plt.bar(

    posiciones - ancho / 2,

    comparacion["gridsearch"],

    width=ancho,

    label="GridSearchCV"

)

plt.bar(

    posiciones + ancho / 2,

    comparacion["randomizedsearch"],

    width=ancho,

    label="RandomizedSearchCV"

)

plt.xticks(

    posiciones,

    comparacion["algoritmo"]

)

plt.ylim(0, 1.05)

plt.ylabel("F1-score")

plt.xlabel("Algoritmo")

plt.title(
    "Comparacion de metodos de optimizacion"
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "p06_comparacion_optimizacion.png",
    dpi=300
)

plt.show()


# ================================================================
# 19. EXPORTAR RESULTADOS
# ================================================================

df_grid.to_csv(

    "p06_resultados_gridsearch.csv",

    index=False,

    encoding="utf-8"

)

df_random.to_csv(

    "p06_resultados_randomizedsearch.csv",

    index=False,

    encoding="utf-8"

)

comparacion.to_csv(

    "p06_comparacion_metodos.csv",

    index=False,

    encoding="utf-8"

)


# ================================================================
# 20. CONCLUSION
# ================================================================

print("\n==========================================")

print("CONCLUSION DEL PROYECTO")

print("==========================================")

print("Mejor algoritmo:", mejor_nombre)

print("Accuracy final:", f"{accuracy:.2%}")

print("Precision final:", f"{precision:.2%}")

print("Recall final:", f"{recall:.2%}")

print("F1-score final:", f"{f1:.2%}")

print("\nArchivos generados:")

print("p06_matriz_confusion.png")

print("p06_comparacion_optimizacion.png")

print("p06_resultados_gridsearch.csv")

print("p06_resultados_randomizedsearch.csv")

print("p06_comparacion_metodos.csv")

print("\nPRACTICA 06 FINALIZADA.")
