
# ============================================================
# SMARTACADEMY
# ETAPA 18.2: ENTRENAMIENTO DE CNN 1D PARA REGRESION
#
# Objetivo:
# Predecir la calificacion del parcial 3 utilizando
# exclusivamente los parciales 1 y 2.
#
# Framework: PyTorch
# Problema: Regresion
# ============================================================

from pathlib import Path
import json
import time
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. CONFIGURACION
# ============================================================

RUTA_PROYECTO = Path(__file__).resolve().parents[1]

RUTA_DATOS = (
    RUTA_PROYECTO
    / "resultados_prediccion_cnn"
)

RUTA_SALIDA = RUTA_DATOS

ARCHIVO_DATOS = (
    RUTA_DATOS
    / "smartacademy_prediccion_parcial3.npz"
)

RUTA_SALIDA.mkdir(
    parents=True,
    exist_ok=True
)

SEMILLA = 42

BATCH_SIZE = 256

MAX_EPOCAS = 60

LEARNING_RATE = 0.001

PACIENCIA = 10


# ============================================================
# 2. CONFIGURAR SEMILLAS Y DISPOSITIVO
# ============================================================

torch.manual_seed(SEMILLA)

np.random.seed(SEMILLA)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEMILLA)

    dispositivo = torch.device("cuda")

else:

    dispositivo = torch.device("cpu")


print("=" * 65)

print("SMARTACADEMY")

print("ETAPA 18.2 - CNN 1D PARA PREDICCION DEL PARCIAL 3")

print("=" * 65)

print("\nDISPOSITIVO DE ENTRENAMIENTO:")

print(dispositivo)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# 3. CARGAR DATOS
# ============================================================

print("\nCARGANDO DATOS PREPARADOS...")

if not ARCHIVO_DATOS.exists():

    raise FileNotFoundError(
        f"No existe el archivo: {ARCHIVO_DATOS}"
    )

datos = np.load(
    ARCHIVO_DATOS,
    allow_pickle=False
)

X_train = datos["X_train"]

y_train = datos["y_train"]

X_val = datos["X_val"]

y_val = datos["y_val"]

X_test = datos["X_test"]

y_test = datos["y_test"]

print("\nDATOS CARGADOS")

print("Entrenamiento:", X_train.shape)

print("Validacion:", X_val.shape)

print("Prueba:", X_test.shape)


# ============================================================
# 4. VALIDAR DIMENSIONES Y VALORES
# ============================================================

assert X_train.shape[1:] == (2, 2)

assert X_val.shape[1:] == (2, 2)

assert X_test.shape[1:] == (2, 2)

assert len(X_train) == len(y_train)

assert len(X_val) == len(y_val)

assert len(X_test) == len(y_test)

assert np.isfinite(X_train).all()

assert np.isfinite(X_val).all()

assert np.isfinite(X_test).all()

assert np.isfinite(y_train).all()

assert np.isfinite(y_val).all()

assert np.isfinite(y_test).all()

print("\nVALIDACIONES CORRECTAS")


# ============================================================
# 5. CREAR DATASET PARA PYTORCH
# ============================================================

class DatasetAcademico(Dataset):

    """
    Convierte los arreglos NumPy en tensores PyTorch.

    Cada muestra contiene:

    2 canales:
        Calificaciones.
        Asistencias.

    2 posiciones:
        Parcial 1.
        Parcial 2.

    El objetivo es la calificacion del parcial 3.
    """

    def __init__(self, X, y):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.float32
        )

    def __len__(self):

        return len(self.X)

    def __getitem__(self, indice):

        return (
            self.X[indice],
            self.y[indice]
        )


# Creamos los tres conjuntos.

dataset_train = DatasetAcademico(
    X_train,
    y_train
)

dataset_val = DatasetAcademico(
    X_val,
    y_val
)

dataset_test = DatasetAcademico(
    X_test,
    y_test
)


# ============================================================
# 6. CREAR DATA LOADERS
# ============================================================

loader_train = DataLoader(

    dataset_train,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0,

    pin_memory=torch.cuda.is_available()

)

loader_val = DataLoader(

    dataset_val,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0,

    pin_memory=torch.cuda.is_available()

)

loader_test = DataLoader(

    dataset_test,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0,

    pin_memory=torch.cuda.is_available()

)

print("\nDATA LOADERS CREADOS")

print("Lotes entrenamiento:", len(loader_train))

print("Lotes validacion:", len(loader_val))

print("Lotes prueba:", len(loader_test))


# ============================================================
# 7. DEFINIR ARQUITECTURA CNN 1D
# ============================================================

class CNNRegresionAcademica(nn.Module):

    """
    Red neuronal convolucional para regresion.

    Entrada:
        (lote, 2 canales, 2 parciales)

    Salida:
        Una calificacion estimada por registro.
    """

    def __init__(self):

        super().__init__()

        # Primera convolucion.
        self.conv1 = nn.Conv1d(

            in_channels=2,

            out_channels=16,

            kernel_size=2,

            padding=1

        )

        self.bn1 = nn.BatchNorm1d(16)

        # Segunda convolucion.
        self.conv2 = nn.Conv1d(

            in_channels=16,

            out_channels=32,

            kernel_size=2

        )

        self.bn2 = nn.BatchNorm1d(32)

        self.relu = nn.ReLU()

        # Reduce la secuencia a un solo valor por canal.
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(0.20)

        # Una sola neurona para regresion.
        self.fc = nn.Linear(32, 1)

    def forward(self, x):

        # Primera capa convolucional.
        x = self.conv1(x)

        x = self.bn1(x)

        x = self.relu(x)

        # Segunda capa convolucional.
        x = self.conv2(x)

        x = self.bn2(x)

        x = self.relu(x)

        # Reducimos las dimensiones temporales.
        x = self.pool(x)

        # Aplanamos los canales.
        x = torch.flatten(x, start_dim=1)

        # Regularizacion.
        x = self.dropout(x)

        # Prediccion de la calificacion.
        x = self.fc(x)

        # Obtenemos un vector de predicciones.
        return x.squeeze(1)


# ============================================================
# 8. CREAR MODELO
# ============================================================

print("\nCONSTRUYENDO CNN 1D...")

modelo = CNNRegresionAcademica().to(dispositivo)

total_parametros = sum(

    parametro.numel()

    for parametro in modelo.parameters()

    if parametro.requires_grad

)

print("\nARQUITECTURA:")

print(modelo)

print(
    "\nParametros entrenables:",
    total_parametros
)


# ============================================================
# 9. DEFINIR FUNCION DE PERDIDA Y OPTIMIZADOR
# ============================================================

# MSE mide el promedio del error cuadratico.
funcion_perdida = nn.MSELoss()

# Adam actualiza los pesos de la red.
optimizador = torch.optim.Adam(

    modelo.parameters(),

    lr=LEARNING_RATE

)

# Reduce la tasa de aprendizaje cuando la perdida
# de validacion deja de mejorar.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizador,

    mode="min",

    factor=0.5,

    patience=4

)


# ============================================================
# 10. FUNCION PARA ENTRENAR UNA EPOCA
# ============================================================

def entrenar_epoca(modelo, loader):

    modelo.train()

    perdida_acumulada = 0.0

    total = 0

    for entradas, objetivos in loader:

        entradas = entradas.to(dispositivo)

        objetivos = objetivos.to(dispositivo)

        # Limpiamos gradientes anteriores.
        optimizador.zero_grad()

        # Propagacion hacia adelante.
        predicciones = modelo(entradas)

        # Calculamos error.
        perdida = funcion_perdida(
            predicciones,
            objetivos
        )

        # Retropropagacion.
        perdida.backward()

        # Actualizacion de pesos.
        optimizador.step()

        cantidad = len(entradas)

        perdida_acumulada += (
            perdida.item() * cantidad
        )

        total += cantidad

    return perdida_acumulada / total


# ============================================================
# 11. FUNCION PARA VALIDAR UNA EPOCA
# ============================================================

def validar_epoca(modelo, loader):

    modelo.eval()

    perdida_acumulada = 0.0

    total = 0

    with torch.no_grad():

        for entradas, objetivos in loader:

            entradas = entradas.to(dispositivo)

            objetivos = objetivos.to(dispositivo)

            predicciones = modelo(entradas)

            perdida = funcion_perdida(

                predicciones,

                objetivos

            )

            cantidad = len(entradas)

            perdida_acumulada += (
                perdida.item() * cantidad
            )

            total += cantidad

    return perdida_acumulada / total


# ============================================================
# 12. ENTRENAMIENTO CON EARLY STOPPING
# ============================================================

print("\nINICIANDO ENTRENAMIENTO...")

historial = []

mejor_perdida = float("inf")

mejor_epoca = 0

mejores_pesos = None

epocas_sin_mejora = 0

inicio_entrenamiento = time.perf_counter()

for epoca in range(1, MAX_EPOCAS + 1):

    perdida_train = entrenar_epoca(

        modelo,

        loader_train

    )

    perdida_val = validar_epoca(

        modelo,

        loader_val

    )

    scheduler.step(perdida_val)

    tasa_actual = optimizador.param_groups[0]["lr"]

    historial.append({

        "epoca": epoca,

        "perdida_train": perdida_train,

        "perdida_val": perdida_val,

        "learning_rate": tasa_actual

    })

    print(

        f"Epoca {epoca:02d}/{MAX_EPOCAS} | "

        f"Train: {perdida_train:.5f} | "

        f"Val: {perdida_val:.5f} | "

        f"LR: {tasa_actual:.6f}"

    )

    # Guardamos los mejores pesos.
    if perdida_val < mejor_perdida:

        mejor_perdida = perdida_val

        mejor_epoca = epoca

        mejores_pesos = copy.deepcopy(
            modelo.state_dict()
        )

        epocas_sin_mejora = 0

    else:

        epocas_sin_mejora += 1

    # Detenemos si no mejora durante varias epocas.
    if epocas_sin_mejora >= PACIENCIA:

        print(
            "\nEARLY STOPPING ACTIVADO"
        )

        break


tiempo_entrenamiento = (
    time.perf_counter() - inicio_entrenamiento
)

print("\nENTRENAMIENTO FINALIZADO")

print("Mejor epoca:", mejor_epoca)

print(
    "Mejor perdida validacion:",
    f"{mejor_perdida:.6f}"
)

print(
    "Tiempo entrenamiento:",
    f"{tiempo_entrenamiento:.2f} segundos"
)


# ============================================================
# 13. RECUPERAR LOS MEJORES PESOS
# ============================================================

if mejores_pesos is None:

    raise RuntimeError(
        "No se encontraron pesos validos."
    )

modelo.load_state_dict(mejores_pesos)

modelo.eval()


# ============================================================
# 14. GENERAR PREDICCIONES
# ============================================================

def obtener_predicciones(modelo, loader):

    modelo.eval()

    predicciones = []

    reales = []

    with torch.no_grad():

        for entradas, objetivos in loader:

            entradas = entradas.to(dispositivo)

            resultado = modelo(entradas)

            predicciones.extend(
                resultado.cpu().numpy()
            )

            reales.extend(
                objetivos.numpy()
            )

    return (

        np.asarray(predicciones),

        np.asarray(reales)

    )


# Evaluamos el conjunto de prueba.
print("\nEVALUANDO CONJUNTO DE PRUEBA...")

inicio_inferencia = time.perf_counter()

predicciones_test, reales_test = obtener_predicciones(

    modelo,

    loader_test

)

tiempo_inferencia = (
    time.perf_counter() - inicio_inferencia
)


# ============================================================
# 15. CALCULAR METRICAS DE REGRESION
# ============================================================

mae = mean_absolute_error(

    reales_test,

    predicciones_test

)

mse = mean_squared_error(

    reales_test,

    predicciones_test

)

rmse = np.sqrt(mse)

r2 = r2_score(

    reales_test,

    predicciones_test

)

print("\nRESULTADOS DE LA CNN 1D")

print(f"MAE:  {mae:.4f}")

print(f"MSE:  {mse:.4f}")

print(f"RMSE: {rmse:.4f}")

print(f"R2:   {r2:.4f}")


# ============================================================
# 16. COMPARAR CONTRA MODELO BASE
# ============================================================

# El modelo base siempre predice el promedio
# del parcial 3 en el conjunto de entrenamiento.

prediccion_base = np.full(

    len(y_test),

    y_train.mean()

)

mae_base = mean_absolute_error(

    y_test,

    prediccion_base

)

rmse_base = np.sqrt(

    mean_squared_error(

        y_test,

        prediccion_base

    )

)

print("\nMODELO BASE: PREDICCION DEL PROMEDIO")

print(f"MAE base:  {mae_base:.4f}")

print(f"RMSE base: {rmse_base:.4f}")


# ============================================================
# 17. GUARDAR MODELO
# ============================================================

ARCHIVO_MODELO = (

    RUTA_SALIDA

    / "smartacademy_cnn_regresion.pth"

)

torch.save({

    "model_state_dict": modelo.state_dict(),

    "mejor_epoca": mejor_epoca,

    "mejor_perdida_validacion": mejor_perdida,

    "arquitectura": "CNNRegresionAcademica",

    "canales": 2,

    "parciales_entrada": 2,

    "variable_objetivo": "parcial3",

    "parametros_entrenables": total_parametros

}, ARCHIVO_MODELO)

print("\nMODELO GUARDADO:")

print(ARCHIVO_MODELO)


# ============================================================
# 18. GUARDAR HISTORIAL
# ============================================================

df_historial = pd.DataFrame(historial)

ARCHIVO_HISTORIAL = (

    RUTA_SALIDA

    / "historial_entrenamiento_regresion.csv"

)

df_historial.to_csv(

    ARCHIVO_HISTORIAL,

    index=False

)


# ============================================================
# 19. GUARDAR METRICAS
# ============================================================

metricas = {

    "proyecto": "SmartAcademy",

    "etapa": 18.2,

    "tipo_problema": "Regresion",

    "mejor_epoca": int(mejor_epoca),

    "mejor_perdida_validacion": float(mejor_perdida),

    "metricas_prueba": {

        "MAE": float(mae),

        "MSE": float(mse),

        "RMSE": float(rmse),

        "R2": float(r2)

    },

    "modelo_base": {

        "MAE": float(mae_base),

        "RMSE": float(rmse_base)

    },

    "tiempos": {

        "entrenamiento_segundos": float(
            tiempo_entrenamiento
        ),

        "inferencia_segundos": float(
            tiempo_inferencia
        )

    },

    "advertencia": (
        "Datos sinteticos. Los resultados no representan "
        "una validacion con estudiantes reales."
    )

}

ARCHIVO_METRICAS = (

    RUTA_SALIDA

    / "metricas_prediccion_cnn.json"

)

with open(

    ARCHIVO_METRICAS,

    "w",

    encoding="utf-8"

) as archivo:

    json.dump(

        metricas,

        archivo,

        indent=4,

        ensure_ascii=False

    )


# ============================================================
# 20. GRAFICAR HISTORIAL DE ENTRENAMIENTO
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(

    df_historial["epoca"],

    df_historial["perdida_train"],

    label="Entrenamiento"

)

plt.plot(

    df_historial["epoca"],

    df_historial["perdida_val"],

    label="Validacion"

)

plt.xlabel("Epoca")

plt.ylabel("MSE")

plt.title("SmartAcademy - Perdida de entrenamiento")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

ARCHIVO_GRAFICA = (

    RUTA_SALIDA

    / "curva_entrenamiento_regresion.png"

)

plt.savefig(

    ARCHIVO_GRAFICA,

    dpi=300

)

plt.close()


# ============================================================
# 21. GRAFICAR PREDICCIONES CONTRA VALORES REALES
# ============================================================

plt.figure(figsize=(8, 7))

plt.scatter(

    reales_test,

    predicciones_test,

    alpha=0.25

)

plt.plot(

    [0, 10],

    [0, 10],

    linestyle="--"

)

plt.xlabel("Calificacion real del parcial 3")

plt.ylabel("Calificacion predicha")

plt.title("SmartAcademy - Real vs Predicho")

plt.xlim(0, 10)

plt.ylim(0, 10)

plt.grid(alpha=0.3)

plt.tight_layout()

ARCHIVO_DISPERSION = (

    RUTA_SALIDA

    / "predicciones_real_vs_predicho.png"

)

plt.savefig(

    ARCHIVO_DISPERSION,

    dpi=300

)

plt.close()


# ============================================================
# 22. GUARDAR PREDICCIONES DE PRUEBA
# ============================================================

ARCHIVO_DATOS_ORIGINALES = np.load(

    ARCHIVO_DATOS,

    allow_pickle=False

)

ids_test = ARCHIVO_DATOS_ORIGINALES["ids_test"]

materias_test = ARCHIVO_DATOS_ORIGINALES["materias_test"]

df_predicciones = pd.DataFrame({

    "id_estudiante": ids_test,

    "materia": materias_test,

    "calificacion_real_p3": reales_test,

    "calificacion_predicha_p3": predicciones_test,

    "error_absoluto": np.abs(

        reales_test - predicciones_test

    )

})

ARCHIVO_PREDICCIONES = (

    RUTA_SALIDA

    / "predicciones_prueba_cnn.csv"

)

df_predicciones.to_csv(

    ARCHIVO_PREDICCIONES,

    index=False

)


# ============================================================
# 23. RESUMEN FINAL
# ============================================================

print("\n" + "=" * 65)

print("ETAPA 18.2 FINALIZADA")

print("=" * 65)

print("\nMETRICAS CNN:")

print(f"MAE:  {mae:.4f}")

print(f"RMSE: {rmse:.4f}")

print(f"R2:   {r2:.4f}")

print("\nCOMPARACION CONTRA MODELO BASE:")

print(f"MAE CNN:  {mae:.4f}")

print(f"MAE base: {mae_base:.4f}")

print("\nARCHIVOS GENERADOS:")

print(ARCHIVO_MODELO)

print(ARCHIVO_HISTORIAL)

print(ARCHIVO_METRICAS)

print(ARCHIVO_GRAFICA)

print(ARCHIVO_DISPERSION)

print(ARCHIVO_PREDICCIONES)

print("\nNO SE UTILIZO EL PARCIAL 3 COMO ENTRADA.")

print("=" * 65)
