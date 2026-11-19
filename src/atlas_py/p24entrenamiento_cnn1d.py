
# ============================================================
# PROYECTO: SMARTACADEMY
# ETAPA 16 - ENTRENAMIENTO DE UNA CNN 1D
#
# Archivo:
# p24entrenamiento_cnn1d.py
#
# Objetivo:
# 1. Cargar los datos preparados en la etapa 15.
# 2. Crear Dataset y DataLoader de PyTorch.
# 3. Construir una red neuronal convolucional 1D.
# 4. Entrenar con GPU CUDA cuando esté disponible.
# 5. Aplicar Early Stopping.
# 6. Evaluar el mejor modelo con datos de prueba.
# 7. Guardar el modelo y sus resultados.
#
# IMPORTANTE:
# La prueba no participa en el entrenamiento ni
# en la selección del mejor modelo.
# ============================================================


# ============================================================
# 1. IMPORTACION DE LIBRERIAS
# ============================================================

from pathlib import Path
import json
import copy
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 2. CONFIGURACION GENERAL
# ============================================================

# Ruta principal del proyecto.
RUTA_PROYECTO = Path(__file__).resolve().parents[1]

# Directorio con los datos preparados.
RUTA_DATOS = (
    RUTA_PROYECTO
    / "resultados_cnn"
)

# Archivo que contiene los tensores.
ARCHIVO_DATOS = (
    RUTA_DATOS
    / "smartacademy_datos_cnn.npz"
)

# Archivo con los metadatos de preparación.
ARCHIVO_METADATOS = (
    RUTA_DATOS
    / "metadatos_cnn.json"
)

# Directorio para guardar los resultados del entrenamiento.
RUTA_RESULTADOS = (
    RUTA_PROYECTO
    / "resultados_cnn"
)

RUTA_RESULTADOS.mkdir(
    parents=True,
    exist_ok=True
)

# Semilla para reproducibilidad.
SEMILLA = 42

# Número de clases que clasificará la red.
NUM_CLASES = 3

# Número de canales de entrada:
# 1. Calificaciones.
# 2. Asistencia.
CANALES_ENTRADA = 2

# Longitud de la secuencia:
# Tres parciales.
LONGITUD_SECUENCIA = 3

# Configuración del entrenamiento.
TAMANO_LOTE = 256

NUM_EPOCAS = 40

# Si la pérdida de validación no mejora durante
# estas épocas consecutivas, se detendrá el entrenamiento.
PACIENCIA_EARLY_STOPPING = 8

# Configuración del optimizador.
TASA_APRENDIZAJE = 0.001

# Mapeo de clases.
MAPA_ETIQUETAS = {
    "Descendente": 0,
    "Estable": 1,
    "Favorable": 2
}

NOMBRES_CLASES = [
    "Descendente",
    "Estable",
    "Favorable"
]


# ============================================================
# 3. CONFIGURAR REPRODUCIBILIDAD
# ============================================================

def configurar_semillas():

    """
    Configura las semillas aleatorias.

    Esto ayuda a que los experimentos sean
    más reproducibles.
    """

    random.seed(SEMILLA)

    np.random.seed(SEMILLA)

    torch.manual_seed(SEMILLA)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(SEMILLA)

    # Evitar algoritmos no deterministas cuando sea posible.
    # Puede reducir un poco el rendimiento, pero facilita
    # la comparación entre ejecuciones.
    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

    print("Semillas aleatorias configuradas.")


# ============================================================
# 4. SELECCIONAR DISPOSITIVO DE PROCESAMIENTO
# ============================================================

def seleccionar_dispositivo():

    """
    Determina si se utilizará GPU o CPU.

    CUDA permite ejecutar operaciones de tensores
    utilizando una GPU NVIDIA compatible.
    """

    print("\n1. CONFIGURANDO DISPOSITIVO")
    print("-" * 60)

    if torch.cuda.is_available():

        dispositivo = torch.device("cuda")

        print("GPU disponible.")

        print(
            "Dispositivo: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            "Memoria GPU total: "
            f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
        )

    else:

        dispositivo = torch.device("cpu")

        print(
            "CUDA no está disponible."
        )

        print(
            "Se utilizará el procesador CPU."
        )

    print(
        f"Dispositivo seleccionado: {dispositivo}"
    )

    return dispositivo


# ============================================================
# 5. CARGAR LOS DATOS PREPARADOS
# ============================================================

def cargar_datos():

    """
    Recupera los tensores generados en la etapa 15.

    Los archivos contienen:
    - X_train, y_train.
    - X_val, y_val.
    - X_test, y_test.
    """

    print("\n2. CARGANDO DATOS")
    print("-" * 60)

    # Comprobar que exista el archivo.
    if not ARCHIVO_DATOS.exists():

        raise FileNotFoundError(
            f"No existe el archivo: {ARCHIVO_DATOS}"
        )

    if not ARCHIVO_METADATOS.exists():

        raise FileNotFoundError(
            f"No existe el archivo: {ARCHIVO_METADATOS}"
        )

    # Cargar los tensores.
    # allow_pickle=False evita cargar objetos Python arbitrarios.
    datos = np.load(
        ARCHIVO_DATOS,
        allow_pickle=False
    )

    # Recuperar los conjuntos.
    X_train = datos["X_train"]
    y_train = datos["y_train"]

    X_val = datos["X_val"]
    y_val = datos["y_val"]

    X_test = datos["X_test"]
    y_test = datos["y_test"]

    # Cargar metadatos.
    with open(
        ARCHIVO_METADATOS,
        "r",
        encoding="utf-8"
    ) as archivo:

        metadatos = json.load(archivo)

    # Mostrar dimensiones.
    print(
        f"Entrenamiento: X={X_train.shape}, y={y_train.shape}"
    )

    print(
        f"Validación: X={X_val.shape}, y={y_val.shape}"
    )

    print(
        f"Prueba: X={X_test.shape}, y={y_test.shape}"
    )

    # Validar las dimensiones.
    for nombre, X, y in [
        ("Entrenamiento", X_train, y_train),
        ("Validación", X_val, y_val),
        ("Prueba", X_test, y_test)
    ]:

        if X.ndim != 3:

            raise ValueError(
                f"{nombre}: X debe tener tres dimensiones."
            )

        if X.shape[1:] != (
            CANALES_ENTRADA,
            LONGITUD_SECUENCIA
        ):

            raise ValueError(
                f"{nombre}: dimensiones incorrectas en X."
            )

        if len(X) != len(y):

            raise ValueError(
                f"{nombre}: X e y tienen tamaños diferentes."
            )

        if not np.isfinite(X).all():

            raise ValueError(
                f"{nombre}: existen valores inválidos en X."
            )

        if not np.isin(
            y,
            [0, 1, 2]
        ).all():

            raise ValueError(
                f"{nombre}: existen etiquetas inválidas."
            )

    print("Datos cargados y dimensiones verificadas.")

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        metadatos
    )


# ============================================================
# 6. CONSTRUIR DATASET Y DATALOADER
# ============================================================

def crear_dataloaders(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    dispositivo
):

    """
    Dataset:
        Almacena las muestras y sus etiquetas.

    DataLoader:
        Entrega los datos al modelo por lotes.

    El entrenamiento utilizará lotes de 256 registros.
    """

    print("\n3. CONSTRUYENDO DATASET Y DATALOADER")
    print("-" * 60)

    # Convertir matrices NumPy en tensores PyTorch.
    tensor_X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    tensor_y_train = torch.tensor(
        y_train,
        dtype=torch.long
    )

    tensor_X_val = torch.tensor(
        X_val,
        dtype=torch.float32
    )

    tensor_y_val = torch.tensor(
        y_val,
        dtype=torch.long
    )

    tensor_X_test = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    tensor_y_test = torch.tensor(
        y_test,
        dtype=torch.long
    )

    # Crear los conjuntos de datos.
    dataset_train = TensorDataset(
        tensor_X_train,
        tensor_y_train
    )

    dataset_val = TensorDataset(
        tensor_X_val,
        tensor_y_val
    )

    dataset_test = TensorDataset(
        tensor_X_test,
        tensor_y_test
    )

    # Usar memoria fijada cuando se trabaja con GPU.
    usar_pin_memory = (
        dispositivo.type == "cuda"
    )

    # Generador para controlar el orden aleatorio.
    generador = torch.Generator()

    generador.manual_seed(SEMILLA)

    # DataLoader de entrenamiento.
    loader_train = DataLoader(
        dataset_train,
        batch_size=TAMANO_LOTE,
        shuffle=True,
        num_workers=0,
        pin_memory=usar_pin_memory,
        generator=generador
    )

    # En validación no mezclamos las muestras.
    loader_val = DataLoader(
        dataset_val,
        batch_size=TAMANO_LOTE,
        shuffle=False,
        num_workers=0,
        pin_memory=usar_pin_memory
    )

    # En prueba tampoco mezclamos las muestras.
    loader_test = DataLoader(
        dataset_test,
        batch_size=TAMANO_LOTE,
        shuffle=False,
        num_workers=0,
        pin_memory=usar_pin_memory
    )

    print(
        f"Muestras de entrenamiento: {len(dataset_train):,}"
    )

    print(
        f"Lotes de entrenamiento: {len(loader_train)}"
    )

    print(
        f"Tamaño máximo de lote: {TAMANO_LOTE}"
    )

    print("Dataset y DataLoader preparados.")

    return (
        loader_train,
        loader_val,
        loader_test
    )


# ============================================================
# 7. DEFINIR ARQUITECTURA CNN 1D
# ============================================================

class SmartAcademyCNN(nn.Module):

    """
    Red neuronal convolucional unidimensional.

    Entrada:
        [lote, 2 canales, 3 parciales]

    Capas:

        Conv1d:
            Aprende patrones temporales entre parciales.

        BatchNorm1d:
            Ayuda a estabilizar las activaciones.

        ReLU:
            Introduce no linealidad.

        AdaptiveAvgPool1d:
            Resume las características temporales.

        Dropout:
            Reduce el sobreajuste.

        Linear:
            Genera tres puntuaciones de clasificación.
    """

    def __init__(self):

        super().__init__()

        # Primera convolución.
        #
        # Entrada: 2 canales.
        # Salida: 16 mapas de características.
        # Kernel 2: analiza pares de posiciones temporales.
        self.conv1 = nn.Conv1d(
            in_channels=2,
            out_channels=16,
            kernel_size=2,
            padding=1
        )

        self.bn1 = nn.BatchNorm1d(16)

        # Segunda convolución.
        #
        # Aprende combinaciones de características
        # generadas por la primera convolución.
        self.conv2 = nn.Conv1d(
            in_channels=16,
            out_channels=32,
            kernel_size=2
        )

        self.bn2 = nn.BatchNorm1d(32)

        # Resume la dimensión temporal a una posición.
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Dropout para reducir sobreajuste.
        self.dropout = nn.Dropout(
            p=0.25
        )

        # Capa final de clasificación.
        #
        # Tres neuronas:
        # 0 = Descendente.
        # 1 = Estable.
        # 2 = Favorable.
        self.clasificador = nn.Linear(
            in_features=32,
            out_features=NUM_CLASES
        )

    def forward(self, x):

        """
        Define el recorrido de los datos por la red.
        """

        # Primera convolución.
        x = self.conv1(x)

        x = self.bn1(x)

        x = torch.relu(x)

        # Segunda convolución.
        x = self.conv2(x)

        x = self.bn2(x)

        x = torch.relu(x)

        # Reducir la dimensión temporal.
        x = self.pool(x)

        # Eliminar la dimensión temporal final.
        x = torch.flatten(
            x,
            start_dim=1
        )

        # Aplicar regularización.
        x = self.dropout(x)

        # Obtener tres puntuaciones.
        x = self.clasificador(x)

        return x


# ============================================================
# 8. DEFINIR FUNCION DE ENTRENAMIENTO
# ============================================================

def entrenar_epoca(
    modelo,
    loader,
    criterio,
    optimizador,
    dispositivo
):

    """
    Ejecuta una época de entrenamiento.

    Una época significa que el modelo recorre
    una vez todos los registros de entrenamiento.
    """

    # Activar modo entrenamiento.
    modelo.train()

    perdida_acumulada = 0.0

    predicciones = []

    etiquetas_reales = []

    total = 0

    # Recorrer los lotes.
    for entradas, etiquetas in loader:

        # Enviar datos a GPU o CPU.
        entradas = entradas.to(
            dispositivo,
            non_blocking=True
        )

        etiquetas = etiquetas.to(
            dispositivo,
            non_blocking=True
        )

        # Limpiar gradientes anteriores.
        optimizador.zero_grad(
            set_to_none=True
        )

        # Propagación hacia adelante.
        salidas = modelo(entradas)

        # Calcular pérdida.
        perdida = criterio(
            salidas,
            etiquetas
        )

        # Propagación hacia atrás.
        perdida.backward()

        # Actualizar parámetros.
        optimizador.step()

        # Acumular pérdida ponderada por tamaño del lote.
        tamano_real = entradas.size(0)

        perdida_acumulada += (
            perdida.item() * tamano_real
        )

        total += tamano_real

        # Obtener la clase con mayor puntuación.
        clases_predichas = torch.argmax(
            salidas,
            dim=1
        )

        predicciones.extend(
            clases_predichas.detach().cpu().numpy()
        )

        etiquetas_reales.extend(
            etiquetas.detach().cpu().numpy()
        )

    perdida_promedio = (
        perdida_acumulada / total
    )

    exactitud = accuracy_score(
        etiquetas_reales,
        predicciones
    )

    return perdida_promedio, exactitud


# ============================================================
# 9. DEFINIR FUNCION DE VALIDACION
# ============================================================

def evaluar_perdida(
    modelo,
    loader,
    criterio,
    dispositivo
):

    """
    Evalúa la pérdida de validación.

    No actualiza los pesos del modelo.
    """

    # Desactivar el comportamiento de entrenamiento.
    modelo.eval()

    perdida_acumulada = 0.0

    total = 0

    # No calcular gradientes durante la validación.
    with torch.no_grad():

        for entradas, etiquetas in loader:

            entradas = entradas.to(
                dispositivo,
                non_blocking=True
            )

            etiquetas = etiquetas.to(
                dispositivo,
                non_blocking=True
            )

            salidas = modelo(entradas)

            perdida = criterio(
                salidas,
                etiquetas
            )

            tamano_real = entradas.size(0)

            perdida_acumulada += (
                perdida.item() * tamano_real
            )

            total += tamano_real

    return perdida_acumulada / total


# ============================================================
# 10. EVALUAR MODELO CON DATOS DE PRUEBA
# ============================================================

def evaluar_prueba(
    modelo,
    loader_test,
    dispositivo
):

    """
    Realiza la evaluación final con el conjunto de prueba.

    Se utiliza el mejor modelo encontrado durante
    el entrenamiento y la validación.
    """

    print("\n6. EVALUACION FINAL DEL MODELO")
    print("-" * 60)

    modelo.eval()

    predicciones = []

    etiquetas_reales = []

    # Desactivar gradientes.
    with torch.no_grad():

        for entradas, etiquetas in loader_test:

            entradas = entradas.to(
                dispositivo,
                non_blocking=True
            )

            # Obtener puntuaciones.
            salidas = modelo(entradas)

            # Elegir la clase con mayor puntuación.
            clases = torch.argmax(
                salidas,
                dim=1
            )

            predicciones.extend(
                clases.cpu().numpy()
            )

            etiquetas_reales.extend(
                etiquetas.numpy()
            )

    # Calcular métricas.
    exactitud = accuracy_score(
        etiquetas_reales,
        predicciones
    )

    precision = precision_score(
        etiquetas_reales,
        predicciones,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        etiquetas_reales,
        predicciones,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        etiquetas_reales,
        predicciones,
        average="macro",
        zero_division=0
    )

    # Matriz de confusión.
    matriz = confusion_matrix(
        etiquetas_reales,
        predicciones,
        labels=[0, 1, 2]
    )

    # Reporte por categoría.
    reporte = classification_report(
        etiquetas_reales,
        predicciones,
        labels=[0, 1, 2],
        target_names=NOMBRES_CLASES,
        zero_division=0
    )

    # Mostrar métricas.
    print(
        f"Accuracy:  {exactitud:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1-score:  {f1:.4f}"
    )

    print("\nMATRIZ DE CONFUSION")

    print(matriz)

    print("\nREPORTE DE CLASIFICACION")

    print(reporte)

    return (
        {
            "accuracy": float(exactitud),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(f1)
        },
        matriz,
        reporte
    )


# ============================================================
# 11. GUARDAR MODELO Y RESULTADOS
# ============================================================

def guardar_resultados(
    modelo,
    historial,
    matriz,
    reporte,
    metricas,
    metadatos,
    mejor_epoca,
    mejor_perdida
):

    """
    Guarda el modelo y los resultados del experimento.
    """

    print("\n7. GUARDANDO MODELO Y RESULTADOS")
    print("-" * 60)

    # Ruta del modelo entrenado.
    ruta_modelo = (
        RUTA_RESULTADOS
        / "smartacademy_cnn1d.pth"
    )

    # Guardar estado del modelo y configuración.
    checkpoint = {

        "model_state_dict": modelo.state_dict(),

        "arquitectura": "SmartAcademyCNN",

        "canales_entrada": CANALES_ENTRADA,

        "longitud_secuencia": LONGITUD_SECUENCIA,

        "numero_clases": NUM_CLASES,

        "mapa_etiquetas": MAPA_ETIQUETAS,

        "metadatos_preparacion": metadatos,

        "mejor_epoca": mejor_epoca,

        "mejor_perdida_validacion": mejor_perdida,

        "metricas_prueba": metricas,

        "version_pytorch": torch.__version__
    }

    torch.save(
        checkpoint,
        ruta_modelo
    )

    # Guardar historial de entrenamiento.
    ruta_historial = (
        RUTA_RESULTADOS
        / "historial_entrenamiento_cnn.csv"
    )

    pd.DataFrame(
        historial
    ).to_csv(
        ruta_historial,
        index=False
    )

    # Guardar matriz de confusión.
    ruta_matriz = (
        RUTA_RESULTADOS
        / "matriz_confusion_cnn.csv"
    )

    pd.DataFrame(
        matriz,
        index=NOMBRES_CLASES,
        columns=NOMBRES_CLASES
    ).to_csv(
        ruta_matriz,
        index=True
    )

    # Guardar reporte.
    ruta_reporte = (
        RUTA_RESULTADOS
        / "reporte_clasificacion_cnn.txt"
    )

    with open(
        ruta_reporte,
        "w",
        encoding="utf-8"
    ) as archivo:

        archivo.write(reporte)

    # Guardar métricas en JSON.
    ruta_metricas = (
        RUTA_RESULTADOS
        / "metricas_cnn.json"
    )

    with open(
        ruta_metricas,
        "w",
        encoding="utf-8"
    ) as archivo:

        json.dump(
            {
                "mejor_epoca": mejor_epoca,
                "mejor_perdida_validacion": mejor_perdida,
                "metricas_prueba": metricas
            },
            archivo,
            indent=4,
            ensure_ascii=False
        )

    print(f"Modelo: {ruta_modelo}")

    print(f"Historial: {ruta_historial}")

    print(f"Matriz: {ruta_matriz}")

    print(f"Reporte: {ruta_reporte}")

    print(f"Métricas: {ruta_metricas}")

    print("Archivos guardados correctamente.")


# ============================================================
# 12. PROCESO PRINCIPAL DE ENTRENAMIENTO
# ============================================================

def main():

    """
    Coordina el entrenamiento completo de SmartAcademy CNN.
    """

    print("=" * 65)

    print("SMARTACADEMY")

    print("ETAPA 16 - ENTRENAMIENTO CNN 1D")

    print("=" * 65)

    configurar_semillas()

    # Seleccionar GPU o CPU.
    dispositivo = seleccionar_dispositivo()

    # Cargar tensores.
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        metadatos
    ) = cargar_datos()

    # Crear DataLoaders.
    (
        loader_train,
        loader_val,
        loader_test
    ) = crear_dataloaders(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        dispositivo
    )

    # Crear modelo.
    print("\n4. CONSTRUYENDO CNN 1D")
    print("-" * 60)

    modelo = SmartAcademyCNN().to(
        dispositivo
    )

    print(modelo)

    # Obtener cantidad de parámetros entrenables.
    parametros = sum(
        parametro.numel()
        for parametro in modelo.parameters()
        if parametro.requires_grad
    )

    print(
        f"\nParámetros entrenables: {parametros:,}"
    )

    # Función de pérdida para clasificación multiclase.
    criterio = nn.CrossEntropyLoss()

    # Optimizador Adam.
    optimizador = torch.optim.Adam(
        modelo.parameters(),
        lr=TASA_APRENDIZAJE
    )

    # Reducir la tasa de aprendizaje cuando
    # la pérdida de validación deje de mejorar.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizador,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=0.00001
    )

    print("\n5. INICIANDO ENTRENAMIENTO")
    print("-" * 60)

    historial = []

    mejor_perdida = float("inf")

    mejor_epoca = 0

    mejor_estado = None

    epocas_sin_mejora = 0

    for epoca in range(1, NUM_EPOCAS + 1):

        # Entrenar una época.
        perdida_train, accuracy_train = entrenar_epoca(
            modelo,
            loader_train,
            criterio,
            optimizador,
            dispositivo
        )

        # Evaluar en validación.
        perdida_val = evaluar_perdida(
            modelo,
            loader_val,
            criterio,
            dispositivo
        )

        # Actualizar tasa de aprendizaje.
        scheduler.step(
            perdida_val
        )

        tasa_actual = optimizador.param_groups[0]["lr"]

        # Guardar datos de la época.
        historial.append(
            {
                "epoca": epoca,
                "perdida_train": perdida_train,
                "accuracy_train": accuracy_train,
                "perdida_val": perdida_val,
                "tasa_aprendizaje": tasa_actual
            }
        )

        print(
            f"Época {epoca:02d}/{NUM_EPOCAS} | "
            f"Loss Train: {perdida_train:.4f} | "
            f"Accuracy Train: {accuracy_train:.4f} | "
            f"Loss Val: {perdida_val:.4f} | "
            f"LR: {tasa_actual:.6f}"
        )

        # Comprobar si mejoró la validación.
        if perdida_val < mejor_perdida:

            mejor_perdida = perdida_val

            mejor_epoca = epoca

            # Copiar los pesos actuales del modelo.
            mejor_estado = copy.deepcopy(
                modelo.state_dict()
            )

            epocas_sin_mejora = 0

            print(
                "  -> Nuevo mejor modelo guardado en memoria."
            )

        else:

            epocas_sin_mejora += 1

        # Detención temprana.
        if (
            epocas_sin_mejora
            >= PACIENCIA_EARLY_STOPPING
        ):

            print(
                "\nEarly Stopping activado."
            )

            print(
                "La pérdida de validación dejó de mejorar."
            )

            break

    # Verificar que se haya encontrado un modelo válido.
    if mejor_estado is None:

        raise RuntimeError(
            "No se obtuvo un modelo válido."
        )

    # Recuperar los mejores pesos.
    modelo.load_state_dict(
        mejor_estado
    )

    print("\nMEJOR MODELO")

    print(
        f"Época seleccionada: {mejor_epoca}"
    )

    print(
        f"Mejor pérdida de validación: {mejor_perdida:.4f}"
    )

    # Evaluar una sola vez con prueba.
    (
        metricas,
        matriz,
        reporte
    ) = evaluar_prueba(
        modelo,
        loader_test,
        dispositivo
    )

    # Guardar el modelo y sus resultados.
    guardar_resultados(
        modelo,
        historial,
        matriz,
        reporte,
        metricas,
        metadatos,
        mejor_epoca,
        mejor_perdida
    )

    print("\n" + "=" * 65)

    print(
        "ETAPA 16 FINALIZADA CORRECTAMENTE"
    )

    print(
        "CNN 1D entrenada y evaluada."
    )

    print("=" * 65)


# ============================================================
# 13. PUNTO DE ENTRADA
# ============================================================

if __name__ == "__main__":

    main()
