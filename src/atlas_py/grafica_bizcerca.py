"""
============================================================
MÓDULO DE ANALÍTICA Y VISUALIZACIÓN DE DATOS - BIZCERCA
Proyecto: Plataforma de Directorio Inteligente
Universidad Tecnológica del Valle de Toluca (UTVT)
============================================================
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Directorio de salida para la analítica del proyecto
CARPETA_SALIDA = Path("salida_graficas_bizcerca")
CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

# Paleta visual institucional adaptada a BizCerca
COLORES = {
    "primario": "#2563EB",     # Azul
    "secundario": "#0D9488",   # Turquesa (Éxito / Geoposicionamiento)
    "alerta": "#F59E0B",       # Naranja (Vitalidad)
    "peligro": "#DC2626",      # Rojo (Inactividad / Umbrales)
    "fondo": "#FFFFFF",
    "texto": "#1E293B",
}

# Configuración global de apariencia para reportes técnicos
plt.rcParams.update({
    "figure.facecolor": COLORES["fondo"],
    "axes.facecolor": COLORES["fondo"],
    "axes.edgecolor": "#CBD5E1",
    "axes.labelcolor": COLORES["texto"],
    "text.color": COLORES["texto"],
    "xtick.color": COLORES["texto"],
    "ytick.color": COLORES["texto"],
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "font.family": "DejaVu Sans",
    "grid.color": "#E2E8F0",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})

def guardar_grafica(figura, nombre_archivo):
    ruta = CARPETA_SALIDA / nombre_archivo
    figura.tight_layout()
    figura.savefig(ruta, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Métrica generada y guardada: {ruta}")
    plt.close(figura)

# ============================================================
# DATOS SIMULADOS BASADOS EN LOS REQUERIMIENTOS DE BIZCERCA
# ============================================================
rng = np.random.default_rng(42)

# 1. Simulación de Matchmaking (Similitud semántica vs Puntuación de coincidencia)
similitud_nlp = rng.uniform(0.3, 1.0, 150)
score_match = np.clip(similitud_nlp * 85 + rng.normal(0, 5, 150), 0, 100)

# 2. Simulación Geoespacial (Distancias en km de los oferentes al consumidor - RF-04)
distancias_km = rng.exponential(scale=3.0, size=200)
distancias_km = distancias_km[distancias_km <= 15] # Filtrado dentro de un radio lógico

# 3. Motor de Vitalidad y Gobernanza (Días de inactividad vs Multiplicador - RNF-04)
dias_inactividad = rng.integers(0, 45, size=100)
multiplicador_visibilidad = np.where(dias_inactividad <= 30, 1.0 - (dias_inactividad / 45) * 0.4, 0.1)

# 4. Distribución por tipo de perfil en la plataforma
tipos_perfil = ["Microempresas", "Oficios Técnicos", "Estudiantes UTVT"]
conteo_perfiles = [140, 95, 115]


# ============================================================
# GRÁFICA 1: Precisión del Matchmaking Semántico (IA)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(
    similitud_nlp * 100, score_match,
    color=COLORES["primario"], alpha=0.7, edgecolors="white", s=60
)
ax.set_title("BizCerca: Relación entre Similitud Semántica y Score de Match")
ax.set_xlabel("Índice de Similitud Semántica (%)")
ax.set_ylabel("Puntuación de Coincidencia Final")
ax.grid(True)
guardar_grafica(fig, "01_matchmaking_ia.png")


# ============================================================
# GRÁFICA 2: Análisis de Proximidad Geoespacial (RF-04)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(
    distancias_km, bins=12,
    color=COLORES["secundario"], edgecolor="white", alpha=0.85
)
ax.axvline(5.0, color=COLORES["peligro"], linestyle="--", linewidth=2, label="Radio óptimo de búsqueda (5 km)")
ax.set_title("BizCerca: Concentración de Oferentes por Radio Geográfico")
ax.set_xlabel("Distancia al consumidor (Kilómetros)")
ax.set_ylabel("Número de Oferentes Disponibles")
ax.legend()
guardar_grafica(fig, "02_geolocalizacion_radio.png")


# ============================================================
# GRÁFICA 3: Gobernanza y Motor de Vitalidad (RNF-04)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(
    dias_inactividad, multiplicador_visibilidad,
    color=COLORES["alerta"], alpha=0.8, s=50, edgecolors="white"
)
ax.axvline(30, color=COLORES["peligro"], linestyle=":", linewidth=2, label="Umbral de Inactividad (30 días -> Oculto)")
ax.set_title("BizCerca: Impacto de la Inactividad en la Visibilidad del Oferente")
ax.set_xlabel("Días sin actualización de perfil")
ax.set_ylabel("Factor Multiplicador de Visibilidad")
ax.legend()
guardar_grafica(fig, "03_vitalidad_gobernanza.png")


# ============================================================
# GRÁFICA 4: Composición de Oferentes en la Plataforma
# ============================================================
fig, ax = plt.subplots(figsize=(7, 7))
ax.pie(
    conteo_perfiles, labels=tipos_perfil, autopct="%1.1f%%",
    startangle=90, colors=[COLORES["primario"], COLORES["secundario"], COLORES["alerta"]],
    wedgeprops={"edgecolor": "white", "linewidth": 2}
)
ax.set_title("BizCerca: Composición de Perfiles Registrados")
ax.axis("equal")
guardar_grafica(fig, "04_composicion_oferentes.png")

print("\n¡Simulación de analítica para BizCerca completada con éxito!")