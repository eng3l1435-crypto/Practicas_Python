"""
==============================================================
Primer Gráfico con Matplotlib
==============================================================
Materia: Extracción de Conocimiento en Bases de Datos
Objetivo: Aprender a construir una gráfica utilizando
Matplotlib.
==============================================================
"""

# ==========================================================
# IMPORTACIÓN DE LIBRERÍAS
# ==========================================================
import matplotlib.pyplot as plt
import numpy as np

# ==========================================================
# DATOS
# ==========================================================
# Meses del año
meses = [
    "Ene",
    "Feb",
    "Mar",
    "Abr",
    "May",
    "Jun",
    "Jul",
    "Ago",
    "Sep",
    "Oct",
    "Nov",
    "Dic",
]

# Ventas simuladas (miles de pesos)
ventas = [85, 90, 96, 105, 120, 135, 150, 148, 160, 172, 181, 195]

# ==========================================================
# CONFIGURACIÓN GENERAL: ESTABLECE EL TAMAÑO DE LA FIGURA
# ==========================================================
plt.figure(figsize=(14, 8))

# ==========================================================
# GRÁFICA PRINCIPAL: DIBUJA LA LÍNEA CON COLOR Y MARCADORES
# ==========================================================
plt.plot(
    meses,
    ventas,
    color="#1f77b4",
    linewidth=3,
    marker="o",
    markersize=8,
    markerfacecolor="white",
    markeredgewidth=2,
    label="Ventas mensuales",
)

# ==========================================================
# TÍTULO PRINCIPAL
# ==========================================================
plt.title(
    "Ventas Mensuales de la Empresa EcoMart",
    fontsize=22,
    fontweight="bold",
    pad=45,
)

# ==========================================================
# SUBTÍTULO CON CONTEXTO
# ==========================================================
plt.suptitle(
    "Periodo Enero - Diciembre 2026", 
    fontsize=13, color="gray", y=.9
)

# ==========================================================
# ETIQUETAS EN LOS EJES
# ==========================================================
plt.xlabel("Mes", fontsize=15, fontweight="bold")
plt.ylabel("Ventas (Miles de pesos)", fontsize=15, fontweight="bold")

# ==========================================================
# CUADRÍCULA
# ==========================================================
plt.grid(linestyle="--", linewidth=0.7, alpha=0.6)

# ==========================================================
# LEYENDA
# ==========================================================
plt.legend(fontsize=12, loc="upper left")

# ==========================================================
# ANOTACIONES: MUESTRA EL VALOR EXACTO DE CADA OBSERVACIÓN
# ==========================================================
for x, y in zip(meses, ventas):
    plt.text(x, y + 2, str(y), fontsize=10, ha="center")

# ==========================================================
# LÍNEA DE PROMEDIO
# ==========================================================
promedio = np.mean(ventas)
plt.axhline(
    promedio, color="green", linestyle="--", linewidth=2, label="Promedio", alpha=0.2
)
plt.text(10.5, 
         promedio + 2, 
         f"Promedio = {promedio:.1f}", 
         color="red", 
         fontsize=11,
         alpha = 0.6
         )

# ==========================================================
# DESTACAR EL MÁXIMO
# ==========================================================
indice_max = np.argmax(ventas)

plt.scatter(
    indice_max,
    ventas[indice_max],
    color="#1b7ec5",  
    marker="o", 
    s=9,
    zorder=5,
    label="Máximo",
)

plt.annotate(
    "Mayor venta del año",
    xy=(indice_max, ventas[indice_max]),
    xytext=(8, 185),
    arrowprops=dict(arrowstyle="-", color="#14ad28", lw=2, alpha=0.4),
    fontsize=12,
)

# ==========================================================
# PIE DE FIGURA
# ==========================================================
plt.figtext(
    0.01,
    0.01,
    "Conclusión: Estimado de las ventas | Elaboración propia | UTVT",
    fontsize=10,
    color="gray",
)

# ==========================================================
# QUITAR MARCO SUPERIOR Y DERECHO
# ==========================================================
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ==========================================================
# AJUSTAR ESPACIOS Y GUARDAR
# ==========================================================
plt.tight_layout()
plt.savefig("Grafica_Profesional_01.png", dpi=300, bbox_inches="tight")

# ==========================================================
# MOSTRAR
# ==========================================================
plt.show()

print("=" * 60)
print("Gráfica generada correctamente.")
print("Archivo guardado como:")
print("Grafica_Profesional_01.png")
print("=" * 60)