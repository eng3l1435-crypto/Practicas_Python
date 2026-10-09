import matplotlib.pyplot as plt
import numpy as np

# ==========================================================
# DATOS
# ==========================================================
meses = [
    "Ene", "Feb", "Mar", "Abr", "May", "Jun",
    "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"
]
ventas = [85, 90, 96, 105, 120, 135, 150, 148, 160, 172, 181, 195]

# ==========================================================
# CONFIGURACIÓN Y GRÁFICA BASE
# ==========================================================
plt.figure(figsize=(14, 8))

# Títulos
plt.title("Ventas Mensuales de la Empresa EcoMart", fontsize=22, fontweight="bold", pad=30)
plt.suptitle("Periodo Enero - Diciembre 2026", fontsize=13, color="gray", y=0.94)

# Línea principal
plt.plot(meses, ventas, color="#1f77b4", linewidth=3, marker="o", markersize=8, 
         markerfacecolor="white", markeredgewidth=2, label="Ventas mensuales")

# ==========================================================
# ELEMENTOS VISUALES Y ANALÍTICOS
# ==========================================================
# Ejes y cuadrícula
plt.xlabel("Mes", fontsize=15, fontweight="bold")
plt.ylabel("Ventas (Miles de pesos)", fontsize=15, fontweight="bold")
plt.grid(linestyle="--", linewidth=0.7, alpha=0.6)
plt.legend(fontsize=12, loc="upper left")

# Etiquetas de datos sobre cada punto
for x, y in zip(meses, ventas):
    plt.text(x, y + 5, str(y), fontsize=10, ha="center", color="black") # Subí un poco la etiqueta (+5)

# Línea de promedio
promedio = np.mean(ventas)
plt.axhline(promedio, color="red", linestyle="--", linewidth=2, label="Promedio")
plt.text(11.5, promedio + 2, f"Promedio = {promedio:.1f}", color="red", fontsize=11, ha="right")

# ==========================================================
# DESTACAR MÁXIMO (SOLUCIÓN DE SUPERPOSICIÓN)
# ==========================================================
indice_max = np.argmax(ventas)
mes_max = meses[indice_max]
venta_max = ventas[indice_max]

# Gráfica de dispersión para el punto verde (ligeramente desplazado a la derecha para visibilidad)
# Usamos indice_max + 0.1 para moverlo sutilmente y que no tape al azul
plt.scatter(indice_max + 0.1, venta_max, color="green", s=200, zorder=5, label="Máximo")

# Anotación con flecha apuntando al punto verde
plt.annotate('Mayor venta\ndel año', 
             xy=(indice_max + 0.1, venta_max), # Apunta al punto desplazado
             xytext=(indice_max + 0.5, venta_max + 10), # Texto más arriba y a la derecha
             arrowprops=dict(arrowstyle="->", color="green", lw=2, connectionstyle="arc3,rad=0.2"),
             fontsize=12, color="green", fontweight="bold")

# ==========================================================
# FINALIZAR Y GUARDAR
# ==========================================================
# Quitar marcos
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("Grafica_EcoMart_2026.png", dpi=300, bbox_inches="tight")
plt.show()