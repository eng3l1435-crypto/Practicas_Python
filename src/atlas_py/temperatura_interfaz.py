import tkinter as tk
from tkinter import messagebox, ttk


class AgenteClimatizacionGUI:
  """Modelo de un Agente Reactivo Simple con Interfaz Gráfica."""

  def __init__(self, root):
    self.root = root
    self.root.title("Agente Reactivo - Climatización")
    self.root.geometry("420x300")
    self.root.resizable(False, False)

    self.temperatura = 0.0
    self.humedad = 0.0
    self.accion = ""

    # --- DISEÑO DE LA INTERFAZ ---
    # Título superior
    ttk.Label(
        root, text="Panel de Control del Agente", font=("Arial", 12, "bold")
    ).pack(pady=10)

    # Marco para entradas
    frame_inputs = ttk.LabelFrame(root, text=" Percepción del Entorno ")
    frame_inputs.pack(fill="x", padx=20, pady=10)

    ttk.Label(frame_inputs, text="Temperatura actual (°C):").grid(
        row=0, column=0, sticky="w", padx=10, pady=8
    )
    self.entry_temp = ttk.Entry(frame_inputs, width=15)
    self.entry_temp.grid(row=0, column=1, padx=10, pady=8)

    ttk.Label(frame_inputs, text="Humedad actual (%):").grid(
        row=1, column=0, sticky="w", padx=10, pady=8
    )
    self.entry_hum = ttk.Entry(frame_inputs, width=15)
    self.entry_hum.grid(row=1, column=1, padx=10, pady=8)

    # Botón de acción principal
    ttk.Button(
        root, text="Evaluar y Tomar Decisión", command=self.ejecutar_ciclo
    ).pack(pady=5)

    # Marco para mostrar el resultado
    frame_result = ttk.LabelFrame(root, text=" Resultado del Agente ")
    frame_result.pack(fill="both", expand=True, padx=20, pady=10)

    self.lbl_resultado = ttk.Label(
        frame_result,
        text="Esperando datos del entorno...",
        font=("Arial", 10, "bold"),
        foreground="gray",
        wraplength=350,
        justify="center",
    )
    self.lbl_resultado.pack(expand=True, padx=10, pady=10)

  def percibir(self):
    """Captura los datos desde la interfaz y valida que sean numéricos."""
    try:
      self.temperatura = float(self.entry_temp.get())
      self.humedad = float(self.entry_hum.get())
      return True
    except ValueError:
      messagebox.showerror(
          "Error de Validación",
          "Por favor, ingresa únicamente números válidos en temperatura y"
          " humedad.",
      )
      return False

  def tomar_decision(self):
    """Aplica la regla condición-acción basada en la percepción."""
    if self.temperatura > 30 and self.humedad > 70:
      self.accion = "Encender aire acondicionado (Modo Deshumidificador)"
    elif self.temperatura > 30:
      self.accion = "Encender ventilador"
    elif self.temperatura < 18:
      self.accion = "Encender calefacción"
    else:
      self.accion = "Mantener sistema apagado"

  def mostrar_resultado(self):
    """Muestra el estado del agente y la acción directamente en la interfaz."""
    texto_resumen = (
        f"Percepción -> Temp: {self.temperatura}°C | Humedad:"
        f" {self.humedad}%\nAcción -> {self.accion}"
    )
    self.lbl_resultado.config(text=texto_resumen, foreground="#004d40")

  def ejecutar_ciclo(self):
    """Ciclo completo del agente disparado por el botón."""
    if self.percibir():
      self.tomar_decision()
      self.mostrar_resultado()


# --- EJECUCIÓN DE LA APLICACIÓN ---
if __name__ == "__main__":
  root = tk.Tk()
  app = AgenteClimatizacionGUI(root)
  root.mainloop()

# Abstracción: Se modeló el agente como un Agente Reactivo Simple usando una clase (AgenteClimatizacion),
# reflejando mejor el concepto teórico de la práctica.
# Validación de entradas: El método _leer_float evita que el programa se cierre si el usuario escribe
# caracteres no numéricos.
# Escalabilidad: Si en el futuro necesitas agregar más sensores (como calidad de aire o luz),
# solo debes extender el método percibir() y agregar condiciones en tomar_decision().