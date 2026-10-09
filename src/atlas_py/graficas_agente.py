from bson.objectid import ObjectId
import os
from dotenv import load_dotenv
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from pymongo import MongoClient
from urllib.parse import quote_plus
import tkinter as tk
from tkinter import messagebox, ttk

# --- CARGAR VARIABLES DE ENTORNO ---
load_dotenv()

usuario = quote_plus(os.getenv("MONGO_USER"))
password = quote_plus(os.getenv("MONGO_PASSWORD"))
cluster = os.getenv("MONGO_CLUSTER")
nombre_db = os.getenv("MONGO_DB")
nombre_coleccion = os.getenv("MONGO_COLLECTION")

uri = f"mongodb+srv://{usuario}:{password}@{cluster}/"

class AgenteClimatizacionApp:

  def __init__(self, root):
    self.root = root
    self.root.title("Agente Climatización")
    self.root.geometry("1100x650")

    self.temperatura = 0.0
    self.humedad = 0.0
    self.accion = ""
    self.id_seleccionado = None

    # Variables de control para los filtros de las gráficas (Checkbuttons)
    self.ver_aire = tk.BooleanVar(value=True)
    self.ver_ventilador = tk.BooleanVar(value=True)
    self.ver_calefaccion = tk.BooleanVar(value=True)
    self.ver_apagado = tk.BooleanVar(value=True)

    # --- CONEXIÓN MONGODB ---
    try:
      self.cliente = MongoClient(uri)
      self.db = self.cliente[nombre_db]
      self.coleccion = self.db[nombre_coleccion]
    except Exception as e:
      messagebox.showerror("Error de Conexión", f"No se pudo conectar: {e}")
      self.root.destroy()
      return

    # --- DISEÑO DE LA INTERFAZ (Layout de dos columnas) ---
    left_frame = ttk.Frame(root)
    left_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

    right_frame = ttk.Frame(root)
    right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

    # 1. FORMULARIO (Columna Izquierda)
    form_frame = ttk.LabelFrame(left_frame, text=" Percepción del Entorno ")
    form_frame.pack(fill="x", padx=5, pady=5)

    ttk.Label(form_frame, text="Temperatura (°C):").grid(
        row=0, column=0, sticky="w", padx=8, pady=6
    )
    self.entry_temp = ttk.Entry(form_frame, width=12)
    self.entry_temp.grid(row=0, column=1, padx=8, pady=6)

    ttk.Label(form_frame, text="Humedad (%):").grid(
        row=0, column=2, sticky="w", padx=8, pady=6
    )
    self.entry_hum = ttk.Entry(form_frame, width=12)
    self.entry_hum.grid(row=0, column=3, padx=8, pady=6)

    # Botones CRUD
    btn_frame = ttk.Frame(left_frame)
    btn_frame.pack(fill="x", padx=5, pady=5)

    ttk.Button(
        btn_frame, text="Evaluar y Crear", command=self.crear_registro
    ).pack(side="left", padx=2)
    ttk.Button(
        btn_frame, text="Actualizar", command=self.actualizar_registro
    ).pack(side="left", padx=2)
    ttk.Button(
        btn_frame, text="Eliminar", command=self.eliminar_registro
    ).pack(side="left", padx=2)
    ttk.Button(btn_frame, text="Limpiar", command=self.limpiar_form).pack(
        side="left", padx=2
    )

    # Tabla (Treeview)
    table_container = ttk.Frame(left_frame)
    table_container.pack(fill="both", expand=True, padx=5, pady=5)

    self.tree = ttk.Treeview(
        table_container,
        columns=("ID", "Temperatura", "Humedad", "Acción"),
        show="headings",
    )
    self.tree.heading("ID", text="ID")
    self.tree.heading("Temperatura", text="Temp")
    self.tree.heading("Humedad", text="Humedad")
    self.tree.heading("Acción", text="Acción")

    self.tree.column("ID", width=140, anchor="center")
    self.tree.column("Temperatura", width=60, anchor="center")
    self.tree.column("Humedad", width=60, anchor="center")
    self.tree.column("Acción", width=220, anchor="w")

    self.tree.pack(side="left", fill="both", expand=True)
    self.tree.bind("<<TreeviewSelect>>", self.seleccionar_registro)

    scrollbar = ttk.Scrollbar(
        table_container, orient="vertical", command=self.tree.yview
    )
    scrollbar.pack(side="right", fill="y")
    self.tree.configure(yscrollcommand=scrollbar.set)

    # 2. SECCIÓN DE GRÁFICAS (Columna Derecha)
    graph_filter_frame = ttk.LabelFrame(
        right_frame, text=" Filtros de Gráfica por Acción "
    )
    graph_filter_frame.pack(fill="x", padx=5, pady=5)

    # Casillas de verificación para cada acción con su color distintivo representativo
    ttk.Checkbutton(
        graph_filter_frame,
        text="Aire Acondicionado",
        variable=self.ver_aire,
        command=self.actualizar_grafica,
    ).grid(row=0, column=0, sticky="w", padx=5, pady=3)
    ttk.Checkbutton(
        graph_filter_frame,
        text="Ventilador",
        variable=self.ver_ventilador,
        command=self.actualizar_grafica,
    ).grid(row=0, column=1, sticky="w", padx=5, pady=3)
    ttk.Checkbutton(
        graph_filter_frame,
        text="Calefacción",
        variable=self.ver_calefaccion,
        command=self.actualizar_grafica,
    ).grid(row=1, column=0, sticky="w", padx=5, pady=3)
    ttk.Checkbutton(
        graph_filter_frame,
        text="Sistema Apagado",
        variable=self.ver_apagado,
        command=self.actualizar_grafica,
    ).grid(row=1, column=1, sticky="w", padx=5, pady=3)

    # Contenedor de Matplotlib dentro de Tkinter
    self.fig, self.ax = plt.subplots(figsize=(5, 4))
    self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
    self.canvas.get_tk_widget().pack(
        fill="both", expand=True, padx=5, pady=5
    )

    self.cargar_datos()

  def percibir(self):
    try:
      self.temperatura = float(self.entry_temp.get())
      self.humedad = float(self.entry_hum.get())
      return True
    except ValueError:
      messagebox.showerror(
          "Error de Validación",
          "Ingresa valores numéricos válidos en temperatura y humedad.",
      )
      return False

  def tomar_decision(self):
    if self.temperatura > 30 and self.humedad > 70:
      self.accion = "Encender aire acondicionado (Modo Deshumidificador)"
    elif self.temperatura > 30:
      self.accion = "Encender ventilador"
    elif self.temperatura < 18:
      self.accion = "Encender calefacción"
    else:
      self.accion = "Mantener sistema apagado"

  def cargar_datos(self):
    for item in self.tree.get_children():
      self.tree.delete(item)
    try:
      for doc in self.coleccion.find():
        self.tree.insert(
            "",
            "end",
            values=(
                str(doc.get("_id")),
                doc.get("temperatura"),
                doc.get("humedad"),
                doc.get("accion"),
            ),
        )
      self.actualizar_grafica()
    except Exception as e:
      messagebox.showerror("Error", f"Error al cargar datos: {e}")

  def actualizar_grafica(self):
    """Filtra y dibuja los puntos en el plano cartesiano X=Temp, Y=Humedad de forma fluida."""
    self.ax.clear()

    x_aire, y_aire = [], []
    x_vent, y_vent = [], []
    x_cal, y_cal = [], []
    x_apag, y_apag = [], []

    try:
      # Consultamos los datos directo de la BD para asegurar sincronización
      for doc in self.coleccion.find():
        t = doc.get("temperatura")
        h = doc.get("humedad")
        acc = doc.get("accion", "")

        if t is not None and h is not None:
          # Usamos conversiones seguras por si vienen como string o float
          t = float(t)
          h = float(h)

          if "aire acondicionado" in acc.lower():
            x_aire.append(t)
            y_aire.append(h)
          elif "ventilador" in acc.lower():
            x_vent.append(t)
            y_vent.append(h)
          elif "calefacción" in acc.lower():
            x_cal.append(t)
            y_cal.append(h)
          else:
            x_apag.append(t)
            y_apag.append(h)
    except Exception as e:
      print(f"Error procesando datos para gráfica: {e}")

    # Dibujar las series según los checkboxes
    if self.ver_aire.get() and x_aire:
      self.ax.scatter(
          x_aire,
          y_aire,
          color="crimson",
          label="Aire Acondicionado",
          s=60,
          alpha=0.8,
      )
    if self.ver_ventilador.get() and x_vent:
      self.ax.scatter(
          x_vent,
          y_vent,
          color="orange",
          label="Ventilador",
          s=60,
          alpha=0.8,
      )
    if self.ver_calefaccion.get() and x_cal:
      self.ax.scatter(
          x_cal,
          y_cal,
          color="dodgerblue",
          label="Calefacción",
          s=60,
          alpha=0.8,
      )
    if self.ver_apagado.get() and x_apag:
      self.ax.scatter(
          x_apag,
          y_apag,
          color="gray",
          label="Sistema Apagado",
          s=60,
          alpha=0.8,
      )

    self.ax.set_title("Relación Temperatura vs Humedad")
    self.ax.set_xlabel("Temperatura (°C) [X]")
    self.ax.set_ylabel("Humedad (%) [Y]")
    self.ax.grid(True, linestyle="--", alpha=0.5)

    # Evitar errores en la leyenda si no hay elementos visibles
    handles, labels = self.ax.get_legend_handles_labels()
    if handles:
      self.ax.legend(loc="upper left", fontsize=8)

    self.fig.tight_layout()

    # Usar draw_idle previene que la interfaz se congele al procesar múltiples eventos
    self.canvas.draw_idle()

  def limpiar_form(self):
    self.entry_temp.delete(0, tk.END)
    self.entry_hum.delete(0, tk.END)
    self.id_seleccionado = None
    if self.tree.selection():
      self.tree.selection_remove(self.tree.selection())

  def seleccionar_registro(self, event):
    seleccion = self.tree.selection()
    if seleccion:
      item = self.tree.item(seleccion)
      valores = item["values"]
      self.id_seleccionado = valores[0]
      self.entry_temp.delete(0, tk.END)
      self.entry_temp.insert(0, valores[1])
      self.entry_hum.delete(0, tk.END)
      self.entry_hum.insert(0, valores[2])

  def crear_registro(self):
    if not self.percibir():
      return
    self.tomar_decision()
    try:
      self.coleccion.insert_one({
          "temperatura": self.temperatura,
          "humedad": self.humedad,
          "accion": self.accion,
      })
      self.cargar_datos()
      self.limpiar_form()
    except Exception as e:
      messagebox.showerror("Error", f"No se pudo guardar: {e}")

  def actualizar_registro(self):
    if not self.id_seleccionado:
      messagebox.showwarning("Atención", "Selecciona un registro de la tabla")
      return
    if not self.percibir():
      return
    self.tomar_decision()
    try:
      self.coleccion.update_one(
          {"_id": ObjectId(self.id_seleccionado)},
          {
              "$set": {
                  "temperatura": self.temperatura,
                  "humedad": self.humedad,
                  "accion": self.accion,
              }
          },
      )
      self.cargar_datos()
      self.limpiar_form()
    except Exception as e:
      messagebox.showerror("Error", f"No se pudo actualizar: {e}")

  def eliminar_registro(self):
    if not self.id_seleccionado:
      messagebox.showwarning("Atención", "Selecciona un registro de la tabla")
      return
    if messagebox.askyesno(
        "Confirmar", "¿Estás seguro de eliminar este registro?"
    ):
      try:
        self.coleccion.delete_one({"_id": ObjectId(self.id_seleccionado)})
        self.cargar_datos()
        self.limpiar_form()
      except Exception as e:
        messagebox.showerror("Error", f"No se pudo eliminar: {e}")


if __name__ == "__main__":
  root = tk.Tk()
  app = AgenteClimatizacionApp(root)
  root.mainloop()