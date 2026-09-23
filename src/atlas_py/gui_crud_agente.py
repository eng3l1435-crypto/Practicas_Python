from bson.objectid import ObjectId
import os
from dotenv import load_dotenv
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

class AgenteClimatizacionCRUD:

  def __init__(self, root):
    self.root = root
    self.root.title("Agente Climatización")
    self.root.geometry("720x480")
    self.root.resizable(False, False)

    self.temperatura = 0.0
    self.humedad = 0.0
    self.accion = ""
    self.id_seleccionado = None

    # --- CONEXIÓN MONGODB ---
    try:
      self.cliente = MongoClient(uri)
      self.db = self.cliente[nombre_db]
      self.coleccion = self.db[nombre_coleccion]
    except Exception as e:
      messagebox.showerror("Error de Conexión", f"No se pudo conectar: {e}")
      self.root.destroy()
      return

    # --- INTERFAZ GRÁFICA ---
    form_frame = ttk.LabelFrame(
        root, text=" Percepción del Entorno & Decisiones "
    )
    form_frame.pack(fill="x", padx=15, pady=10)

    ttk.Label(form_frame, text="Temperatura (°C):").grid(
        row=0, column=0, sticky="w", padx=10, pady=8
    )
    self.entry_temp = ttk.Entry(form_frame, width=15)
    self.entry_temp.grid(row=0, column=1, padx=10, pady=8)

    ttk.Label(form_frame, text="Humedad (%):").grid(
        row=0, column=2, sticky="w", padx=10, pady=8
    )
    self.entry_hum = ttk.Entry(form_frame, width=15)
    self.entry_hum.grid(row=0, column=3, padx=10, pady=8)

    # Botones de Acción CRUD
    btn_frame = ttk.Frame(root)
    btn_frame.pack(fill="x", padx=15, pady=5)

    ttk.Button(
        btn_frame, text="Crear", command=self.crear_registro
    ).pack(side="left", padx=5)
    ttk.Button(
        btn_frame, text="Actualizar", command=self.actualizar_registro
    ).pack(side="left", padx=5)
    ttk.Button(
        btn_frame, text="Eliminar", command=self.eliminar_registro
    ).pack(side="left", padx=5)
    ttk.Button(btn_frame, text="Limpiar", command=self.limpiar_form).pack(
        side="left", padx=5
    )

    # Tabla para mostrar registros (Treeview)
    table_frame = ttk.Frame(root)
    table_frame.pack(fill="both", expand=True, padx=15, pady=10)

    self.tree = ttk.Treeview(
        table_frame,
        columns=("ID", "Temperatura", "Humedad", "Acción"),
        show="headings",
    )
    self.tree.heading("ID", text="ID MongoDB")
    self.tree.heading("Temperatura", text="Temp (°C)")
    self.tree.heading("Humedad", text="Humedad (%)")
    self.tree.heading("Acción", text="Agente")

    self.tree.column("ID", width=200, anchor="center")
    self.tree.column("Temperatura", width=80, anchor="center")
    self.tree.column("Humedad", width=80, anchor="center")
    self.tree.column("Acción", width=300, anchor="w")

    self.tree.pack(side="left", fill="both", expand=True)
    self.tree.bind("<<TreeviewSelect>>", self.seleccionar_registro)

    scrollbar = ttk.Scrollbar(
        table_frame, orient="vertical", command=self.tree.yview
    )
    scrollbar.pack(side="right", fill="y")
    self.tree.configure(yscrollcommand=scrollbar.set)

    self.cargar_datos()

  def percibir(self):
    """Captura los datos desde la interfaz y valida que sean numéricos."""
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
    """Aplica la regla condición-acción basada en la percepción."""
    if self.temperatura > 30 and self.humedad > 70:
      self.accion = "Encender aire acondicionado (Modo Deshumidificador)"
    elif self.temperatura > 30:
      self.accion = "Encender ventilador"
    elif self.temperatura < 18:
      self.accion = "Encender calefacción"
    else:
      self.accion = "Mantener sistema apagado"

  def cargar_datos(self):
    """Lee y muestra todos los documentos almacenados en MongoDB."""
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
    except Exception as e:
      messagebox.showerror("Error", f"Error al cargar datos: {e}")

  def limpiar_form(self):
    """Limpia los campos y deselecciona la tabla."""
    self.entry_temp.delete(0, tk.END)
    self.entry_hum.delete(0, tk.END)
    self.id_seleccionado = None
    if self.tree.selection():
      self.tree.selection_remove(self.tree.selection())

  def seleccionar_registro(self, event):
    """Carga los datos del elemento seleccionado al formulario."""
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
    """Ejecuta el agente y guarda el resultado en MongoDB."""
    if not self.percibir():
      return
    self.tomar_decision()

    documento = {
        "temperatura": self.temperatura,
        "humedad": self.humedad,
        "accion": self.accion,
    }
    try:
      self.coleccion.insert_one(documento)
      self.cargar_datos()
      self.limpiar_form()
      messagebox.showinfo(
          "Éxito", f"Decisión tomada y registrada:\n{self.accion}"
      )
    except Exception as e:
      messagebox.showerror("Error", f"No se pudo guardar: {e}")

  def actualizar_registro(self):
    """Actualiza el registro seleccionado en la base de datos."""
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
      messagebox.showinfo("Éxito", "Registro actualizado correctamente")
    except Exception as e:
      messagebox.showerror("Error", f"No se pudo actualizar: {e}")

  def eliminar_registro(self):
    """Elimina el registro seleccionado de MongoDB."""
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
        messagebox.showinfo("Éxito", "Registro eliminado correctamente")
      except Exception as e:
        messagebox.showerror("Error", f"No se pudo eliminar: {e}")


# --- EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
  root = tk.Tk()
  app = AgenteClimatizacionCRUD(root)
  root.mainloop()