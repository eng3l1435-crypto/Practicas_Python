from bson.objectid import ObjectId
import tkinter as tk
from tkinter import messagebox, ttk
from pymongo import MongoClient

# --- CONFIGURACIÓN (Basada en tu código) ---
MONGO_URI = "mongodb://angel:mongo123@localhost:27017/" 
DB_NAME = "escuela"
COLLECTION_NAME = "alumnos"


class SimpleEscuelaCRUD:

  def __init__(self, root):
    self.root = root
    self.root.title("CRUD")
    self.root.geometry("680x420")

    # Conexión directa a MongoDB
    try:
      self.client = MongoClient(MONGO_URI)
      self.db = self.client[DB_NAME]
      self.collection = self.db[COLLECTION_NAME]
    except Exception as e:
      messagebox.showerror("Error de Conexión", f"No se pudo conectar: {e}")
      self.root.destroy()
      return

    # --- INTERFAZ GRÁFICA ---
    # 1. Formulario de campos (Nombre, Edad, Carrera)
    form_frame = ttk.LabelFrame(root, text=" Datos del Alumno ")
    form_frame.pack(fill="x", padx=10, pady=10)

    ttk.Label(form_frame, text="Nombre:").grid(
        row=0, column=0, padx=5, pady=8, sticky="w"
    )
    self.entry_nombre = ttk.Entry(form_frame, width=22)
    self.entry_nombre.grid(row=0, column=1, padx=5, pady=8)

    ttk.Label(form_frame, text="Edad:").grid(
        row=0, column=2, padx=5, pady=8, sticky="w"
    )
    self.entry_edad = ttk.Entry(form_frame, width=10)
    self.entry_edad.grid(row=0, column=3, padx=5, pady=8)

    ttk.Label(form_frame, text="Carrera:").grid(
        row=0, column=4, padx=5, pady=8, sticky="w"
    )
    self.entry_carrera = ttk.Entry(form_frame, width=22)
    self.entry_carrera.grid(row=0, column=5, padx=5, pady=8)

    # 2. Botones de acción CRUD
    btn_frame = ttk.Frame(root)
    btn_frame.pack(fill="x", padx=10, pady=5)

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

    # 3. Tabla para mostrar datos (Treeview)
    table_frame = ttk.Frame(root)
    table_frame.pack(fill="both", expand=True, padx=10, pady=10)

    self.tree = ttk.Treeview(
        table_frame, columns=("ID", "Nombre", "Edad", "Carrera"), show="headings"
    )
    self.tree.heading("ID", text="ID MongoDB")
    self.tree.heading("Nombre", text="Nombre")
    self.tree.heading("Edad", text="Edad")
    self.tree.heading("Carrera", text="Carrera")

    self.tree.column("ID", width=220, anchor="center")
    self.tree.column("Nombre", width=160, anchor="w")
    self.tree.column("Edad", width=60, anchor="center")
    self.tree.column("Carrera", width=180, anchor="w")

    self.tree.pack(side="left", fill="both", expand=True)
    self.tree.bind("<<TreeviewSelect>>", self.seleccionar_registro)

    # Scrollbar para la tabla
    scrollbar = ttk.Scrollbar(
        table_frame, orient="vertical", command=self.tree.yview
    )
    scrollbar.pack(side="right", fill="y")
    self.tree.configure(yscrollcommand=scrollbar.set)

    self.id_seleccionado = None
    self.cargar_datos()

  def cargar_datos(self):
    """Lee y muestra todos los documentos de la colección 'alumnos'"""
    for item in self.tree.get_children():
      self.tree.delete(item)
    try:
      for doc in self.collection.find():
        self.tree.insert(
            "",
            "end",
            values=(
                str(doc.get("_id")),
                doc.get("nombre", ""),
                doc.get("edad", ""),
                doc.get("carrera", ""),
            ),
        )
    except Exception as e:
      messagebox.showerror("Error", f"Error al cargar datos: {e}")

  def limpiar_form(self):
    """Limpia las cajas de texto y la selección"""
    self.entry_nombre.delete(0, tk.END)
    self.entry_edad.delete(0, tk.END)
    self.entry_carrera.delete(0, tk.END)
    self.id_seleccionado = None
    if self.tree.selection():
      self.tree.selection_remove(self.tree.selection())

  def seleccionar_registro(self, event):
    """Pasa los datos de la fila seleccionada hacia el formulario superior"""
    seleccion = self.tree.selection()
    if seleccion:
      item = self.tree.item(seleccion)
      valores = item["values"]
      self.id_seleccionado = valores[0]

      self.entry_nombre.delete(0, tk.END)
      self.entry_nombre.insert(0, valores[1])

      self.entry_edad.delete(0, tk.END)
      self.entry_edad.insert(0, valores[2])

      self.entry_carrera.delete(0, tk.END)
      self.entry_carrera.insert(0, valores[3])

  def crear_registro(self):
    """Inserta un nuevo alumno en la base de datos"""
    nombre = self.entry_nombre.get().strip()
    edad_str = self.entry_edad.get().strip()
    carrera = self.entry_carrera.get().strip()

    if not nombre or not edad_str or not carrera:
      messagebox.showwarning("Atención", "Todos los campos son obligatorios")
      return

    try:
      edad = int(edad_str)  # Convertir edad a número entero
    except ValueError:
      messagebox.showerror("Error", "La edad debe ser un número válido")
      return

    try:
      self.collection.insert_one(
          {"nombre": nombre, "edad": edad, "carrera": carrera}
      )
      self.cargar_datos()
      self.limpiar_form()
      messagebox.showinfo("Éxito", "Alumno registrado correctamente")
    except Exception as e:
      messagebox.showerror("Error", f"No se pudo insertar: {e}")

  def actualizar_registro(self):
    """Actualiza el alumno seleccionado por su _id"""
    if not self.id_seleccionado:
      messagebox.showwarning("Atención", "Selecciona un alumno de la tabla")
      return

    nombre = self.entry_nombre.get().strip()
    edad_str = self.entry_edad.get().strip()
    carrera = self.entry_carrera.get().strip()

    try:
      edad = int(edad_str)
    except ValueError:
      messagebox.showerror("Error", "La edad debe ser un número válido")
      return

    try:
      self.collection.update_one(
          {"_id": ObjectId(self.id_seleccionado)},
          {"$set": {"nombre": nombre, "edad": edad, "carrera": carrera}},
      )
      self.cargar_datos()
      self.limpiar_form()
      messagebox.showinfo("Éxito", "Alumno actualizado correctamente")
    except Exception as e:
      messagebox.showerror("Error", f"No se pudo actualizar: {e}")

  def eliminar_registro(self):
    """Elimina el alumno seleccionado"""
    if not self.id_seleccionado:
      messagebox.showwarning("Atención", "Selecciona un alumno de la tabla")
      return

    if messagebox.askyesno(
        "Confirmar", "¿Estás seguro de eliminar este alumno?"
    ):
      try:
        self.collection.delete_one({"_id": ObjectId(self.id_seleccionado)})
        self.cargar_datos()
        self.limpiar_form()
        messagebox.showinfo("Éxito", "Alumno eliminado correctamente")
      except Exception as e:
        messagebox.showerror("Error", f"No se pudo eliminar: {e}")


if __name__ == "__main__":
  root = tk.Tk()
  app = SimpleEscuelaCRUD(root)
  root.mainloop()