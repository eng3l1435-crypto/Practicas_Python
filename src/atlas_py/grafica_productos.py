import math
import tkinter as tk


def resolver_mochila():
  productos = ['a', 'b', 'c', 'd', 'e']
  costos = {'a': 200, 'b': 300, 'c': 400, 'd': 250, 'e': 350}
  beneficios = {'a': 30, 'b': 50, 'c': 70, 'd': 40, 'e': 60}
  presupuesto_limite = 1000

  # Encontrar la mejor combinación por fuerza bruta (todas las combinaciones posibles)
  mejor_beneficio = -1
  mejor_combinacion = []
  mejor_costo = 0

  n = len(productos)
  for i in range(1 << n):
    comb = []
    costo_actual = 0
    beneficio_actual = 0
    for j in range(n):
      if (i >> j) & 1:
        prod = productos[j]
        comb.append(prod)
        costo_actual += costos[prod]
        beneficio_actual += beneficios[prod]

    if costo_actual <= presupuesto_limite:
      if beneficio_actual > mejor_beneficio:
        mejor_beneficio = beneficio_actual
        mejor_combinacion = comb
        mejor_costo = costo_actual

  return (
      productos,
      costos,
      beneficios,
      mejor_combinacion,
      mejor_costo,
      mejor_beneficio,
  )


def dibujar_grafica(productos, costos, beneficios, seleccionados):
  canvas.delete('all')

  width = canvas.winfo_width()
  height = canvas.winfo_height()
  if width < 10:
    width = 700
  if height < 10:
    height = 500

  margin_x = 70
  margin_y = 60
  chart_width = width - 2 * margin_x
  chart_height = height - 2 * margin_y

  # Rangos máximos para los ejes (Costo hasta 500, Beneficio hasta 90)
  max_costo = 500
  max_beneficio = 90

  # Funciones para mapear coordenadas matemáticas a píxeles del Canvas
  def map_x(c):
    return margin_x + (c / max_costo) * chart_width

  def map_y(b):
    return height - margin_y - (b / max_beneficio) * chart_height

  # Dibujar Ejes X e Y
  canvas.create_line(
      margin_x,
      height - margin_y,
      width - margin_x,
      height - margin_y,
      width=2,
      fill='black',
  )
  canvas.create_line(
      margin_x,
      margin_y,
      margin_x,
      height - margin_y,
      width=2,
      fill='black',
  )

  # Etiquetas de los Ejes
  canvas.create_text(
      width / 2,
      height - 20,
      text='Costo ($)',
      font=('Arial', 10, 'bold'),
  )
  canvas.create_text(
      25,
      height / 2,
      text='Beneficio',
      font=('Arial', 10, 'bold'),
      angle=90,
  )

  canvas.create_text(
      width / 2,
      25,
      text='Diagrama de Dispersión: Costo vs Beneficio por Producto',
      font=('Arial', 12, 'bold'),
  )

  # Dibujar líneas de referencia / cuadrícula sutil
  for c_val in range(100, max_costo + 1, 100):
    px = map_x(c_val)
    canvas.create_line(
        px, margin_y, px, height - margin_y, fill='#e0e0e0', dash=(2, 2)
    )
    canvas.create_text(
        px, height - margin_y + 15, text=str(c_val), font=('Arial', 9)
    )

  for b_val in range(20, max_beneficio + 1, 20):
    py = map_y(b_val)
    canvas.create_line(
        margin_x, py, width - margin_x, py, fill='#e0e0e0', dash=(2, 2)
    )
    canvas.create_text(
        margin_x - 25, py, text=str(b_val), font=('Arial', 9)
    )

  # Dibujar los puntos (productos)
  for prod in productos:
    c = costos[prod]
    b = beneficios[prod]

    px = map_x(c)
    py = map_y(b)

    es_seleccionado = prod in seleccionados

    # Estilo del puntito
    radio = 10 if es_seleccionado else 7
    color = '#2E7D32' if es_seleccionado else '#90A4AE'
    borde_color = 'black' if es_seleccionado else '#455A64'
    grosor_borde = 2 if es_seleccionado else 1

    # Dibujar círculo (puntito)
    canvas.create_oval(
        px - radio,
        py - radio,
        px + radio,
        py + radio,
        fill=color,
        outline=borde_color,
        width=grosor_borde,
    )

    # Etiqueta del producto al lado del punto
    texto_etiqueta = (
        f'Prod {prod.upper()} (C:{c}, B:{b})'
        if es_seleccionado
        else f'{prod} (C:{c}, B:{b})'
    )
    fuente_estilo = 'bold' if es_seleccionado else 'normal'
    color_texto = '#1B5E20' if es_seleccionado else '#546E7A'

    canvas.create_text(
        px,
        py - 18,
        text=texto_etiqueta,
        font=('Arial', 10, fuente_estilo),
        fill=color_texto,
    )


def actualizar_interfaz():
  productos, costos, beneficios, seleccionados, costo_total, beneficio_total = (
      resolver_mochila()
  )

  sel_str = ', '.join([p.upper() for p in seleccionados])
  lbl_resultado.config(
      text=(
          f'Productos óptimos: {sel_str}  |  Costo Total: ${costo_total} /'
          f' $1000  |  Beneficio Máximo: {beneficio_total}'
      )
  )

  dibujar_grafica(productos, costos, beneficios, seleccionados)


# --- CONFIGURACIÓN DE LA VENTANA TKINTER ---
root = tk.Tk()
root.title('Presupuesto')
root.geometry('750x550')

frame_info = tk.Frame(root, padx=15, pady=10)
frame_info.pack(fill='x')

lbl_titulo = tk.Label(
    frame_info,
    text='Presupuesto máximo: $1000 pesos',
    font=('Arial', 11, 'bold'),
    fg='#333',
)
lbl_titulo.pack(anchor='w', pady=2)

lbl_resultado = tk.Label(
    frame_info, text='', font=('Arial', 10, 'bold'), fg='#2E7D32'
)
lbl_resultado.pack(anchor='w', pady=2)

canvas = tk.Canvas(root, bg='white', highlightbackground='#ccc')
canvas.pack(fill='both', expand=True, padx=15, pady=10)
canvas.bind('<Configure>', lambda e: actualizar_interfaz())

actualizar_interfaz()

root.mainloop()
