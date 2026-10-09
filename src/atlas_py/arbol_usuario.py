import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
import networkx as nx

def generar_arbol():
    # 1. Leer los datos de la única pantalla
    raiz = entry_raiz.get().strip()
    izq_str = entry_izq.get().strip()
    der_str = entry_der.get().strip()
    
    if not raiz:
        messagebox.showerror("Error", "El nodo de inicio (raíz) es obligatorio.")
        return
        
    arbol = {raiz: []}
    
    # 2. Construir la rama izquierda conectando en cadena (ej. A -> B -> D)
    if izq_str:
        nodos_izq = [n.strip() for n in izq_str.split(',') if n.strip()]
        if nodos_izq:
            arbol[raiz].append(nodos_izq[0]) # Conectar raíz con el primer izquierdo
            for i in range(len(nodos_izq)):
                # Conectar cada nodo con el siguiente, o dejarlo vacío si es el último (hoja)
                hijos = [nodos_izq[i+1]] if i + 1 < len(nodos_izq) else []
                arbol[nodos_izq[i]] = hijos

    # 3. Construir la rama derecha conectando en cadena (ej. A -> C -> E -> F)
    if der_str:
        nodos_der = [n.strip() for n in der_str.split(',') if n.strip()]
        if nodos_der:
            arbol[raiz].append(nodos_der[0]) # Conectar raíz con el primer derecho
            for i in range(len(nodos_der)):
                hijos = [nodos_der[i+1]] if i + 1 < len(nodos_der) else []
                arbol[nodos_der[i]] = hijos

    # Cerrar la interfaz gráfica porque ya tenemos los datos
    root.quit()
    root.destroy()
    
    # Ejecutar la lógica y el dibujo
    procesar_y_dibujar(arbol, raiz)


def procesar_y_dibujar(arbol, raiz):
    # --- TERMINAL: Buscar y mostrar caminos ---
    print(f"\nCaminos desde '{raiz}':")
    
    def caminos(nodo, camino_actual):
        camino = camino_actual + [nodo]
        hijos = arbol.get(nodo, [])
        
        if not hijos:  # Si no tiene hijos, llegamos al final del camino
            print(" --> ".join(camino))
            return
            
        for hijo in hijos:
            caminos(hijo, camino)

    caminos(raiz, [])

    # --- GRÁFICO: Dibujar el árbol cayendo ---
    # Usamos DiGraph (Grafo Dirigido) para que pinte flechas de arriba a abajo
    G = nx.DiGraph(arbol)
    
    # Calcular niveles para que "caiga"
    niveles = nx.single_source_shortest_path_length(G, raiz)
    nx.set_node_attributes(G, niveles, 'nivel')
    
    pos_horizontal = nx.multipartite_layout(G, subset_key='nivel')
    pos_arbol = {nodo: (y, -x) for nodo, (x, y) in pos_horizontal.items()}

    plt.figure("Árbol Generado", figsize=(6, 5))
    nx.draw(G, pos=pos_arbol, with_labels=True, node_color='lightgreen', 
            node_size=2000, font_size=12, font_weight='bold', 
            edge_color='gray', arrows=True, arrowsize=20)
    plt.show()


# --- INTERFAZ GRÁFICA (Única pantalla) ---
root = tk.Tk()
root.title("Árbol")
root.geometry("400x220")
root.eval('tk::PlaceWindow . center') # Centrar ventana

# Título e input de la Raíz
tk.Label(root, text="Nodo de Inicio (Raíz):", font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=2, pady=(15, 5))
entry_raiz = tk.Entry(root, justify="center", font=("Arial", 10, "bold"))
entry_raiz.grid(row=1, column=0, columnspan=2, pady=5)
entry_raiz.insert(0, "A") # Texto por defecto para que sea más rápido

# Etiquetas para ramas
tk.Label(root, text="Camino Izquierdo\n").grid(row=2, column=0, pady=(10,0))
tk.Label(root, text="Camino Derecho\n").grid(row=2, column=1, pady=(10,0))

# Inputs para ramas
entry_izq = tk.Entry(root, width=20, justify="center")
entry_izq.grid(row=3, column=0, padx=15, pady=5)
entry_izq.insert(0, "B, D")

entry_der = tk.Entry(root, width=20, justify="center")
entry_der.grid(row=3, column=1, padx=15, pady=5)
entry_der.insert(0, "C, E, F")

# Botón de acción único
tk.Button(root, text="Generar Árbol", command=generar_arbol, bg="lightblue", font=("Arial", 10, "bold")).grid(row=4, column=0, columnspan=2, pady=20)

root.mainloop()