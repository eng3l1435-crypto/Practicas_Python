import matplotlib.pyplot as plt
import networkx as nx

arbol = {
    'A': ['B', 'C'],
    'B': ['A', 'D'], 
    'C': ['A', 'E'],
    'D': ['B'],
    'E': ['C', 'F'],  
    'F': ['E']
}

def caminos(nodo, camino_actual):
    camino = camino_actual + [nodo]
    hijos_no_visitados = [h for h in arbol[nodo] if h not in camino]
    
    if not hijos_no_visitados:
        print(" ---> ".join(camino))
        return

    for hijo in hijos_no_visitados:
        caminos(hijo, camino)

caminos('A', [])

G = nx.Graph(arbol)

niveles = nx.single_source_shortest_path_length(G, 'A')
nx.set_node_attributes(G, niveles, 'nivel')

pos_horizontal = nx.multipartite_layout(G, subset_key='nivel')

pos_arbol = {nodo: (y, -x) for nodo, (x, y) in pos_horizontal.items()}

plt.figure("Árbol de Búsqueda", figsize=(6, 5))
nx.draw(G, pos=pos_arbol, with_labels=True, node_color='skyblue', node_size=2000, font_size=12, font_weight='bold', edge_color='gray')

plt.show()