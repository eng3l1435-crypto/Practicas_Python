# ============================================================
# BÚSQUEDA INFORMADA - A* (A ESTRELLA)
# ============================================================

import heapq

grafo = {
    "A": [("B", 6), ("C", 4)],
    "B": [("D", 7), ("E", 5)],
    "C": [("F", 3), ("G", 6)],
    "D": [("H", 8)],
    "E": [],
    "F": [("I", 1)],
    "G": [],
    "H": [],
    "I": [("J", 0)],
    "J": [],
    "K": []
}

# Diccionario de heurísticas (estimación al objetivo)
heuristicas = {
    "A": 7,
    "B": 5,
    "C": 4,
    "D": 3,
    "E": 2,
    "F": 2,
    "G": 6,
    "H": 1,
    "I": 1,
    "J": 0,
    "K": 5
}

def a_star(inicio, objetivo):

    f_inicial = 0 + heuristicas[inicio]
    cola_prioridad = [(f_inicial, 0, inicio, [inicio])]
    visitados = set()
    
    stats = {
        "movimientos": 0, 
        "max_profundidad": 0
    }
    
    while cola_prioridad:
        f_actual, costo_real, actual, camino = heapq.heappop(cola_prioridad)
        
        profundidad = len(camino) - 1
        if profundidad > stats["max_profundidad"]:
            stats["max_profundidad"] = profundidad
            
        print(f"Visitando: {actual} | Costo Real (g): {costo_real} | f(n): {f_actual} | Profundidad: {profundidad}")
        
        if actual == objetivo:
            return camino, costo_real, stats
            
        if actual in visitados:
            continue
        visitados.add(actual)
        
        for vecino, peso in grafo[actual]:
            if vecino not in visitados:
                stats["movimientos"] += 1
                # g(n): Costo acumulado real sumando el peso de la arista
                costo_real_nuevo = costo_real + peso
                # f(n) = g(n) + h(n)
                f_nuevo = costo_real_nuevo + heuristicas[vecino]
                nuevo_camino = camino + [vecino]
                
                heapq.heappush(cola_prioridad, (f_nuevo, costo_real_nuevo, vecino, nuevo_camino))
                
    return None, float('inf'), stats

print("\n")
ruta, costo_total, estadisticas = a_star("A", "J")

print("\n=")
if ruta:
    print("Ruta óptima encontrada:", " -> ".join(ruta))
    print(f"Estimación: {costo_total}")
else:
    print("No se encontró una ruta.")

print(f"Movimientos: {estadisticas['movimientos']}")
print(f"Profundidad máxima: {estadisticas['max_profundidad']}")