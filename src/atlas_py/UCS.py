
import heapq

grafo = {
    "A": [("B", 2), ("C", 5)],
    "B": [("D", 4), ("E", 1)],
    "C": [("F", 2), ("G", 8)],
    "D": [("H", 3), ("K",9)],
    "E": [("I", 2)],
    "F": [("J", 3)],
    "G": [],
    "H": [],
    "I": [],
    "J": [],
    "K": []
}

def ucs(inicio, objetivo):
    cola_prioridad = [(0, inicio, [inicio])]
    visitados = set()
    
    stats = {
        "movimientos": 0,     
        "max_profundidad": 0
    }
    
    while cola_prioridad:
        # Extraer el nodo con el menor costo acumulado
        costo_actual, actual, camino = heapq.heappop(cola_prioridad)
        
        profundidad = len(camino) - 1
        if profundidad > stats["max_profundidad"]:
            stats["max_profundidad"] = profundidad
            
        print(f"Visitando: {actual} | Costo Acumulado: {costo_actual} | Profundidad: {profundidad}")
        
        if actual == objetivo:
            return camino, costo_actual, stats
            
        if actual in visitados:
            continue
            
        visitados.add(actual)
        
        for vecino, peso in grafo[actual]:
            if vecino not in visitados:
                stats["movimientos"] += 1
                nuevo_costo = costo_actual + peso
                nuevo_camino = camino + [vecino]
                
                # Insertar en la cola de prioridad
                heapq.heappush(cola_prioridad, (nuevo_costo, vecino, nuevo_camino))
                
    return None, float('inf'), stats

ruta, costo_total, estadisticas = ucs("A", "K")

print("\nRUTA Y COSTO")
if ruta:
    print("Ruta óptima encontrada:", " -> ".join(ruta))
    print(f"Costo total del camino: {costo_total}")
else:
    print("No se encontró una ruta.")

print(f"Movimientos: {estadisticas['movimientos']}")
print(f"Profundidad máxima: {estadisticas['max_profundidad']}")