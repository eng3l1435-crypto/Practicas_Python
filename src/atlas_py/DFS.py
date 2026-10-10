# ============================================================
# BÚSQUEDA CIEGA - DEPTH FIRST SEARCH (DFS)
# ============================================================
#
# DFS = Depth First Search
#      Búsqueda en Profundidad
#
# IDEA PRINCIPAL:
# DFS intenta avanzar lo más profundo posible por un camino
# antes de regresar y probar otro camino.
#
# Es una búsqueda CIEGA porque NO utiliza información adicional
# que indique qué tan cerca estamos del objetivo.
# ============================================================

# DFS conceptualmente utiliza una estructura tipo pila:

#          ┌─────┐
#entra →   │  A  │
#          │  B  │
#          │  D  │ ← sale primero
#          └─────┘
#            LIFO

# ------------------------------------------------------------
# 1. DEFINIMOS EL ESPACIO DE BÚSQUEDA
# ------------------------------------------------------------
#
# El diccionario representa un grafo.
#
# Cada clave representa un estado.
# La lista asociada contiene los estados vecinos
# a los que podemos desplazarnos.
#
# Visualmente podemos imaginar:
#
#              A
#             / \
#            B   C
#           / \   \
#          D   E   F
#
# DFS comenzará en A y buscará llegar a F.
# ------------------------------------------------------------

grafo = {

    "A": ["B", "C"],

    "B": ["D", "E"],

    "C": ["F"],

    "D": [],

    "E": ["G", "H"],

    "F": [],

    "G": [],

    "H": []
}


# ------------------------------------------------------------
# 2. FUNCIÓN DFS
# ------------------------------------------------------------

def dfs(actual, objetivo, visitados=None, camino=None, profundidad=0, stats=None):

    # --------------------------------------------------------
    # Crear las estructuras en la primera llamada
    # --------------------------------------------------------
    #
    # visitados:
    # Guarda los estados que ya hemos explorado.
    #
    # camino:
    # Guarda la ruta que estamos siguiendo actualmente.
    # --------------------------------------------------------

    if visitados is None:
        visitados = set()

    if camino is None:
        camino = []

    if stats is None:
        stats = {
            "movimientos": 0,
            "retrocesos": 0,
            "max_profundidad": 0
        }

# Registrar la profundidad máxima alcanzada
    if profundidad > stats["max_profundidad"]:
        stats["max_profundidad"] = profundidad

    # --------------------------------------------------------
    # 3. MARCAR EL ESTADO ACTUAL COMO VISITADO
    # --------------------------------------------------------

    visitados.add(actual)

    # Agregamos el estado al camino actual.
    camino.append(actual)


    # Mostramos qué estado estamos explorando.
    print(f"Visitando: {actual}, Profundidad: {profundidad} ")


    # --------------------------------------------------------
    # 4. COMPROBAR SI ENCONTRAMOS EL OBJETIVO
    # --------------------------------------------------------

    if actual == objetivo:

        # copy() es importante porque camino seguirá
        # modificándose durante el proceso.
        return camino.copy(), stats


    # --------------------------------------------------------
    # 5. EXPLORAR LOS VECINOS
    # --------------------------------------------------------
    #
    # DFS toma un vecino y profundiza en él.
    # --------------------------------------------------------

    for vecino in grafo[actual]:

        # Solo exploramos estados que todavía
        # no hayan sido visitados.

        if vecino not in visitados:
            stats["movimientos"] += 1 

            # Llamamos nuevamente a DFS.
            #
            # Esto es una llamada RECURSIVA.
            resultado, stats = dfs(
                vecino,
                objetivo,
                visitados,
                camino,
                profundidad +1,
                stats
            )


            # ------------------------------------------------
            # 6. ¿SE ENCONTRÓ EL OBJETIVO?
            # ------------------------------------------------

            if resultado:

                return resultado, stats


    # --------------------------------------------------------
    # 7. RETROCESO (BACKTRACKING)
    # --------------------------------------------------------
    #
    # Si llegamos aquí significa que desde "actual"
    # no encontramos el objetivo.
    #
    # Entonces eliminamos este estado del camino
    # y regresamos al estado anterior.
    # --------------------------------------------------------

    stats["retrocesos"] += 1
    print(f"<- Retrocediendo desde: {actual}")
    camino.pop()

    return None, stats


# ============================================================
# 8. EJECUTAR LA BÚSQUEDA
# ============================================================
ruta, estadisticas = dfs("A", "H")
print(f"Ruta: {ruta}")


# ============================================================
# 9. MOSTRAR EL RESULTADO
# ============================================================

print("\nRuta encontrada:")

if ruta:

    print(" -> ".join(ruta))

else:

    print("No se encontró una ruta.")

print(f"avances: {estadisticas['movimientos']}")
print(f"retrocesos: {estadisticas['retrocesos']}")
print(f"Profundidad max: {estadisticas['max_profundidad']}")
#¿Qué significa "ciega"?
#No significa que el algoritmo no tenga información del problema.
#Significa que no utiliza una heurística que le diga qué tan prometedor es un estado.
#DFS solamente sabe:
#"Estoy en A y desde A puedo ir a B o C."
#No sabe:
#"C está más cerca de F que B."


#DFS significa "profundizar"
#La regla mental es: "Primero bajo todo lo posible; si no funciona, regreso."

#¿Encontré el objetivo?
 
#Sí regreso la ruta
#No continúo buscando