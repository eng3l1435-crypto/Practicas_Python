#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 SOLUCIÓN EN PYTHON · EXAMEN UNIDAD II · FUNDAMENTOS DE IA (E-FIA-3)
 Caso: AutoPartes del Valle de Toluca - Ruteo inteligente de un AGV
=============================================================================
 Reproduce TODOS los resultados de la clave de respuestas del examen:

   Sección I   -> Formulación del problema y tamaño del espacio (n!)
   Sección II  -> BFS y búsqueda de costo uniforme (UCS)
   Sección III -> h*(n), admisibilidad, consistencia, Voraz, A*, h(C)=14
   Sección IV  -> Ascenso de colina (hill climbing)
   Sección V   -> Algoritmo genético (cálculo a mano + corrida completa)
   Verificación-> Enumeración exhaustiva de las 24 rutas

 Requisitos: Python 3.8+ (solo biblioteca estándar, no instala nada).
 Ejecución : python3 solucion_examen_unidad2.py

 Convenciones del examen (se respetan aquí):
   * Costos en minutos.
   * Se usa conjunto explorado (verificación de estados repetidos).
   * La prueba de meta se aplica AL EXPANDIR el nodo.
   * Desempates: nombre de estación alfabéticamente menor primero.
=============================================================================
"""

import heapq
import itertools
import math
import random
import statistics
from collections import deque

# =============================================================================
# 0. DATOS DEL CASO DE ESTUDIO
# =============================================================================

# Tramos bidireccionales (estación_1, estación_2, minutos)
TRAMOS = [
    ("A", "B", 4), ("A", "C", 2), ("A", "F", 3),
    ("B", "C", 1), ("B", "D", 5), ("C", "D", 8),
    ("C", "E", 10), ("D", "E", 2), ("D", "G", 6),
    ("E", "G", 3), ("F", "H", 2),
]

INICIO = "A"   # Almacén de materia prima
META = "G"     # Línea de ensamble

# Heurística h(n): tiempo estimado (línea recta / velocidad máxima) hacia G
H = {"A": 11, "B": 8, "C": 9, "D": 5, "E": 3, "F": 12, "G": 0, "H": 14}


def construir_grafo(tramos):
    """Convierte la lista de tramos en un diccionario de adyacencia.
    Como los tramos son bidireccionales, se agrega cada arco en ambos sentidos.
    Resultado: {estación: {vecino: costo, ...}, ...}
    """
    grafo = {}
    for u, v, costo in tramos:
        grafo.setdefault(u, {})[v] = costo
        grafo.setdefault(v, {})[u] = costo
    return grafo


GRAFO = construir_grafo(TRAMOS)


def titulo(texto):
    """Imprime un encabezado vistoso para separar secciones de la salida."""
    print("\n" + "=" * 74)
    print(texto)
    print("=" * 74)


def ruta_str(ruta):
    """Convierte ['A','C','B'] en 'A-C-B'."""
    return "-".join(ruta)


# =============================================================================
# SECCIÓN I. FORMULACIÓN DEL PROBLEMA DE BÚSQUEDA
# =============================================================================

class ProblemaRuteo:
    """Define formalmente el problema R1 (ruta de costo mínimo A -> G).

    Los seis componentes de un problema de búsqueda (Russell y Norvig, cap. 3.1):
    estados, estado inicial, acciones, modelo de transición, prueba de meta
    y costo de camino.
    """

    def __init__(self, grafo, inicio, meta):
        self.grafo = grafo
        self.estado_inicial = inicio
        self.meta = meta
        self.estados = sorted(grafo.keys())          # Espacio de estados

    def acciones(self, s):
        """ACCIONES(s): 'ir a' cada estación adyacente (orden alfabético)."""
        return sorted(self.grafo[s].keys())

    def resultado(self, s, accion):
        """RESULTADO(s, ir a s'): el AGV queda en la estación destino."""
        return accion

    def es_meta(self, s):
        """Prueba de meta: ¿el AGV llegó a la línea de ensamble?"""
        return s == self.meta

    def costo_paso(self, s, accion):
        """Costo de paso c(s, a, s'): minutos del tramo (siempre positivo)."""
        return self.grafo[s][accion]


def seccion_1():
    titulo("SECCIÓN I · FORMULACIÓN DEL PROBLEMA")
    p = ProblemaRuteo(GRAFO, INICIO, META)

    print("1.1 Componentes del problema:")
    print(f"   Estados          : {p.estados}  ({len(p.estados)} estados)")
    print(f"   Estado inicial   : {p.estado_inicial}")
    print(f"   ACCIONES(C)      : {p.acciones('C')}")
    print(f"   Prueba de meta   : s == {p.meta}")
    print("   Costo de camino  : suma de los minutos de cada tramo")

    print("\n1.2 Estructura del espacio de búsqueda:")
    n_nodos = len(GRAFO)
    n_aristas = len(TRAMOS)
    grados = {n: len(v) for n, v in sorted(GRAFO.items())}
    print(f"   Nodos={n_nodos}, aristas={n_aristas}, arcos dirigidos={2 * n_aristas}")
    print(f"   Grados: {grados}  -> grado máximo = {max(grados.values())}")
    print("   Es un GRAFO con ciclos (A-B-C-A, B-C-D): hace falta conjunto explorado.")

    print("\n1.3 Tamaño del espacio de R3 (n estaciones, salida y regreso en A):")
    print(f"   {'n':>3} | {'rutas = n!':>28} | tiempo a 1e9 rutas/s")
    print("   " + "-" * 62)
    for n in (4, 12, 20):
        rutas = math.factorial(n)
        seg = rutas / 1e9
        if seg < 1:
            tiempo = f"{seg:.2f} s"
        elif seg < 3.156e7:
            tiempo = f"{seg:.1f} s"
        else:
            tiempo = f"{seg / 3.156e7:.0f} años"
        print(f"   {n:>3} | {rutas:>28,} | {tiempo}")


# =============================================================================
# SECCIÓN II. BÚSQUEDA CIEGA (BFS y UCS)
# =============================================================================

def reconstruir_ruta(padres, meta):
    """Sigue los punteros 'padre' desde la meta hasta el inicio."""
    ruta = []
    nodo = meta
    while nodo is not None:
        ruta.append(nodo)
        nodo = padres[nodo]
    return list(reversed(ruta))


def costo_ruta(ruta):
    """Suma el costo de los tramos de una ruta [A, C, B, ...]."""
    return sum(GRAFO[a][b] for a, b in zip(ruta, ruta[1:]))


def bfs(problema):
    """Búsqueda en anchura (FIFO) con conjunto explorado.

    El padre de un nodo es quien lo DESCUBRE primero. Prueba de meta al
    expandir. Devuelve (ruta, costo, orden_de_expansion).
    """
    frontera = deque([problema.estado_inicial])
    padres = {problema.estado_inicial: None}
    descubiertos = {problema.estado_inicial}
    orden = []

    while frontera:
        s = frontera.popleft()
        orden.append(s)
        if problema.es_meta(s):
            ruta = reconstruir_ruta(padres, s)
            return ruta, costo_ruta(ruta), orden
        for a in problema.acciones(s):            # orden alfabético
            hijo = problema.resultado(s, a)
            if hijo not in descubiertos:
                descubiertos.add(hijo)
                padres[hijo] = s
                frontera.append(hijo)
    return None, math.inf, orden


def busqueda_mejor_primero(problema, tipo, h=None, verbose=False):
    """Algoritmo genérico de mejor-primero con conjunto explorado.

    El tipo decide la función de evaluación f(n):
        'ucs'    -> f = g          (costo uniforme)
        'voraz'  -> f = h          (greedy best-first)
        'astar'  -> f = g + h      (A*)

    Usa un montículo con "borrado perezoso": si un nodo mejora su g, se
    inserta de nuevo y las entradas viejas se ignoran al sacarlas.
    Desempate: (f, nombre) -> alfabético, como pide el examen.
    Devuelve (ruta, costo, orden_de_expansion).
    """
    h = h or {}

    def f_de(nodo, g):
        if tipo == "ucs":
            return g
        if tipo == "voraz":
            return h[nodo]
        return g + h[nodo]            # astar

    inicio = problema.estado_inicial
    mejor_g = {inicio: 0}
    padres = {inicio: None}
    explorados = set()
    heap = [(f_de(inicio, 0), inicio, 0)]
    orden = []
    iteracion = 0

    if verbose:
        print(f"   {'Iter':<5}{'Expande':<9}{'g':<5}{'f':<5}Frontera (nodo:f)")
        print("   " + "-" * 66)

    while heap:
        f, s, g = heapq.heappop(heap)
        if s in explorados or g > mejor_g[s]:
            continue                                   # entrada obsoleta
        explorados.add(s)
        orden.append(s)
        iteracion += 1

        if problema.es_meta(s):                        # meta al expandir
            if verbose:
                print(f"   {iteracion:<5}{s:<9}{g:<5}{f:<5}META")
            ruta = reconstruir_ruta(padres, s)
            return ruta, costo_ruta(ruta), orden

        for a in problema.acciones(s):
            hijo = problema.resultado(s, a)
            g_hijo = g + problema.costo_paso(s, a)
            if hijo in explorados:
                continue
            if hijo not in mejor_g or g_hijo < mejor_g[hijo]:
                mejor_g[hijo] = g_hijo
                padres[hijo] = s
                heapq.heappush(heap, (f_de(hijo, g_hijo), hijo, g_hijo))

        if verbose:
            # Frontera vigente: entradas no obsoletas y no exploradas
            vivos = {}
            for fe, ne, ge in heap:
                if ne not in explorados and ge == mejor_g[ne]:
                    vivos[ne] = fe
            txt = ", ".join(f"{n}:{fv}" for n, fv in
                            sorted(vivos.items(), key=lambda t: (t[1], t[0])))
            print(f"   {iteracion:<5}{s:<9}{g:<5}{f:<5}{txt}")

    return None, math.inf, orden


def seccion_2():
    titulo("SECCIÓN II · BÚSQUEDA CIEGA")
    p = ProblemaRuteo(GRAFO, INICIO, META)

    print("2.1 BFS")
    ruta, costo, orden = bfs(p)
    print(f"   Orden de expansión: {orden}  ({len(orden)} expansiones)")
    print(f"   Ruta: {ruta_str(ruta)}   Costo: {costo} min")

    print("\n2.2 UCS (tabla de iteraciones; f = g)")
    ruta, costo, orden = busqueda_mejor_primero(p, "ucs", verbose=True)
    print(f"   Ruta: {ruta_str(ruta)}   Costo: {costo} min   "
          f"Expansiones: {len(orden)}")

    print("\n2.3 BFS minimiza NÚMERO DE ARCOS, no costo: por eso 15 vs 13 min.")
    print("    BFS es óptimo solo si todos los costos de paso son iguales.")


# =============================================================================
# SECCIÓN III. BÚSQUEDA CON HEURÍSTICAS
# =============================================================================

def costos_optimos_a_meta(grafo, meta):
    """Calcula h*(n): costo óptimo REAL de cada nodo a la meta (Dijkstra).

    Como el grafo es no dirigido, basta correr Dijkstra desde la meta.
    """
    dist = {n: math.inf for n in grafo}
    dist[meta] = 0
    heap = [(0, meta)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, c in grafo[u].items():
            if d + c < dist[v]:
                dist[v] = d + c
                heapq.heappush(heap, (dist[v], v))
    return dist


def es_admisible(h, h_star):
    """h es admisible si NUNCA sobreestima: h(n) <= h*(n) para todo n."""
    return all(h[n] <= h_star[n] for n in h)


def es_consistente(h, grafo):
    """h es consistente si h(n) <= c(n,n') + h(n') en TODOS los arcos."""
    violaciones = []
    for u in grafo:
        for v, c in grafo[u].items():
            if h[u] > c + h[v]:
                violaciones.append((u, v, h[u], c + h[v]))
    return violaciones


def seccion_3():
    titulo("SECCIÓN III · BÚSQUEDA CON HEURÍSTICAS")
    p = ProblemaRuteo(GRAFO, INICIO, META)

    # ---- 3.1 h* y admisibilidad ----
    print("3.1 Costo óptimo real h*(n) y admisibilidad")
    h_star = costos_optimos_a_meta(GRAFO, META)
    print(f"   {'Estación':<10}" + "".join(f"{n:>5}" for n in sorted(H)))
    print(f"   {'h*(n)':<10}" + "".join(f"{h_star[n]:>5}" for n in sorted(H)))
    print(f"   {'h(n)':<10}" + "".join(f"{H[n]:>5}" for n in sorted(H)))
    print(f"   ¿h admisible? {es_admisible(H, h_star)}")

    # ---- 3.2 consistencia ----
    print("\n3.2 Consistencia: h(n) <= c(n,n') + h(n')")
    for u, v in (("A", "C"), ("B", "C"), ("D", "E")):
        c = GRAFO[u][v]
        ok = H[u] <= c + H[v]
        print(f"   {u}->{v}: {H[u]} <= {c} + {H[v]} = {c + H[v]}  "
              f"{'OK' if ok else 'FALLA'}")
    viol = es_consistente(H, GRAFO)
    print(f"   Revisión de los {2 * len(TRAMOS)} arcos -> "
          f"{'h CONSISTENTE' if not viol else f'violaciones: {viol}'}")

    # ---- 3.3 Voraz ----
    print("\n3.3 Búsqueda voraz (f = h)")
    ruta, costo, orden = busqueda_mejor_primero(p, "voraz", H, verbose=True)
    print(f"   Orden: {orden}   Ruta: {ruta_str(ruta)}   Costo: {costo} min")
    print("   No es óptima: ignora g y nunca revisa la opción A-C-B.")

    # ---- 3.4 A* y comparativa ----
    print("\n3.4 A* (f = g + h)")
    ruta_a, costo_a, orden_a = busqueda_mejor_primero(p, "astar", H, verbose=True)
    print(f"   Ruta: {ruta_str(ruta_a)}   Costo: {costo_a} min")
    nunca = sorted(set(GRAFO) - set(orden_a))
    print(f"   Nunca expandidos por A*: {nunca}")

    r_bfs, c_bfs, o_bfs = bfs(p)
    r_ucs, c_ucs, o_ucs = busqueda_mejor_primero(p, "ucs")
    r_vor, c_vor, o_vor = busqueda_mejor_primero(p, "voraz", H)

    print("\n   Tabla comparativa")
    print(f"   {'Algoritmo':<10}{'Expandidos':<12}{'Ruta':<14}{'Costo':<7}Óptimo")
    print("   " + "-" * 52)
    optimo = costo_a
    for nombre, r, c, o in (("BFS", r_bfs, c_bfs, o_bfs),
                            ("UCS", r_ucs, c_ucs, o_ucs),
                            ("Voraz", r_vor, c_vor, o_vor),
                            ("A*", ruta_a, costo_a, orden_a)):
        print(f"   {nombre:<10}{len(o):<12}{ruta_str(r):<14}{c:<7}"
              f"{'Sí' if c == optimo else 'No'}")

    # ---- 3.5 heurística inadmisible ----
    print("\n3.5 Heurística modificada: h(C) = 14")
    h_mal = dict(H)
    h_mal["C"] = 14
    print(f"   h*(C) = {h_star['C']}  ->  ¿admisible? "
          f"{es_admisible(h_mal, h_star)}")
    ruta_m, costo_m, orden_m = busqueda_mejor_primero(p, "astar", h_mal,
                                                      verbose=True)
    print(f"   Ruta: {ruta_str(ruta_m)}   Costo: {costo_m} min "
          f"(el óptimo es {costo_a}) -> A* pierde la optimalidad.")


# =============================================================================
# SECCIÓN IV. BÚSQUEDA LOCAL (ASCENSO DE COLINA)
# =============================================================================

# Matriz de costos mínimos entre estaciones (derivada del grafo)
ESTACIONES_R3 = ["B", "C", "D", "E"]
DIST = {
    "A": {"A": 0, "B": 3, "C": 2, "D": 8, "E": 10},
    "B": {"A": 3, "B": 0, "C": 1, "D": 5, "E": 7},
    "C": {"A": 2, "B": 1, "C": 0, "D": 6, "E": 8},
    "D": {"A": 8, "B": 5, "C": 6, "D": 0, "E": 2},
    "E": {"A": 10, "B": 7, "C": 8, "D": 2, "E": 0},
}


def verificar_matriz():
    """Comprueba que DIST coincide con los caminos mínimos reales del grafo."""
    for origen in DIST:
        dist = costos_optimos_a_meta(GRAFO, origen)   # Dijkstra desde 'origen'
        for destino in DIST[origen]:
            assert dist[destino] == DIST[origen][destino], (origen, destino)


def costo_ronda(cromosoma):
    """Costo de A -> c1 -> c2 -> c3 -> c4 -> A usando la matriz DIST."""
    tour = ["A"] + list(cromosoma) + ["A"]
    return sum(DIST[a][b] for a, b in zip(tour, tour[1:]))


def vecinos_adyacentes(sol):
    """Vecindario: intercambiar dos posiciones ADYACENTES (3 vecinos)."""
    for i in range(len(sol) - 1):
        v = list(sol)
        v[i], v[i + 1] = v[i + 1], v[i]
        yield v


def ascenso_colina(inicial):
    """Ascenso de colina por máxima pendiente (minimizando el costo).

    Se mueve al mejor vecino mientras este sea ESTRICTAMENTE mejor.
    Desempate: primer vecino generado (menor posición de intercambio).
    """
    actual = list(inicial)
    it = 0
    while True:
        c_actual = costo_ronda(actual)
        vecinos = list(vecinos_adyacentes(actual))
        costos = [costo_ronda(v) for v in vecinos]
        detalle = "; ".join(f"{v} {c}" for v, c in zip(vecinos, costos))
        print(f"   Iter {it}: {actual} ({c_actual}) | vecinos: {detalle}")
        mejor_c = min(costos)
        if mejor_c < c_actual:
            actual = vecinos[costos.index(mejor_c)]   # primer mejor
            it += 1
        else:
            print(f"   -> Ningún vecino mejora estrictamente: PARO en "
                  f"{actual} (costo {c_actual})")
            if costos.count(mejor_c) and mejor_c == c_actual:
                print("   -> Hay un vecino con igual costo: es una MESETA.")
            return actual, c_actual


def seccion_4():
    titulo("SECCIÓN IV · BÚSQUEDA LOCAL (ASCENSO DE COLINA)")
    verificar_matriz()
    print("   (La matriz DIST fue verificada contra Dijkstra sobre el grafo)\n")
    print("4.1 Ascenso de colina desde [D, B, C, E]")
    sol, c = ascenso_colina(["D", "B", "C", "E"])
    print(f"\n   Resultado: {sol} con costo {c} min")


# =============================================================================
# SECCIÓN V. ALGORITMO GENÉTICO
# =============================================================================

def aptitud(cromosoma):
    """Aptitud f = 1 / costo (a menor costo, mayor aptitud)."""
    return 1.0 / costo_ronda(cromosoma)


def ruleta_acumulada(poblacion):
    """Devuelve (aptitudes, probabilidades, acumuladas) para la ruleta."""
    apt = [aptitud(x) for x in poblacion]
    total = sum(apt)
    prob = [a / total for a in apt]
    acum = list(itertools.accumulate(prob))
    return apt, prob, acum


def seleccion_ruleta_con_r(acum, r):
    """Elige el primer individuo cuya acumulada es >= r."""
    for i, a in enumerate(acum):
        if r <= a:
            return i
    return len(acum) - 1


def cruce_orden_ox(p1, p2, c1, c2):
    """Cruce por orden (OX).

    - El hijo hereda el segmento p1[c1:c2] (posiciones entre los cortes).
    - El resto se llena con los genes de p2, recorriéndolo en orden circular
      a partir del segundo corte y omitiendo los genes ya heredados.
    - Los huecos se llenan también desde el segundo corte, circularmente.
    Con cortes c1=1, c2=3 reproduce exactamente el procedimiento del examen
    (cortes después de las posiciones 1 y 3).
    """
    n = len(p1)
    hijo = [None] * n
    segmento = p1[c1:c2]
    hijo[c1:c2] = segmento
    orden_p2 = p2[c2:] + p2[:c2]                      # recorrido circular
    restantes = [g for g in orden_p2 if g not in segmento]
    huecos = list(range(c2, n)) + list(range(0, c1))  # también circular
    for pos, gen in zip(huecos, restantes):
        hijo[pos] = gen
    return hijo


def mutacion_intercambio(cromosoma, i, j):
    """Mutación por intercambio de dos posiciones (índices base 0)."""
    m = list(cromosoma)
    m[i], m[j] = m[j], m[i]
    return m


def seccion_5_a_mano():
    titulo("SECCIÓN V · ALGORITMO GENÉTICO (cálculo del examen)")
    poblacion = {
        "X1": ["B", "C", "D", "E"],
        "X2": ["C", "E", "B", "D"],
        "X3": ["D", "B", "C", "E"],
        "X4": ["B", "D", "E", "C"],
    }
    nombres = list(poblacion)
    cromos = list(poblacion.values())

    print("5.1 Costo y aptitud")
    for n, x in poblacion.items():
        print(f"   {n} {x}: costo={costo_ronda(x)}  f=1/costo={aptitud(x):.5f}")

    print("\n5.2 Ruleta")
    apt, prob, acum = ruleta_acumulada(cromos)
    print(f"   Suma de aptitudes = {sum(apt):.5f}")
    for n, p_, a_ in zip(nombres, prob, acum):
        print(f"   {n}: prob={100 * p_:.2f}%  acumulada={a_:.4f}")
    r1, r2 = 0.35, 0.80
    i1 = seleccion_ruleta_con_r(acum, r1)
    i2 = seleccion_ruleta_con_r(acum, r2)
    print(f"   r1={r1} -> {nombres[i1]}   r2={r2} -> {nombres[i2]}")

    print("\n5.3 Cruce OX (cortes después de las posiciones 1 y 3)")
    p1, p2 = cromos[i1], cromos[i2]
    hijo = cruce_orden_ox(p1, p2, 1, 3)
    print(f"   P1={p1}  P2={p2}")
    print(f"   Hijo 1 = {hijo}  costo={costo_ronda(hijo)}")

    print("\n5.4 Mutación: intercambio de posiciones 2 y 4 (índices 1 y 3)")
    mutante = mutacion_intercambio(hijo, 1, 3)
    print(f"   Mutante = {mutante}  costo={costo_ronda(mutante)}  "
          f"(empeoró de {costo_ronda(hijo)} a {costo_ronda(mutante)})")

    print("\n5.5 Un cruce de un punto genera rutas inválidas:")
    corte = 2
    invalido = poblacion["X2"][:corte] + poblacion["X4"][corte:]
    print(f"   X2[:2] + X4[2:] = {invalido}  -> genes repetidos: "
          f"{len(set(invalido)) != len(invalido)}")


def algoritmo_genetico(estaciones, tam_pob=20, generaciones=40, p_cruce=0.9,
                       p_mut=0.1, n_elite=2, semilla=None):
    """AG completo para R3 (corrida real, no solo el ejemplo a mano).

    - Representación : permutación de las estaciones.
    - Aptitud        : 1 / costo de la ronda.
    - Selección      : ruleta.
    - Cruce          : OX con cortes aleatorios.
    - Mutación       : intercambio de dos posiciones.
    - Elitismo       : los n_elite mejores pasan sin cambios.
    - Paro           : número máximo de generaciones.
    Devuelve (mejor_cromosoma, mejor_costo, historial_del_mejor_costo).
    """
    rng = random.Random(semilla)
    n = len(estaciones)
    pob = [rng.sample(estaciones, n) for _ in range(tam_pob)]
    historial = []

    for _ in range(generaciones):
        pob.sort(key=costo_ronda)                       # mejores primero
        historial.append(costo_ronda(pob[0]))
        nueva = [list(x) for x in pob[:n_elite]]        # elitismo

        _, _, acum = ruleta_acumulada(pob)
        while len(nueva) < tam_pob:
            a = pob[seleccion_ruleta_con_r(acum, rng.random())]
            b = pob[seleccion_ruleta_con_r(acum, rng.random())]
            if rng.random() < p_cruce:
                c1, c2 = sorted(rng.sample(range(n + 1), 2))
                hijo = cruce_orden_ox(a, b, c1, c2)
            else:
                hijo = list(a)
            if rng.random() < p_mut:
                i, j = rng.sample(range(n), 2)
                hijo = mutacion_intercambio(hijo, i, j)
            nueva.append(hijo)
        pob = nueva

    pob.sort(key=costo_ronda)
    historial.append(costo_ronda(pob[0]))
    return pob[0], costo_ronda(pob[0]), historial


def seccion_5_corrida_completa():
    titulo("SECCIÓN V · CORRIDA COMPLETA DEL AG (plan de evaluación 6.2)")
    corridas = 30
    costos = []
    for semilla in range(corridas):
        _, c, _ = algoritmo_genetico(ESTACIONES_R3, semilla=semilla)
        costos.append(c)
    mejor, c_mejor, hist = algoritmo_genetico(ESTACIONES_R3, semilla=0)
    print(f"   Corridas independientes : {corridas}")
    print(f"   Mejor costo encontrado  : {min(costos)} min")
    print(f"   Media ± desv. estándar  : {statistics.mean(costos):.2f} ± "
          f"{statistics.pstdev(costos):.2f}")
    print(f"   Corridas que hallaron el óptimo (20): "
          f"{costos.count(20)}/{corridas}")
    print(f"   Ejemplo (semilla 0)     : {mejor} costo={c_mejor}")
    print(f"   Convergencia (mejor costo por generación, primeras 10): "
          f"{hist[:10]}")


# =============================================================================
# VERIFICACIÓN: ENUMERACIÓN EXHAUSTIVA DE R3
# =============================================================================

def verificacion_exhaustiva():
    titulo("VERIFICACIÓN · ENUMERACIÓN EXHAUSTIVA DE LAS 24 RUTAS")
    resultados = sorted((costo_ronda(p), list(p))
                        for p in itertools.permutations(ESTACIONES_R3))
    minimo = resultados[0][0]
    optimas = [r for c, r in resultados if c == minimo]
    print(f"   Rutas evaluadas : {len(resultados)}")
    print(f"   Costo mínimo    : {minimo} min")
    print(f"   Rutas óptimas   : {len(optimas)} -> {optimas}")


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================

def main():
    seccion_1()
    seccion_2()
    seccion_3()
    seccion_4()
    seccion_5_a_mano()
    seccion_5_corrida_completa()
    verificacion_exhaustiva()
    print("\nFin de la solución.")


if __name__ == "__main__":
    main()