mapa={
    'LAB': ['CAFE', 'AULA'],
    'CAFE': ['BIBLIO'],
    'AULA': ['AUDI', 'BIBLIO'],
    'AUDI': ['BIBLIO'],
    'BIBLIO': [],
}

inicio = 'LAB'
objetivo = 'BIBLIO'

def eval_bfs(mapa, inicio, objetivo):
    visitados = set([inicio])
    bfs = [inicio]
    
    while bfs:
        nodo_act = bfs.pop(0)
        
        if nodo_act == objetivo:
            print(f"Llegó de {inicio} a {objetivo}")
            return True
            
        for vecino in mapa[nodo_act]:
            if vecino not in visitados:
                visitados.add(vecino)
                bfs.append(vecino)
                
    print(f"No se encontró ruta de {inicio} a {objetivo}")
    return False
    
eval_bfs(mapa, inicio, objetivo)
