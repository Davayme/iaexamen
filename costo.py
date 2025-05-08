import heapq  # Necesario para la cola de prioridad

def ucs(grafo, inicio, meta):
    # Inicializamos la cola de prioridad: (costo_acumulado, nodo, camino)
    cola = [(0, inicio, [inicio])]
    visitados = set()  # Conjunto para nodos ya explorados

    while cola:
        costo, nodo, camino = heapq.heappop(cola)  # Sacamos el nodo con menor costo
        if nodo == meta:
            return camino  # ¡Solución encontrada!
        
        if nodo not in visitados:
            visitados.add(nodo)
            # Recorremos los vecinos y sus costos
            for vecino, costo_arista in grafo[nodo].items():
                if vecino not in visitados:
                    nuevo_costo = costo + costo_arista
                    heapq.heappush(cola, (nuevo_costo, vecino, camino + [vecino]))
    
    return None  # No se encontró solución

# Grafo con costos (ej: costo de gasolina entre ciudades)
grafo_costos = {
    'A': {'B': 1, 'C': 3},
    'B': {'D': 2, 'E': 4},
    'C': {'F': 5},
    'D': {},
    'E': {'F': 1},
    'F': {}
}

# Ejecución
print(ucs(grafo_costos, 'A', 'F'))  # Salida: ['A', 'B', 'E', 'F'] (Costo total: 6)