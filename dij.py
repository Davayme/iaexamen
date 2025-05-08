import heapq

def dijkstra_meta(grafo, inicio, meta):
    heap = [(0, inicio, [inicio])]
    visitados = set()
    costos_minimos = {nodo: float('inf') for nodo in grafo}
    costos_minimos[inicio] = 0

    while heap:
        costo, nodo, camino = heapq.heappop(heap)
        if nodo == meta:
            return camino
        if nodo not in visitados:
            visitados.add(nodo)
            for vecino, costo_arista in grafo[nodo].items():
                nuevo_costo = costo + costo_arista
                if nuevo_costo < costos_minimos[vecino]:
                    costos_minimos[vecino] = nuevo_costo
                    heapq.heappush(heap, (nuevo_costo, vecino, camino + [vecino]))
    return None

grafo = {
    'A': {'B': 2, 'C': 3},
    'B': {'D': 1, 'E': 4},
    'C': {'G': 1},
    'D': {},
    'E': {'H': 2},
    'G': {},
    'H': {}
}

print(dijkstra_meta(grafo, 'A', 'H'))