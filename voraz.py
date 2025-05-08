import heapq

def distancia_manhattan(nodo_actual, meta, coordenadas):
    x1, y1 = coordenadas[nodo_actual]
    x2, y2 = coordenadas[meta]
    return abs(x1 - x2) + abs(y1 - y2)

def busqueda_voraz(grafo, inicio, meta, coordenadas):
    heap = [(0, inicio, [inicio])]  # (h(n), nodo, camino)
    visitados = set()
    while heap:
        _, nodo, camino = heapq.heappop(heap)
        if nodo == meta:
            return camino
        if nodo not in visitados:
            visitados.add(nodo)
            for vecino in grafo[nodo]:
                h = distancia_manhattan(vecino, meta, coordenadas)  # Calcula h(n) para el vecino
                heapq.heappush(heap, (h, vecino, camino + [vecino]))
    return None

grafo = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F', 'G'],
    'D': [],
    'E': ['H'],
    'F': [],
    'G': [],
    'H': ['I'],
    'I': []
}

coordenadas = {
    'A': (7, 9),
    'B': (5, 10),
    'C': (5, 5),
    'D': (7, 6),
    'E': (4, 0),
    'F': (0, 8),
    'G': (0, 0),
    'H': (1, 5),
    'I': (3, 5)
}

print(busqueda_voraz(grafo, 'A', 'I', coordenadas))  # Salida: ['A', 'B', 'E', 'H', 'I']