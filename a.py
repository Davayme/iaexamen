import heapq

def distancia_manhattan(nodo, meta, coordenadas):
    x1, y1 = coordenadas[nodo]
    x2, y2 = coordenadas[meta]
    return abs(x1 - x2) + abs(y1 - y2)

def a_estrella(grafo, inicio, meta, coordenadas):
    heap = [(0, inicio, [inicio], 0)]  # (f(n), nodo, camino, g(n))
    visitados = set()
    while heap:
        _, nodo, camino, g = heapq.heappop(heap)
        if nodo == meta:
            return camino
        if nodo not in visitados:
            visitados.add(nodo)
            for vecino, costo in grafo[nodo].items():
                nuevo_g = g + costo
                h = distancia_manhattan(vecino, meta, coordenadas)
                heapq.heappush(heap, (nuevo_g + h, vecino, camino + [vecino], nuevo_g))
    return None

grafo_costos = {
    'A': {'B': 2, 'C': 3},
    'B': {'D': 1, 'E': 4},
    'C': {'F': 5, 'G': 1},
    'D': {},
    'E': {'H': 2},
    'F': {},
    'G': {},
    'H': {'I': 3},
    'I': {}
}

coordenadas = {
    'A': (7, 9), 'B': (5, 10), 'C': (5, 5),
    'D': (7, 6), 'E': (4, 0), 'F': (0, 8),
    'G': (0, 0), 'H': (1, 5), 'I': (3, 5)
}

# Ejecución
print(a_estrella(grafo_costos, 'A', 'I', coordenadas))  # Salida: ['A', 'B', 'E', 'H', 'I']