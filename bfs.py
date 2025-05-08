from collections import deque  # Importamos una cola eficiente

def bfs(grafo, inicio, meta):
    # Inicializamos una cola con el nodo inicial y su camino
    cola = deque([ (inicio, [inicio]) ])  # Ejemplo: ('A', ['A'])
    visitados = set()  # Conjunto para nodos ya vistos

    while cola:  # Mientras la cola no esté vacía
        nodo, camino = cola.popleft()  # Sacamos el primer nodo de la cola
        if nodo == meta:
            return camino  # Si encontramos la meta, retornamos el camino
        
        for vecino in grafo[nodo]:  # Recorremos los vecinos del nodo actual
            if vecino not in visitados:  # Si no lo hemos visitado
                visitados.add(vecino)  # Lo marcamos como visitado
                cola.append((vecino, camino + [vecino]))  # Lo agregamos a la cola

    return None  # Si no se encuentra la meta

# Grafo de ejemplo (diccionario de listas)
grafo = {
    "A": ["B", "C", "D"],
    "B": ["A", "C", "H", "F"],
    "C": ["A", "B", "D", "E", "I", "H"],
    "D": ["A", "C", "E"],
    "E": ["C", "D", "G", "H"],
    "F": ["B", "G", "H"],
    "G": ["E", "F", "H"],
    "H": ["B", "C", "E", "G", "F", "I"],
    "I": ["C", "H"]
}

# Llamamos a la función
print(bfs(grafo, 'A', 'F'))  # Salida: ['A', 'B', 'E', 'F']