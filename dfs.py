from collections import deque  # También podemos usar deque para implementar una pila

def dfs(grafo, inicio, meta):
    # Inicializamos una pila con el nodo inicial y su camino
    pila = [ (inicio, [inicio]) ]  # Ejemplo: ('A', ['A'])
    visitados = set()  # Conjunto para nodos ya vistos

    while pila:  # Mientras la pila no esté vacía
        nodo, camino = pila.pop()  # Sacamos el último nodo de la pila
        if nodo == meta:
            return camino  # Si encontramos la meta, retornamos el camino
        
        if nodo not in visitados:  # Si no lo hemos visitado
            visitados.add(nodo)  # Lo marcamos como visitado
            # Agregamos los vecinos en orden inverso para explorar izquierda primero
            for vecino in reversed(grafo[nodo]):  
                pila.append((vecino, camino + [vecino]))  # Apilamos el vecino

    return None  # Si no se encuentra la meta
 # Si no se encuentra la meta

# Grafo de ejemplo (mismo que en BFS)
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
print(dfs(grafo, 'A', 'F'))  # Ejemplo de salida: ['A', 'B', 'F']