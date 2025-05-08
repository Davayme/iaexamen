from collections import deque
import heapq
def bfs(grafo, inicio, objetivo):
    cola = deque([[inicio]])
    visitados = set()

    print(f"🚀 Iniciando Búsqueda en Amplitud (BFS) desde '{inicio}' hasta '{objetivo}'\n")
    
    while cola:
        camino = cola.popleft()
        nodo = camino[-1]

        print(f"🔍 Explorando camino actual: {camino}")
        print(f"📌 Nodo actual: {nodo}")
        print(f"✅ Nodos visitados: {list(visitados)}")
        print(f"📂 Caminos restantes en cola: {list(cola)}")
        
        if nodo == objetivo:
            print(f"\n🎯 ¡Objetivo encontrado! Ruta completa: {camino}")
            return camino

        if nodo not in visitados:
            visitados.add(nodo)
            print(f"➕ Marcando '{nodo}' como visitado.")
            for vecino in grafo[nodo]:
                if vecino not in visitados:
                    nuevo_camino = list(camino)
                    nuevo_camino.append(vecino)
                    cola.append(nuevo_camino)
                    print(f"  🔗 Vecino '{vecino}' encontrado. Agregando nuevo camino: {nuevo_camino}")
    
    print("\n🚫 No se encontró un camino hacia el objetivo.")
    return None

def dfs(grafo, inicio, objetivo):
    pila = [[inicio]]  # LIFO
    visitados = set()

    while pila:
        camino = pila.pop()  # saca el último camino
        nodo = camino[-1] # último nodo del camino

        if nodo == objetivo:
            return camino

        if nodo not in visitados:
            visitados.add(nodo)
            for vecino in grafo[nodo]:
                if vecino not in visitados:
                    nuevo_camino = list(camino)
                    nuevo_camino.append(vecino)
                    pila.append(nuevo_camino)
    
    return None

# Implementación del algoritmo de búsqueda de costo uniforme (Uniform Cost Search - UCS)
def ucs(grafo, inicio, objetivo):
    cola = [(0, [inicio])]  # 🟡 (costo acumulado, camino)
    visitados = set()

    while cola:
        # ✅ Sacamos el camino con menor costo
        costo_actual, camino = heapq.heappop(cola)
        nodo = camino[-1]

        print(f"\n🧭 Explorando: {camino} | Costo: {costo_actual}")
        print(f"📌 Nodo actual: {nodo}")
        
        # 🎯 Si llegamos al objetivo, terminamos
        if nodo == objetivo:
            print("🎯 ¡Objetivo encontrado!")
            print(f"📈 Costo total: {costo_actual}")
            return camino

        # ✅ Expandimos si aún no fue visitado
        if nodo not in visitados:
            visitados.add(nodo)

            for vecino, costo in grafo[nodo]:
                if vecino not in visitados:
                    nuevo_camino = list(camino)
                    nuevo_camino.append(vecino)
                    nuevo_costo = costo_actual + costo

                    # ➕ Agregamos el nuevo camino con su costo
                    heapq.heappush(cola, (nuevo_costo, nuevo_camino))
                    print(f"➕ Agregando a la cola: {nuevo_camino} | Costo: {nuevo_costo}")
    
    print("🚫 No se encontró camino")
    return None
# Grafo de ejemplo
grafo = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

bfs(grafo, 'A', 'F')

# Graficar el grafo
import networkx as nx
import matplotlib.pyplot as plt

def graficar_grafo(grafo):
    G = nx.Graph()
    for nodo, vecinos in grafo.items():
        for vecino in vecinos:
            G.add_edge(nodo, vecino)
    
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=700, font_size=10, font_weight='bold')
    plt.title("Grafo de Conexiones")
    plt.show()

graficar_grafo(grafo)