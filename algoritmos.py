from collections import deque
import heapq

class AlgoritmosBusqueda:
    @staticmethod
    def bfs(grafo, origen, destino):
        """Búsqueda en Amplitud (BFS)"""
        cola = deque([(origen, [origen], 0)])
        visitados = {origen}
        
        while cola:
            actual, ruta, costo = cola.popleft()
            
            if actual == destino:
                return ruta, costo
                
            for siguiente in grafo[actual]:
                if siguiente not in visitados:
                    visitados.add(siguiente)
                    nuevo_costo = costo + grafo[actual][siguiente]['distancia']
                    nueva_ruta = ruta + [siguiente]
                    cola.append((siguiente, nueva_ruta, nuevo_costo))
                    
        return None, None

    @staticmethod
    def dfs(grafo, origen, destino):
        """Búsqueda en Profundidad (DFS)"""
        pila = [(origen, [origen], 0)]
        visitados = set()
        
        while pila:
            actual, ruta, costo = pila.pop()
            
            if actual == destino:
                return ruta, costo
                
            if actual not in visitados:
                visitados.add(actual)
                
                for siguiente in sorted(grafo[actual], reverse=True):
                    if siguiente not in visitados:
                        nuevo_costo = costo + grafo[actual][siguiente]['distancia']
                        nueva_ruta = ruta + [siguiente]
                        pila.append((siguiente, nueva_ruta, nuevo_costo))
                        
        return None, None

    @staticmethod
    def ucs(grafo, origen, destino):
        """Búsqueda de Costo Uniforme (UCS)"""
        cola = [(0, origen, [origen])]
        visitados = set()
        
        while cola:
            costo, actual, ruta = heapq.heappop(cola)
            
            if actual == destino:
                return ruta, costo
                
            if actual in visitados:
                continue
                
            visitados.add(actual)
            
            for siguiente in grafo[actual]:
                if siguiente not in visitados:
                    nuevo_costo = costo + grafo[actual][siguiente]['distancia']
                    nueva_ruta = ruta + [siguiente]
                    heapq.heappush(cola, (nuevo_costo, siguiente, nueva_ruta))
                    
        return None, None