import heapq

def dijkstra(grafo, inicio, objetivo):
    cola = [(0, [inicio])]  # 🟡 (costo acumulado, camino)
    visitados = set()
    costos = {inicio: 0}  # 🔢 Para guardar el mejor costo a cada nodo

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

            for vecino, peso in grafo[nodo]:
                nuevo_costo = costo_actual + peso

                # Solo agregamos si es mejor camino (o si no hay uno aún)
                if vecino not in costos or nuevo_costo < costos[vecino]:
                    costos[vecino] = nuevo_costo
                    nuevo_camino = list(camino)
                    nuevo_camino.append(vecino)
                    heapq.heappush(cola, (nuevo_costo, nuevo_camino))
                    print(f"➕ Agregando a la cola: {nuevo_camino} | Costo: {nuevo_costo}")
    
    print("🚫 No se encontró camino")
    return None
import heapq

def a_estrella(grafo, inicio, objetivo, heuristica):
    cola = [(heuristica[inicio], 0, [inicio])]  # 🟡 (f = g + h, g, camino)
    visitados = set()

    while cola:
        f_actual, g_actual, camino = heapq.heappop(cola)
        nodo = camino[-1]

        print(f"\n🧭 Explorando: {camino} | Costo real (g): {g_actual} | Estimado total (f): {f_actual}")
        print(f"📌 Nodo actual: {nodo}")

        if nodo == objetivo:
            print("🎯 ¡Objetivo encontrado!")
            print(f"📈 Costo total: {g_actual}")
            return camino

        if nodo not in visitados:
            visitados.add(nodo)

            for vecino, costo in grafo[nodo]:
                if vecino not in visitados:
                    nuevo_g = g_actual + costo
                    nuevo_f = nuevo_g + heuristica[vecino]
                    nuevo_camino = list(camino)
                    nuevo_camino.append(vecino)

                    heapq.heappush(cola, (nuevo_f, nuevo_g, nuevo_camino))
                    print(f"➕ Agregando a la cola: {nuevo_camino} | g: {nuevo_g} | h: {heuristica[vecino]} | f: {nuevo_f}")

    print("🚫 No se encontró camino")
    return None


# Ejemplo de uso
grafo = {
    'A': [('B', 1), ('C', 4)],
    'B': [('A', 1), ('C', 2), ('D', 5)],
    'C': [('A', 4), ('B', 2), ('D', 1)],
    'D': [('B', 5), ('C', 1)]
}
inicio = 'A'
objetivo = 'D'
camino = dijkstra(grafo, inicio, objetivo)
print("\n✅ Camino encontrado:", camino)