from collections import deque  # 🧰 Usamos deque para una cola eficiente (FIFO)

def bfs_laberinto(laberinto, inicio, meta):
    # 🟡 Inicializamos la cola con la posición de inicio
    # Estructura: (fila, columna, camino_recorrido)
    cola = deque([(inicio[0], inicio[1], [])])

    # 🟢 Conjunto para marcar posiciones ya visitadas
    visitados = set([(inicio[0], inicio[1])])

    print(f"🚀 Iniciando búsqueda desde {inicio} hacia {meta}\n")

    while cola:
        fila, col, camino = cola.popleft()
        print(f"📍 Explorando nodo: ({fila}, {col})")
        print(f"🛣️  Camino hasta aquí: {camino}")

        # 🎯 Si llegamos a la meta, devolvemos el camino completo
        if laberinto[fila][col] == "G":
            print("🎯 ¡Meta encontrada!")
            return camino + [(fila, col)]

        # Movimientos posibles: arriba, abajo, izquierda, derecha
        movimientos = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for df, dc in movimientos:
            nf, nc = fila + df, col + dc  # Nueva fila y columna
            if (0 <= nf < len(laberinto) and 0 <= nc < len(laberinto[0])
                and laberinto[nf][nc] != 1  # No es una pared
                and (nf, nc) not in visitados):  # No visitado aún

                visitados.add((nf, nc))  # Marcamos como visitado
                nuevo_camino = camino + [(fila, col)]  # Nuevo camino
                cola.append((nf, nc, nuevo_camino))  # Agregamos a la cola

                print(f"➕ Agregando a la cola: ({nf}, {nc})")
                print(f"   Nuevo camino: {nuevo_camino}\n")

    print("🚫 No se encontró un camino hacia la meta.")
    return None
laberinto = [
    ["S", 0, 0, 1],
    [1,   0, 1, 0],
    [1,   0, 0, "G"]
]

inicio = (0, 0)  # Posición de "S"
meta = (2, 3)    # Posición de "G"

camino = bfs_laberinto(laberinto, inicio, meta)
print("\n✅ Camino encontrado:", camino)
