import networkx as nx
import matplotlib.pyplot as plt

def graficar_grafo_mejorado(grafo):
    # Crear un grafo no dirigido
    G = nx.Graph()
    
    # Añadir aristas desde nuestro diccionario
    for nodo, vecinos in grafo.items():
        for vecino in vecinos:
            G.add_edge(nodo, vecino)
    
    # Configurar la visualización
    plt.figure(figsize=(12, 10))
    
    # Mejorar el posicionamiento usando un layout más espaciado
    # Aumentamos la distancia entre nodos (k=1.5) y más iteraciones
    pos = nx.kamada_kawai_layout(G)  # Mejor algoritmo para grafos densos
    
    # Dibujar aristas primero para que estén detrás de los nodos
    nx.draw_networkx_edges(G, pos, 
                         width=1.2, 
                         alpha=0.7,
                         edge_color='gray')
    
    # Dibujar nodos con tamaño más adecuado
    nx.draw_networkx_nodes(G, pos, 
                         node_size=800, 
                         node_color='skyblue', 
                         edgecolors='navy',
                         linewidths=2.0)
    
    # Dibujar etiquetas de nodos con un fondo para mejor legibilidad
    label_options = {"font_size": 15,
                    "font_weight": 'bold',
                    "bbox": dict(boxstyle="round,pad=0.3", fc="white", ec="navy", alpha=0.8),
                    "horizontalalignment": 'center',
                    "verticalalignment": 'center'}
    nx.draw_networkx_labels(G, pos, font_color='navy', **label_options)
    
    plt.title("Grafo del problema BFS", fontsize=18, pad=20)
    plt.axis('off')  # Eliminar ejes
    plt.tight_layout()
    plt.show()

# Usar el grafo definido en tu código
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

# Llamar a la función para visualizar el grafo mejorado
graficar_grafo_mejorado(grafo)