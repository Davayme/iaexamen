"""
Módulo para la visualización de grafos y rutas
"""
import matplotlib.pyplot as plt
import networkx as nx

class GraficadorRutas:
    def __init__(self):
        self.figura_size = (12, 8)
        self.colores = {
            'nodo_normal': '#e67e22',      # Naranja
            'nodo_ruta': '#e74c3c',        # Rojo
            'borde_normal': '#d35400',     # Naranja oscuro
            'borde_ruta': '#c0392b',       # Rojo oscuro
            'arista_normal': '#95a5a6',    # Gris
            'arista_ruta': '#c0392b',      # Rojo oscuro
            'texto': '#2c3e50'             # Azul oscuro
        }

    def visualizar_ruta(self, grafo, ruta):
        """Visualiza una ruta específica en el grafo"""
        plt.figure(figsize=self.figura_size)
        pos = nx.spring_layout(grafo, k=1, iterations=50)
        
        # Dibujar grafo base
        self._dibujar_grafo_base(grafo, pos)
        
        # Resaltar ruta
        if ruta:
            self._resaltar_ruta(grafo, pos, ruta)
        
        plt.title("Mapa de Ruta")
        plt.axis('off')
        plt.show()

    def visualizar_mapa_completo(self, grafo):
        """Visualiza el grafo completo"""
        plt.figure(figsize=self.figura_size)
        pos = nx.spring_layout(grafo, k=1, iterations=50)
        
        self._dibujar_grafo_base(grafo, pos)
        
        plt.title("Mapa Completo de Conexiones")
        plt.axis('off')
        plt.show()

    def _dibujar_grafo_base(self, grafo, pos):
        """Dibuja el grafo base con el estilo definido"""
        nx.draw_networkx_nodes(
            grafo, 
            pos, 
            node_color=self.colores['nodo_normal'],
            node_size=500,
            alpha=0.7,
            edgecolors=self.colores['borde_normal']
        )
        nx.draw_networkx_edges(
            grafo, 
            pos,
            edge_color=self.colores['arista_normal'],
            alpha=0.3,
            width=0.8
        )
        nx.draw_networkx_labels(
            grafo, 
            pos, 
            font_size=8,
            font_color=self.colores['texto']
        )

    def _resaltar_ruta(self, grafo, pos, ruta):
        """Resalta una ruta específica en el grafo"""
        path_edges = list(zip(ruta[:-1], ruta[1:]))
        nx.draw_networkx_nodes(
            grafo, 
            pos, 
            nodelist=ruta, 
            node_color=self.colores['nodo_ruta'],
            node_size=700,
            edgecolors=self.colores['borde_ruta']
        )
        nx.draw_networkx_edges(
            grafo, 
            pos, 
            edgelist=path_edges, 
            edge_color=self.colores['arista_ruta'],
            width=2
        )