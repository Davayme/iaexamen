import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import heapq
import os
from time import sleep

class BuscadorRutas:
    """
    Clase principal que implementa la funcionalidad del buscador de rutas.
    Gestiona el grafo de conexiones y las operaciones de búsqueda.
    """
    
    def __init__(self):
        """Inicializa el sistema y carga los datos"""
        self.grafo = None      # Grafo de conexiones entre ciudades
        self.ciudades = None   # Lista de ciudades disponibles
        self.cargar_datos()    
        
    def limpiar_pantalla(self):
        """Limpia la pantalla de la consola"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def cargar_datos(self):
        """
        Carga los datos desde Excel y construye el grafo de conexiones.
        El archivo debe contener una matriz de distancias entre ciudades.
        """
        try:
            print("Cargando datos de ciudades...")
            datos = pd.read_excel(
                "Ecuador_Distancias.xlsx", 
                sheet_name="CUADRO DE DISTANCIAS", 
                header=2
            )
            self.ciudades = datos['CIUDAD'].tolist()
            self.grafo = nx.Graph()
            
            # Construir grafo de conexiones
            for i in range(len(self.ciudades)):
                for j in range(i + 1, len(self.ciudades)):
                    distancia = datos.iloc[i, j + 2]
                    if pd.notna(distancia) and distancia > 0:
                        self.grafo.add_edge(
                            self.ciudades[i], 
                            self.ciudades[j], 
                            distancia=distancia
                        )
            print("✅ Datos cargados exitosamente!")
            sleep(1)
        except Exception as e:
            print(f"❌ Error al cargar datos: {str(e)}")
            exit(1)

    def mostrar_menu(self):
        """Gestiona el menú principal y la interacción con el usuario"""
        while True:
            self.limpiar_pantalla()
            print("\n=== 🗺️ Buscador de Rutas - Ecuador ===")
            print("\n1. 🔍 Buscar ruta entre ciudades")
            print("2. 📋 Ver lista de ciudades")
            print("3. 🌐 Ver mapa completo")
            print("4. ❌ Salir")
            
            opcion = input("\nSeleccione una opción (1-4): ")
            
            if opcion == "1":
                self.buscar_ruta()
            elif opcion == "2":
                self.mostrar_ciudades()
            elif opcion == "3":
                self.visualizar_mapa_completo()
            elif opcion == "4":
                print("\n¡Gracias por usar el sistema! 👋")
                break
            else:
                print("\n❌ Opción no válida. Intente nuevamente.")
                sleep(1)

    def mostrar_ciudades(self):
        """Muestra la lista numerada de ciudades disponibles"""
        self.limpiar_pantalla()
        print("\n=== 📍 Ciudades Disponibles ===\n")
        for i, ciudad in enumerate(sorted(self.ciudades), 1):
            print(f"{i:2d}. {ciudad}")
        input("\nPresione Enter para continuar...")

    def buscar_ruta(self):
        """
        Gestiona el proceso de búsqueda de rutas entre ciudades.
        Permite al usuario seleccionar origen y destino, y muestra el resultado.
        """
        self.limpiar_pantalla()
        print("\n=== 🔍 Búsqueda de Ruta ===\n")
        
        # Mostrar lista de ciudades ordenadas alfabéticamente
        print("Ciudades disponibles:")
        ciudades_ordenadas = sorted(self.ciudades)
        for i, ciudad in enumerate(ciudades_ordenadas, 1):
            print(f"{i:2d}. {ciudad}")
            
        try:
            # Validar selección de origen
            idx_origen = int(input("\nSeleccione número de ciudad origen: ")) - 1
            if not (0 <= idx_origen < len(ciudades_ordenadas)):
                print("❌ Número de ciudad inválido")
                sleep(1)
                return
                
            # Validar selección de destino
            idx_destino = int(input("Seleccione número de ciudad destino: ")) - 1
            if not (0 <= idx_destino < len(ciudades_ordenadas)):
                print("❌ Número de ciudad inválido")
                sleep(1)
                return
                
            origen = ciudades_ordenadas[idx_origen]
            destino = ciudades_ordenadas[idx_destino]
            
            # Validar que origen y destino sean diferentes
            if origen == destino:
                print("❌ El origen y destino deben ser diferentes")
                sleep(1)
                return
                
            # Calcular y mostrar la ruta
            ruta, distancia = self.calcular_ruta(origen, destino)
            if ruta:
                self.mostrar_resultado(ruta, distancia)
            else:
                print("❌ No existe ruta disponible entre estas ciudades")
                sleep(1)
                
        except ValueError:
            print("❌ Por favor, ingrese números válidos")
            sleep(1)
            return

    def calcular_ruta(self, origen, destino):
        """
        Implementa el algoritmo de búsqueda de costo uniforme (UCS).
        
        Args:
            origen (str): Ciudad de origen
            destino (str): Ciudad de destino
            
        Returns:
            tuple: (ruta, costo_total) o (None, None) si no hay ruta
        """
        cola = [(0, origen, [origen])]  # (costo, ciudad_actual, camino)
        visitados = set()
        
        while cola:
            (costo, actual, camino) = heapq.heappop(cola)
            
            # Si llegamos al destino, retornamos la ruta y su costo
            if actual == destino:
                return camino, costo
                
            if actual in visitados:
                continue
                
            visitados.add(actual)
            
            # Explorar ciudades vecinas no visitadas
            for siguiente in self.grafo[actual]:
                if siguiente not in visitados:
                    nuevo_costo = costo + self.grafo[actual][siguiente]['distancia']
                    nueva_ruta = camino + [siguiente]
                    heapq.heappush(cola, (nuevo_costo, siguiente, nueva_ruta))
        
        return None, None

    def mostrar_resultado(self, ruta, distancia_total):
        """
        Muestra los detalles de la ruta encontrada y la visualiza en el mapa.
        
        Args:
            ruta (list): Lista de ciudades en la ruta
            distancia_total (float): Distancia total de la ruta en km
        """
        self.limpiar_pantalla()
        print("\n=== 🛣️ Ruta Encontrada ===\n")
        
        # Mostrar cada segmento de la ruta
        for i in range(len(ruta)-1):
            origen = ruta[i]
            destino = ruta[i+1]
            distancia = self.grafo[origen][destino]['distancia']
            print(f"➡️  {origen} a {destino}: {distancia} km")
            
        print(f"\n📏 Distancia total: {distancia_total} km")
        
        # Visualizar ruta en el mapa
        self.visualizar_ruta(ruta)
        
        input("\nPresione Enter para continuar...")

    def visualizar_ruta(self, ruta):
        """
        Visualiza la ruta encontrada en un mapa interactivo.
        Resalta los nodos y conexiones que forman parte de la ruta.
        
        Args:
            ruta (list): Lista de ciudades que forman la ruta
        """
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.grafo, k=1, iterations=50)
        
        # Dibujar grafo base
        nx.draw_networkx_nodes(
            self.grafo, 
            pos, 
            node_color='#e67e22',    # Naranja/terracota
            node_size=500, 
            alpha=0.6,
            edgecolors='#d35400'     # Borde naranja oscuro
        )
        nx.draw_networkx_edges(
            self.grafo, 
            pos, 
            alpha=0.2, 
            edge_color='#95a5a6'     # Gris claro
        )
        nx.draw_networkx_labels(
            self.grafo, 
            pos, 
            font_size=8, 
            font_color='#2c3e50'     # Azul oscuro
        )
        
        # Resaltar la ruta encontrada
        path_edges = list(zip(ruta[:-1], ruta[1:]))
        nx.draw_networkx_nodes(
            self.grafo, 
            pos, 
            nodelist=ruta, 
            node_color='#e74c3c',    # Rojo terracota
            node_size=700,
            edgecolors='#c0392b'     # Borde rojo oscuro
        )
        nx.draw_networkx_edges(
            self.grafo, 
            pos, 
            edgelist=path_edges, 
            edge_color='#c0392b',    # Rojo oscuro
            width=2
        )
        
        plt.title("Mapa de Ruta")
        plt.axis('off')
        plt.show()

    def visualizar_mapa_completo(self):
        """
        Muestra el mapa completo de todas las conexiones entre ciudades.
        Utiliza un esquema de colores consistente para mejor visualización.
        """
        self.limpiar_pantalla()
        print("\nGenerando mapa completo...")
        
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.grafo, k=1, iterations=50)
        
        # Dibujar grafo completo
        nx.draw_networkx_nodes(
            self.grafo, 
            pos, 
            node_color='#e67e22',    # Naranja/terracota
            node_size=500,
            alpha=0.7,
            edgecolors='#d35400'     # Borde naranja oscuro
        )
        nx.draw_networkx_edges(
            self.grafo, 
            pos,
            edge_color='#95a5a6',    # Gris claro
            alpha=0.3,
            width=0.8
        )
        nx.draw_networkx_labels(
            self.grafo, 
            pos, 
            font_size=8,
            font_color='#2c3e50'     # Azul oscuro
        )
        
        plt.title("Mapa Completo de Conexiones")
        plt.axis('off')
        plt.show()
        
        input("\nPresione Enter para continuar...")

# Punto de entrada del programa
if __name__ == "__main__":
    sistema = BuscadorRutas()
    sistema.mostrar_menu()