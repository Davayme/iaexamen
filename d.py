import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tkinter import Tk, Label, ttk, Frame, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import heapq
class SistemaRutas:
    def __init__(self):
        self.grafo = None
        self.locaciones = None
        self.inicializar_sistema()
        
    def inicializar_sistema(self):
        try:
            self.cargar_datos_desde_excel()
            self.crear_interfaz()
        except Exception as e:
            messagebox.showerror("Error", f"Error al inicializar el sistema: {str(e)}")
            
    def cargar_datos_desde_excel(self):
        """Carga datos desde archivo Excel y construye el grafo de conexiones"""
        datos = pd.read_excel(
            "Ecuador_Distancias.xlsx", 
            sheet_name="CUADRO DE DISTANCIAS", 
            header=2
        )
        self.locaciones = datos['CIUDAD'].tolist()
        self.grafo = nx.Graph()
        
        for i in range(len(self.locaciones)):
            for j in range(i + 1, len(self.locaciones)):
                distancia = datos.iloc[i, j + 2]
                if pd.notna(distancia) and distancia > 0:
                    self.grafo.add_edge(
                        self.locaciones[i], 
                        self.locaciones[j], 
                        distancia=distancia
                    )

    def encontrar_mejor_ruta(self, inicio, destino):
        """Implementa búsqueda de costo uniforme (UCS) para encontrar la mejor ruta"""
        if inicio == destino:
            return None, None, "⚠️ Origen y destino deben ser diferentes"
            
        try:
            cola_prioridad = [(0, inicio, [inicio])]
            visitados = set()
            
            while cola_prioridad:
                costo_actual, ubicacion_actual, ruta = heapq.heappop(cola_prioridad)
                
                if ubicacion_actual == destino:
                    return ruta, costo_actual, None
                    
                if ubicacion_actual in visitados:
                    continue
                    
                visitados.add(ubicacion_actual)
                
                for siguiente in self.grafo[ubicacion_actual]:
                    if siguiente not in visitados:
                        nuevo_costo = costo_actual + self.grafo[ubicacion_actual][siguiente]['distancia']
                        nueva_ruta = ruta + [siguiente]
                        heapq.heappush(cola_prioridad, (nuevo_costo, siguiente, nueva_ruta))
                        
            return None, None, "❌ No existe ruta disponible"
            
        except Exception as e:
            return None, None, f"❌ Error al calcular ruta: {str(e)}"

    def visualizar_ruta(self, ruta, costo_total):
        """Visualiza la ruta en el mapa y muestra detalles"""
        # Limpiar visualización anterior
        for widget in self.frame_mapa.winfo_children():
            widget.destroy()
        for widget in self.frame_info.winfo_children():
            widget.destroy()
            
        # Configurar visualización
        figura = plt.figure(figsize=(10, 8))
        posiciones = nx.spring_layout(self.grafo, seed=42, k=0.9)
        
        # Dibujar grafo base
        nx.draw_networkx_nodes(
            self.grafo, posiciones,
            node_size=600,
            node_color='lightblue',
            edgecolors='navy'
        )
        nx.draw_networkx_edges(
            self.grafo, posiciones,
            width=1.0,
            edge_color='gray',
            alpha=0.5
        )
        nx.draw_networkx_labels(
            self.grafo, posiciones,
            font_size=8,
            font_weight='bold'
        )
        
        # Resaltar ruta seleccionada
        if ruta:
            conexiones_ruta = list(zip(ruta[:-1], ruta[1:]))
            nx.draw_networkx_edges(
                self.grafo, posiciones,
                edgelist=conexiones_ruta,
                width=3.0,
                edge_color='crimson'
            )
            nx.draw_networkx_nodes(
                self.grafo, posiciones,
                nodelist=ruta,
                node_color='lightcoral',
                node_size=800
            )
            
        # Mostrar mapa
        canvas = FigureCanvasTkAgg(figura, master=self.frame_mapa)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Mostrar detalles de la ruta
        if ruta:
            self.mostrar_detalles_ruta(ruta, costo_total)

    def mostrar_detalles_ruta(self, ruta, costo_total):
        """Muestra información detallada de la ruta"""
        # Título
        Label(
            self.frame_info,
            text="📍 Detalles del Recorrido",
            font=('Helvetica', 14, 'bold'),
            fg='navy'
        ).pack(pady=10)
        
        # Segmentos de la ruta
        for i in range(len(ruta)-1):
            origen = ruta[i]
            destino = ruta[i+1]
            distancia = self.grafo[origen][destino]['distancia']
            
            Label(
                self.frame_info,
                text=f"➡️ {origen} a {destino}: {distancia} km",
                font=('Helvetica', 10),
                fg='darkslategray'
            ).pack(anchor='w', pady=2)
            
        # Distancia total
        Label(
            self.frame_info,
            text=f"\n🏁 Distancia total: {costo_total} km",
            font=('Helvetica', 12, 'bold'),
            fg='crimson'
        ).pack(pady=10)

    def crear_interfaz(self):
        """Configura la interfaz gráfica del sistema"""
        self.ventana = Tk()
        self.ventana.title("🗺️ Navegador de Rutas - Ecuador")
        self.ventana.geometry("1300x800")
        self.ventana.configure(bg='white')
        
        # Marco principal
        marco_principal = Frame(self.ventana, bg='white')
        marco_principal.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Panel de control
        panel_control = Frame(marco_principal, bg='white', relief='ridge', bd=2)
        panel_control.pack(side='left', fill='y', padx=10)
        
        # Elementos del panel de control
        Label(
            panel_control,
            text="🚗 Planificador de Viaje",
            font=('Helvetica', 16, 'bold'),
            bg='white',
            fg='navy'
        ).pack(pady=20)
        
        # Selectores de ciudades
        Label(
            panel_control,
            text="📍 Punto de Partida:",
            font=('Helvetica', 12),
            bg='white'
        ).pack(pady=5)
        
        self.origen = ttk.Combobox(
            panel_control,
            values=self.locaciones,
            font=('Helvetica', 12),
            state='readonly'
        )
        self.origen.pack(pady=5)
        
        Label(
            panel_control,
            text="🎯 Destino:",
            font=('Helvetica', 12),
            bg='white'
        ).pack(pady=5)
        
        self.destino = ttk.Combobox(
            panel_control,
            values=self.locaciones,
            font=('Helvetica', 12),
            state='readonly'
        )
        self.destino.pack(pady=5)
        
        # Botón de búsqueda
        ttk.Button(
            panel_control,
            text="🔍 Buscar Ruta",
            command=self.calcular_ruta,
            style='Accent.TButton'
        ).pack(pady=20)
        
        self.mensaje_error = Label(
            panel_control,
            text="",
            font=('Helvetica', 10),
            bg='white',
            fg='red',
            wraplength=200
        )
        self.mensaje_error.pack(pady=10)
        
        # Panel de visualización
        panel_visual = Frame(marco_principal, bg='white')
        panel_visual.pack(side='right', fill='both', expand=True)
        
        self.frame_mapa = Frame(panel_visual, bg='white', relief='ridge', bd=2)
        self.frame_mapa.pack(fill='both', expand=True)
        
        self.frame_info = Frame(panel_visual, bg='white', relief='ridge', bd=2)
        self.frame_info.pack(fill='both', expand=True)
        
        # Estilo personalizado
        style = ttk.Style()
        style.configure('Accent.TButton', font=('Helvetica', 12))
        
        self.ventana.mainloop()

    def calcular_ruta(self):
        """Maneja el evento de búsqueda de ruta"""
        inicio = self.origen.get()
        fin = self.destino.get()
        
        if not inicio or not fin:
            self.mensaje_error.config(text="⚠️ Selecciona origen y destino")
            return
            
        ruta, costo, error = self.encontrar_mejor_ruta(inicio, fin)
        
        if error:
            self.mensaje_error.config(text=error)
            return
            
        self.mensaje_error.config(text="")
        self.visualizar_ruta(ruta, costo)

if __name__ == "__main__":
    sistema = SistemaRutas()