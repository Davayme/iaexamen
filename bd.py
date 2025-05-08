import psycopg2
import pandas as pd
import networkx as nx
from urllib.parse import urlparse


class GestorDB:
    def __init__(self):
        # Credenciales de conexión a PostgreSQL
        self.db_config = {
            'dbname': 'ia_fy6v',
            'user': 'ia_fy6v_user',
            'password': 'ImJOgNpmVF327JL6JAtvk848p7hItwag',
            'host': 'dpg-cvllmbp5pdvs73f76d30-a.oregon-postgres.render.com',  # El host de Render
            'port': '5432'
        }
        self.crear_tablas()
    
    def obtener_conexion(self):
        """Crea y retorna una conexión a PostgreSQL"""
        return psycopg2.connect(**self.db_config)
    
    def crear_tablas(self):
        """Crea las tablas necesarias en PostgreSQL"""
        conn = self.obtener_conexion()
        cursor = conn.cursor()
        
        # Tabla para las ciudades
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS ciudades (
            id SERIAL PRIMARY KEY,
            nombre VARCHAR(100) UNIQUE NOT NULL
        )
        """)
        
        # Tabla para las distancias entre ciudades
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS distancias (
            ciudad_origen VARCHAR(100),
            ciudad_destino VARCHAR(100),
            distancia FLOAT,
            PRIMARY KEY (ciudad_origen, ciudad_destino),
            FOREIGN KEY (ciudad_origen) REFERENCES ciudades(nombre),
            FOREIGN KEY (ciudad_destino) REFERENCES ciudades(nombre)
        )
        """)
        
        conn.commit()
        conn.close()
    
    def importar_excel(self, archivo):
        """Importa datos desde Excel a PostgreSQL"""
        try:
            datos = pd.read_excel(
                archivo,
                sheet_name="CUADRO DE DISTANCIAS",
                header=2
            )
            
            conn = self.obtener_conexion()
            cursor = conn.cursor()
            
            # Guardar ciudades
            ciudades = datos['CIUDAD'].tolist()
            for ciudad in ciudades:
                cursor.execute("""
                INSERT INTO ciudades (nombre)
                VALUES (%s)
                ON CONFLICT (nombre) DO NOTHING
                """, (ciudad,))
            
            # Guardar distancias
            for i in range(len(ciudades)):
                for j in range(i + 1, len(ciudades)):
                    distancia = datos.iloc[i, j + 2]
                    if pd.notna(distancia) and distancia > 0:
                        cursor.execute("""
                        INSERT INTO distancias (ciudad_origen, ciudad_destino, distancia)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (ciudad_origen, ciudad_destino) 
                        DO UPDATE SET distancia = EXCLUDED.distancia
                        """, (ciudades[i], ciudades[j], float(distancia)))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error al importar datos: {str(e)}")
            return False
    
    def obtener_datos(self):
        """Recupera los datos de PostgreSQL"""
        conn = self.obtener_conexion()
        cursor = conn.cursor()
        
        # Obtener lista de ciudades ordenada
        cursor.execute("SELECT nombre FROM ciudades ORDER BY nombre")
        ciudades = [row[0] for row in cursor.fetchall()]
        
        # Crear grafo
        grafo = nx.Graph()
        
        # Agregar conexiones
        cursor.execute("SELECT ciudad_origen, ciudad_destino, distancia FROM distancias")
        for origen, destino, distancia in cursor.fetchall():
            grafo.add_edge(origen, destino, distancia=distancia)
        
        conn.close()
        return grafo, ciudades
    
    def agregar_ruta(self, origen, destino, distancia):
        """Agrega o actualiza una ruta en la base de datos"""
        conn = self.obtener_conexion()
        cursor = conn.cursor()
        
        try:
            # Verificar que las ciudades existan
            cursor.execute("""
            INSERT INTO distancias (ciudad_origen, ciudad_destino, distancia)
            VALUES (%s, %s, %s)
            ON CONFLICT (ciudad_origen, ciudad_destino) 
            DO UPDATE SET distancia = EXCLUDED.distancia
            """, (origen, destino, distancia))
            
            conn.commit()
            return True
        except psycopg2.Error as e:
            print(f"Error al agregar ruta: {str(e)}")
            return False
        finally:
            conn.close()
            