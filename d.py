from openpyxl import load_workbook
import json

# Cargar el archivo Excel
wb = load_workbook("Ecuador_Distancias.xlsx", data_only=True)
sheet = wb["CUADRO DE DISTANCIAS"]  # Cambia el nombre de la hoja si es necesario

# Leer los nombres de las ciudades (columna B, desde la fila 4 en adelante)
ciudades = [sheet.cell(row=row, column=2).value for row in range(4, 44)]  # Ajusta el rango según tu archivo

# Crear un diccionario para almacenar las distancias
distancias = {}

# Leer las distancias desde el Excel
for i, ciudad_origen in enumerate(ciudades):
    distancias[ciudad_origen] = {}
    for j, ciudad_destino in enumerate(ciudades):
        # Leer la distancia desde la celda correspondiente
        distancia = sheet.cell(row=i + 4, column=j + 3).value  # Ajusta las coordenadas según tu archivo
        if distancia is not None and distancia > 0:  # Ignorar distancias no válidas o cero
            distancias[ciudad_origen][ciudad_destino] = distancia

# Crear una lista para almacenar las conexiones
conexiones = []
ciudades_conectadas = set()

# Conectar las ciudades basándonos en la distancia más corta
for origen in ciudades:
    if origen not in ciudades_conectadas:
        # Filtrar las ciudades con distancias válidas y que no estén conectadas aún
        destinos_validos = {ciudad: distancia for ciudad, distancia in distancias[origen].items() if ciudad not in ciudades_conectadas and ciudad != origen}
        if destinos_validos:
            # Encontrar la ciudad más cercana
            ciudad_mas_cercana = min(destinos_validos, key=destinos_validos.get)
            # Agregar la conexión
            conexiones.append((origen, ciudad_mas_cercana, destinos_validos[ciudad_mas_cercana]))
            ciudades_conectadas.add(origen)
            ciudades_conectadas.add(ciudad_mas_cercana)
            print(f"Conexión añadida: {origen} - {ciudad_mas_cercana}, Distancia: {destinos_validos[ciudad_mas_cercana]} km")

# Validar que todas las ciudades estén conectadas
for origen in ciudades:
    if origen not in ciudades_conectadas:
        print(f"La ciudad {origen} no está conectada. Buscando una conexión...")
        for destino, distancia in distancias[origen].items():
            if destino in ciudades_conectadas:
                conexiones.append((origen, destino, distancia))
                ciudades_conectadas.add(origen)
                print(f"Conexión añadida: {origen} - {destino}, Distancia: {distancia} km")
                break

# Guardar las conexiones en un archivo JSON
with open("conexiones_geograficas.json", "w", encoding="utf-8") as archivo:
    json.dump(conexiones, archivo, ensure_ascii=False, indent=4)

print("Conexiones geográficas guardadas en 'conexiones_geograficas.json'.")