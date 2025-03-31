from openpyxl import load_workbook

# Cargar el archivo Excel
wb = load_workbook("Ecuador_Distancias.xlsx", data_only=True)
sheet = wb["CUADRO DE DISTANCIAS"]  # Cambia el nombre de la hoja si es necesario

# Crear un diccionario para almacenar las distancias
distancias = {}

# Leer los nombres de las ciudades (columna B, desde la fila 4 en adelante)
ciudades = [sheet.cell(row=row, column=2).value for row in range(4, 44)]  # Ajusta el rango según tu archivo

# Construir el diccionario
for row in range(4, 44):  # Ajusta el rango según las filas de tu archivo
    ciudad_origen = sheet.cell(row=row, column=2).value  # Columna B tiene los nombres de las ciudades
    distancias[ciudad_origen] = {
        ciudades[col - 3]: sheet.cell(row=row, column=col).value for col in range(3, 43)  # Columnas C en adelante
    }

# Mostrar el diccionario resultante
print(distancias)

# Acceder a una distancia específica
print(f"Distancia de Ambato a Azogues: {distancias['Ambato']['Azogues']} km")