import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import os
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import seaborn as sns

print("Iniciando generación de dataset mejorado para predicción de popularidad musical...")
start_time = time.time()

# Configuración de semillas para reproducibilidad
np.random.seed(42)
random.seed(42)

# Artistas populares (combinando varios géneros y épocas)
artistas = [
    "Bad Bunny", "The Weeknd", "BTS", "Drake", "Taylor Swift", 
    "Billie Eilish", "Post Malone", "Dua Lipa", "Ed Sheeran", "Ariana Grande",
    "Justin Bieber", "Travis Scott", "Juice WRLD", "Harry Styles", "J Balvin",
    "Cardi B", "Olivia Rodrigo", "Lil Nas X", "Doja Cat", "DaBaby",
    "Shawn Mendes", "Megan Thee Stallion", "Daddy Yankee", "Imagine Dragons", "Blackpink",
    "Camila Cabello", "Lil Uzi Vert", "Maluma", "Coldplay", "Kendrick Lamar",
    "Lady Gaga", "Rosalía", "Lil Baby", "Karol G", "Bruno Mars",
    "The Kid LAROI", "SZA", "Jack Harlow", "Rauw Alejandro", "Beyoncé", 
    "Eminem", "Rihanna", "Adele", "Kanye West", "Miley Cyrus", "Twenty One Pilots", 
    "Halsey", "Selena Gomez", "Charlie Puth", "Anuel AA", "Peso Pluma", "Feid",
    "Anitta", "Katy Perry", "Maroon 5", "Shakira", "Ozuna", "Nicki Minaj",
    "Young Thug", "Migos", "Jhayco", "Calvin Harris", "David Guetta", 
    "Martin Garrix", "Alan Walker", "Lewis Capaldi", "Sam Smith", "Demi Lovato",
    "Zara Larsson", "Tini", "María Becerra", "Paulo Londra", "Manuel Turizo",
    "Sebastian Yatra", "Myke Towers", "Don Omar", "Wisin & Yandel", "Nicky Jam", "Farruko"
]

# Eliminar duplicados si hay alguno
artistas = list(set(artistas))
print(f"Total de artistas: {len(artistas)}")

# Géneros musicales con categorización clara
generos = [
    'Pop', 'Hip Hop', 'R&B', 'Reggaeton', 'K-Pop', 'Rock', 'Electrónica', 
    'Country', 'Trap', 'Indie', 'Soul', 'Jazz', 'Metal', 'Folk', 'Clásica',
    'Latino', 'Dance', 'House', 'Drill', 'Funk', 'Disco', 'Flamenco', 
    'Rap', 'Punk', 'Blues', 'Alternativo', 'Techno', 'Pop Latino',
    'Urbano', 'Experimental', 'Lofi', 'Synthwave', 'Tropical'
]

# Escalas musicales
escalas = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Estaciones del año
estaciones = ['Primavera', 'Verano', 'Otoño', 'Invierno']

# Tendencias de popularidad por género - más balanceada
tendencias_genero = {}
base_popularidad_generos = {
    'Pop': 0.65, 'Hip Hop': 0.7, 'R&B': 0.55, 'Reggaeton': 0.72, 'K-Pop': 0.68,
    'Rock': 0.5, 'Electrónica': 0.6, 'Country': 0.45, 'Trap': 0.67, 'Indie': 0.48,
    'Soul': 0.42, 'Jazz': 0.38, 'Metal': 0.4, 'Folk': 0.35, 'Clásica': 0.3,
    'Latino': 0.63, 'Dance': 0.58, 'House': 0.52, 'Drill': 0.54, 'Funk': 0.45,
    'Disco': 0.4, 'Flamenco': 0.38, 'Rap': 0.65, 'Punk': 0.41, 'Blues': 0.36,
    'Alternativo': 0.46, 'Techno': 0.49, 'Pop Latino': 0.71, 'Urbano': 0.69,
    'Experimental': 0.32, 'Lofi': 0.4, 'Synthwave': 0.42, 'Tropical': 0.52
}

# Tendencias a lo largo de los años (2010-2023)
for genero in generos:
    base = base_popularidad_generos.get(genero, 0.5)  # Valor base según el género
    tendencia = []
    for i in range(14):  # 2010-2023
        # Añadir fluctuación aleatoria realista
        cambio = random.uniform(-0.1, 0.1)
        tendencia_anual = base + cambio + 0.01 * i * random.uniform(-0.5, 0.5)
        tendencia.append(max(0.3, min(0.9, tendencia_anual)))  # Limitar entre 0.3 y 0.9
    tendencias_genero[genero] = tendencia

print("Generando perfiles para artistas...")
# Asignar características específicas a artistas
perfiles_artistas = {}
for artista in tqdm(artistas, desc="Creando perfiles de artistas"):
    # Asignar géneros principales y secundarios de manera realista
    n_generos = random.randint(1, 3)  # La mayoría de artistas tienen 1-3 géneros
    generos_artista = random.sample(generos, n_generos)
    genero_principal = generos_artista[0]
    
    # Mejor categorización de BPM por género
    if genero_principal in ['Electrónica', 'House', 'Techno', 'Dance']:
        bpm_min, bpm_max = 120, 160
    elif genero_principal in ['Trap', 'Hip Hop', 'Drill', 'Rap']:
        bpm_min, bpm_max = 70, 140
    elif genero_principal in ['Pop', 'K-Pop', 'Pop Latino']:
        bpm_min, bpm_max = 95, 125
    elif genero_principal in ['Reggaeton', 'Urbano', 'Latino']:
        bpm_min, bpm_max = 88, 105
    elif genero_principal in ['Rock', 'Metal', 'Punk']:
        bpm_min, bpm_max = 110, 180
    elif genero_principal in ['R&B', 'Soul', 'Jazz']:
        bpm_min, bpm_max = 75, 100
    else:
        bpm_min, bpm_max = 80, 130
    
    # Añadir ligera variación al rango de BPM
    bpm_min = max(60, bpm_min + random.randint(-5, 5))
    bpm_max = min(180, bpm_max + random.randint(-5, 5))
    
    # Características de estilo vocal por artista
    estilos_vocales = ['Falsete', 'Grave', 'Medio', 'Alto', 'Rap', 'Melódico', 
                      'Susurro', 'Potente', 'Ronco', 'Suave', 'Agudo', 'Nasal']
    estilo_vocal = random.choice(estilos_vocales)
    
    # Duración típica de canciones
    duracion_min = random.randint(120, 200)  # 2-3.3 minutos mínimo
    duracion_max = random.randint(duracion_min + 30, 300)  # Hasta 5 minutos máximo
    
    # Probabilidad de colaboración realista
    if genero_principal in ['Reggaeton', 'Urbano', 'Latino', 'Hip Hop', 'Trap', 'Pop Latino']:
        prob_colaboracion = random.uniform(0.3, 0.6)  # Géneros con más colaboraciones
    else:
        prob_colaboracion = random.uniform(0.1, 0.4)
    
    # Seguidores base con distribución más realista
    # Usar distribución log-normal para modelar la popularidad inicial
    seguidores_base = np.random.lognormal(mean=1.0, sigma=1.0)
    seguidores_base = max(0.1, min(10, seguidores_base))  # Limitar outliers extremos
    
    # Tasa de crecimiento anual realista
    crecimiento_anual = random.uniform(1.05, 1.4)  # 5-40% de crecimiento anual
    
    # Escalas musicales preferidas
    escalas_preferidas = random.sample(escalas, random.randint(3, 6))
    
    # Probabilidad de lanzar single vs álbum (más singles en años recientes)
    prob_single = random.uniform(0.5, 0.8)
    
    # Factor de experimentación (qué tanto varía su estilo)
    factor_experimentacion = random.uniform(0.1, 0.3)
    
    # Tendencia de popularidad del artista a lo largo del tiempo
    tendencias = ["ascendente", "estable", "fluctuante", "descendente"]
    pesos = [0.4, 0.3, 0.2, 0.1]  # La mayoría tienen tendencia ascendente o estable
    tendencia_temporal = random.choices(tendencias, weights=pesos, k=1)[0]
    
    # Probabilidad de hit viral (bajo pero posible)
    prob_hit_viral = random.uniform(0.03, 0.1)
    
    # Factor de consistencia/inconsistencia en la calidad
    factor_inconsistencia = random.uniform(0.1, 0.25)
    
    # Popularidad base del artista (0-100)
    # Esto determinará el "punto de partida" para la popularidad de sus canciones
    popularidad_base = random.uniform(20, 80)
    
    # Almacenar perfil del artista
    perfiles_artistas[artista] = {
        'generos': generos_artista,
        'genero_principal': genero_principal,
        'bpm_range': (bpm_min, bpm_max),
        'estilo_vocal': estilo_vocal,
        'duracion_range': (duracion_min, duracion_max),
        'prob_colaboracion': prob_colaboracion,
        'seguidores_base': seguidores_base,
        'crecimiento_anual': crecimiento_anual,
        'escalas_preferidas': escalas_preferidas,
        'prob_single': prob_single,
        'factor_experimentacion': factor_experimentacion,
        'tendencia_temporal': tendencia_temporal,
        'prob_hit_viral': prob_hit_viral,
        'factor_inconsistencia': factor_inconsistencia,
        'popularidad_base': popularidad_base
    }

def generar_dataset_canciones(artistas, perfiles_artistas, n_canciones_por_artista=80, 
                              fecha_inicio='2010-01-01', fecha_fin='2023-12-31'):
    """
    Genera un dataset sintético de canciones con distribución de popularidad balanceada
    """
    fecha_inicio = datetime.strptime(fecha_inicio, '%Y-%m-%d')
    fecha_fin = datetime.strptime(fecha_fin, '%Y-%m-%d')
    rango_dias = (fecha_fin - fecha_inicio).days
    
    datos = []
    
    for artista in tqdm(artistas, desc="Generando canciones"):
        perfil = perfiles_artistas[artista]
        
        # Determinar número de canciones con variabilidad natural
        factor_productividad = np.random.normal(1.0, 0.3)
        n_canciones = max(30, min(200, int(n_canciones_por_artista * factor_productividad)))
        
        # Seguidores iniciales
        seguidores = perfil['seguidores_base']
        
        # Fechas de lanzamiento realistas
        fechas_lanzamiento = []
        ultima_fecha = fecha_inicio + timedelta(days=random.randint(0, 365))
        
        # Generar las fechas de lanzamiento primero
        for _ in range(n_canciones):
            # Tiempo entre lanzamientos varía según el tipo de artista
            if perfil['genero_principal'] in ['Pop', 'Hip Hop', 'Reggaeton', 'Trap']:
                dias_hasta_siguiente = random.randint(30, 180)  # Artistas más activos
            else:
                dias_hasta_siguiente = random.randint(60, 240)  # Artistas menos frecuentes
            
            # Los artistas tienden a lanzar más música en ciertos períodos
            if random.random() < 0.2:
                dias_hasta_siguiente = random.randint(10, 40)  # Lanzamientos agrupados
            
            ultima_fecha += timedelta(days=dias_hasta_siguiente)
            
            if ultima_fecha > fecha_fin:
                break
                
            fechas_lanzamiento.append(ultima_fecha)
        
        fechas_lanzamiento.sort()
        
        # Factor de popularidad inicial del artista
        popularidad_artista = perfil['popularidad_base']
        
        # Generar canciones para este artista
        for idx, fecha_lanzamiento in enumerate(fechas_lanzamiento):
            year_idx = min(13, max(0, fecha_lanzamiento.year - 2010))
            
            # Actualizar popularidad del artista con el tiempo
            if idx > 0:
                # Eventos aleatorios (álbum exitoso, controversia, etc.)
                evento_aleatorio = random.random() < 0.1
                efecto_evento = random.uniform(0.7, 1.3) if evento_aleatorio else 1.0
                
                # Ajustar popularidad según la tendencia temporal del artista
                if perfil['tendencia_temporal'] == "ascendente":
                    popularidad_artista += random.uniform(0.2, 1.2) * efecto_evento
                elif perfil['tendencia_temporal'] == "descendente":
                    popularidad_artista -= random.uniform(0.1, 0.8) * efecto_evento
                elif perfil['tendencia_temporal'] == "fluctuante":
                    popularidad_artista += random.uniform(-1.0, 1.0) * efecto_evento
                else:  # estable
                    popularidad_artista += random.uniform(-0.3, 0.3) * efecto_evento
                
                # Mantener la popularidad dentro de límites razonables
                popularidad_artista = max(10, min(90, popularidad_artista))
            
            # Determinar estación del año
            mes = fecha_lanzamiento.month
            if 3 <= mes <= 5:
                estacion = 'Primavera'
            elif 6 <= mes <= 8:
                estacion = 'Verano'
            elif 9 <= mes <= 11:
                estacion = 'Otoño'
            else:
                estacion = 'Invierno'
            
            # Seleccionar género para esta canción
            if random.random() < perfil['factor_experimentacion']:
                # El artista experimenta con un género fuera de sus habituales
                genero = random.choice([g for g in generos if g not in perfil['generos']])
            else:
                # El artista usa uno de sus géneros habituales, con preferencia al principal
                if len(perfil['generos']) > 1 and random.random() < 0.3:
                    generos_sec = [g for g in perfil['generos'] if g != perfil['genero_principal']]
                    genero = random.choice(generos_sec)
                else:
                    genero = perfil['genero_principal']
            
            # BPM específico para esta canción
            bpm = random.randint(perfil['bpm_range'][0], perfil['bpm_range'][1])
            
            # Duración específica para esta canción
            duracion_seg = random.randint(perfil['duracion_range'][0], perfil['duracion_range'][1])
            
            # Clave musical (escala)
            if random.random() < 0.7:
                escala = random.choice(perfil['escalas_preferidas'])
            else:
                escala = random.choice(escalas)
            clave_musical = escalas.index(escala)  # 0-11 representando C a B
            
            # Modo: 0 = menor, 1 = mayor
            if genero in ['Metal', 'Rock', 'Trap', 'Hip Hop', 'Rap']:
                # Géneros que tienden a usar más modo menor
                modo = 0 if random.random() < 0.7 else 1
            elif genero in ['Pop', 'Dance', 'K-Pop', 'Pop Latino']:
                # Géneros que tienden a usar más modo mayor
                modo = 1 if random.random() < 0.6 else 0
            else:
                # Otros géneros más equilibrados
                modo = random.randint(0, 1)
            
            # Instrumental (0 = con voz, 1 = instrumental)
            if genero in ['Electrónica', 'House', 'Techno', 'Lofi', 'Synthwave']:
                # Géneros más propensos a canciones instrumentales
                instrumental = 1 if random.random() < 0.4 else 0
            else:
                # La mayoría de canciones tienen voz
                instrumental = 1 if random.random() < 0.1 else 0
            
            # Variable complementaria a instrumental
            con_voz = 1 - instrumental
            
            # ¿Es una colaboración?
            if random.random() < perfil['prob_colaboracion']:
                colaboracion = 1
                # Seleccionar artista colaborador
                if random.random() < 0.5:
                    # Colaboración con artista del mismo género
                    artistas_mismo_genero = [a for a in artistas 
                                             if a != artista 
                                             and genero in perfiles_artistas[a]['generos']]
                    if artistas_mismo_genero:
                        artista_colaborador = random.choice(artistas_mismo_genero)
                    else:
                        artista_colaborador = random.choice([a for a in artistas if a != artista])
                else:
                    # Colaboración con cualquier otro artista
                    artista_colaborador = random.choice([a for a in artistas if a != artista])
                
                # El factor de popularidad del colaborador influye
                colab_popularidad = perfiles_artistas[artista_colaborador]['popularidad_base']
                factor_colaboracion = 1.0 + min(0.3, (colab_popularidad / 100) * 0.3)
            else:
                colaboracion = 0
                artista_colaborador = None
                factor_colaboracion = 1.0
            
            # Tipo de lanzamiento (single o álbum)
            if random.random() < perfil['prob_single']:
                tipo_lanzamiento = 'Single'
            else:
                tipo_lanzamiento = 'Album'
            
            # Factor de novedad 
            # Los artistas nuevos tienen una desventaja inicial que se reduce con el tiempo
            anio_debut = fecha_inicio.year + int(artistas.index(artista) / len(artistas) * 6)
            anios_carrera = fecha_lanzamiento.year - anio_debut
            factor_novedad = min(1.0, 0.7 + 0.06 * anios_carrera)
            
            # Factor de estacionalidad
            factores_estacionales = {
                'Primavera': 1.0,
                'Verano': 1.2,  # Verano más popular
                'Otoño': 0.9,
                'Invierno': 0.85
            }
            factor_estacional = factores_estacionales[estacion]
            
            # En diciembre hay más atención a música navideña
            if fecha_lanzamiento.month == 12:
                factor_estacional *= 1.15
            
            # Tendencia del género en ese año
            tendencia_genero = tendencias_genero[genero][year_idx]
            
            # Factor de energía (influye en la popularidad)
            energia = min(1.0, max(0.1, (bpm - 60) / 120))
            
            # Determinar si es un hit viral
            es_hit_viral = random.random() < perfil['prob_hit_viral']
            factor_viral = random.uniform(1.5, 2.2) if es_hit_viral else 1.0
            
            # Calidad intrínseca de la canción (varía según la consistencia del artista)
            calidad_base = random.uniform(0.3, 0.9)  # Base de calidad aleatoria
            # Artistas inconsistentes tienen mayor variabilidad
            variacion_calidad = perfil['factor_inconsistencia'] * random.uniform(-0.3, 0.3)
            calidad = max(0.2, min(1.0, calidad_base + variacion_calidad))
            
            # Calcular popularidad base
            # Ponderación de los diferentes factores
            popularidad_base = (
                (popularidad_artista * 0.4) +  # 40% Popularidad del artista
                (calidad * 40) +               # 40% Calidad intrínseca
                (energia * 5) +                # 5% Energía/BPM
                (tendencia_genero * 10)        # 10% Tendencia del género
            )
            
            # Ajustar por factores multiplicativos
            popularidad = popularidad_base * factor_novedad * factor_estacional * factor_colaboracion * factor_viral
            
            # Añadir algo de ruido aleatorio (±15%)
            popularidad *= random.uniform(0.85, 1.15)
            
            # Asegurar que la popularidad esté en el rango [1, 100]
            popularidad = max(1, min(100, popularidad))
            
            # Clasificar la popularidad (útil para entrenar modelos de clasificación)
            if popularidad < 30:
                categoria_popularidad = "Baja"
                exito = 0
            elif popularidad < 70:
                categoria_popularidad = "Media"
                exito = 1 if popularidad >= 50 else 0
            else:
                categoria_popularidad = "Alta"
                exito = 1
            
            # Crear registro de canción
            cancion = {
                'artista': artista,
                'genero': genero,
                'bpm': bpm,
                'duracion_seg': duracion_seg,
                'clave_musical': clave_musical,
                'modo': modo,
                'instrumental': instrumental,
                'con_voz': con_voz,
                'colaboracion': colaboracion,
                'artista_colaborador': artista_colaborador if colaboracion else "",
                'fecha_lanzamiento': fecha_lanzamiento,
                'año': fecha_lanzamiento.year,
                'mes': fecha_lanzamiento.month,
                'dia_semana': fecha_lanzamiento.weekday(),
                'estacion': estacion,
                'tipo_lanzamiento': tipo_lanzamiento,
                'calidad_intrinseca': round(calidad * 100),  # Variable latente (no observable en la realidad)
                'popularidad': round(popularidad, 2),  # Variable objetivo para regresión
                'categoria_popularidad': categoria_popularidad,  # Variable para clasificación
                'exito': exito,  # Variable binaria (0 = no exitosa, 1 = exitosa)
                'hit_viral': 1 if es_hit_viral else 0
            }
            
            datos.append(cancion)
    
    # Convertir a DataFrame
    df = pd.DataFrame(datos)
    
    # Ordenar por fecha de lanzamiento
    df = df.sort_values(['fecha_lanzamiento', 'artista'])
    
    # Renumerar el índice
    df = df.reset_index(drop=True)
    
    return df

def analizar_dataset(df):
    """
    Realiza análisis básico del dataset y genera visualizaciones
    """
    print("\nAnálisis del dataset generado:")
    print(f"Total de canciones: {len(df)}")
    print(f"Periodo: {df['fecha_lanzamiento'].min().date()} a {df['fecha_lanzamiento'].max().date()}")
    print(f"Número de artistas: {df['artista'].nunique()}")
    print(f"Número de géneros: {df['genero'].nunique()}")
    
    # Estadísticas de popularidad
    print("\nEstadísticas de popularidad:")
    print(df['popularidad'].describe())
    
    # Distribución por categoría
    print("\nDistribución por categoría de popularidad:")
    print(df['categoria_popularidad'].value_counts(normalize=True).round(3) * 100)
    
    # Distribución por éxito
    print("\nProporción éxito/fracaso:")
    print(df['exito'].value_counts(normalize=True).round(3) * 100)
    
    # Visualizaciones
    plt.figure(figsize=(12, 10))
    
    # 1. Distribución de popularidad
    plt.subplot(2, 2, 1)
    sns.histplot(df['popularidad'], kde=True, bins=30)
    plt.title('Distribución de Popularidad')
    plt.xlabel('Popularidad')
    plt.ylabel('Frecuencia')
    
    # 2. Distribución por categoría
    plt.subplot(2, 2, 2)
    sns.countplot(x='categoria_popularidad', data=df, order=['Baja', 'Media', 'Alta'])
    plt.title('Distribución por Categoría')
    plt.xlabel('Categoría de Popularidad')
    plt.ylabel('Conteo')
    
    # 3. Popularidad promedio por género (top 10)
    plt.subplot(2, 2, 3)
    top_genres = df.groupby('genero')['popularidad'].mean().sort_values(ascending=False).head(10)
    sns.barplot(x=top_genres.values, y=top_genres.index)
    plt.title('Popularidad Promedio por Género (Top 10)')
    plt.xlabel('Popularidad Promedio')
    
    # 4. Evolución de la popularidad en el tiempo
    plt.subplot(2, 2, 4)
    df_tiempo = df.groupby('año')['popularidad'].mean().reset_index()
    sns.lineplot(x='año', y='popularidad', data=df_tiempo, marker='o')
    plt.title('Evolución de Popularidad en el Tiempo')
    plt.xlabel('Año')
    plt.ylabel('Popularidad Promedio')
    
    plt.tight_layout()
    
    # Creamos la carpeta si no existe
    output_dir = 'dataset_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Guardar el gráfico
    plt.savefig(os.path.join(output_dir, 'analisis_dataset.png'), dpi=300)
    print(f"\nGráfico guardado en {output_dir}/analisis_dataset.png")
    
    # Mostrar gráfico
    plt.show()
    
    return

# Crear directorio para outputs si no existe
output_dir = 'dataset_output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Generar el dataset con distribución balanceada
print("\nGenerando dataset de canciones con distribución balanceada...")
df_canciones = generar_dataset_canciones(artistas, perfiles_artistas, n_canciones_por_artista=100)

# Analizar dataset
analizar_dataset(df_canciones)

# Guardar dataset
ruta_dataset = os.path.join(output_dir, 'dataset_canciones_mejorado.csv')
df_canciones.to_csv(ruta_dataset, index=False)
print(f"\nDataset guardado como '{ruta_dataset}'")

# Tiempo de ejecución
tiempo_total = time.time() - start_time
print(f"\nTiempo total de ejecución: {tiempo_total:.2f} segundos ({tiempo_total/60:.2f} minutos)")

print("\nProceso de generación de dataset completado exitosamente.")