import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import os
from tqdm import tqdm
import time

print("Iniciando generación de dataset ampliado para entrenamiento de redes neuronales...")
start_time = time.time()

# Configuración de semillas para reproducibilidad
np.random.seed(42)
random.seed(42)

# Aumentar la cantidad de artistas incorporando más nombres populares (2010-2023)
artistas_principales = [
    "Bad Bunny", "The Weeknd", "BTS", "Drake", "Taylor Swift", 
    "Billie Eilish", "Post Malone", "Dua Lipa", "Ed Sheeran", "Ariana Grande",
    "Justin Bieber", "Travis Scott", "Juice WRLD", "Harry Styles", "J Balvin",
    "Cardi B", "Olivia Rodrigo", "Lil Nas X", "Doja Cat", "DaBaby",
    "Shawn Mendes", "Megan Thee Stallion", "Daddy Yankee", "Imagine Dragons", "Blackpink",
    "Camila Cabello", "Lil Uzi Vert", "Maluma", "Coldplay", "Kendrick Lamar",
    "Lady Gaga", "Rosalía", "Lil Baby", "Karol G", "Bruno Mars"
]

artistas_adicionales = [
    "The Kid LAROI", "SZA", "Olivia Rodrigo", "Jack Harlow", "Rauw Alejandro",
    "Beyoncé", "Eminem", "Rihanna", "Adele", "Kanye West", "Miley Cyrus",
    "Twenty One Pilots", "Halsey", "Selena Gomez", "Charlie Puth", "Anuel AA",
    "Peso Pluma", "Feid", "Anitta", "Katy Perry", "Maroon 5", "Shakira",
    "Ozuna", "Nicki Minaj", "Cardi B", "Young Thug", "Migos", "Jhayco",
    "Calvin Harris", "David Guetta", "Martin Garrix", "Alan Walker",
    "Lewis Capaldi", "Sam Smith", "Demi Lovato", "Zara Larsson", "Tini",
    "María Becerra", "Paulo Londra", "Manuel Turizo", "Sebastian Yatra",
    "Myke Towers", "Don Omar", "Wisin & Yandel", "Nicky Jam", "Farruko"
]

# Combinar y eliminar duplicados
artistas = list(set(artistas_principales + artistas_adicionales))
print(f"Total de artistas: {len(artistas)}")

# Géneros musicales expandidos
generos = [
    'Pop', 'Hip Hop', 'R&B', 'Reggaeton', 'K-Pop', 'Rock', 'Electrónica', 
    'Country', 'Trap', 'Indie', 'Soul', 'Jazz', 'Metal', 'Folk', 'Clásica',
    'Latino', 'Dance', 'House', 'Drill', 'Funk', 'Disco', 'Flamenco', 
    'Rap', 'Punk', 'Blues', 'Alternativo', 'Techno', 'Pop Latino',
    'Urbano', 'Experimental', 'Lofi', 'Synthwave', 'Tropical'
]

# Escalas musicales
escalas = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Estaciones del año (para simular efectos estacionales)
estaciones = ['Primavera', 'Verano', 'Otoño', 'Invierno']

# Tendencias de popularidad por género ampliadas pero con MAYOR VARIABILIDAD
tendencias_genero = {}
for genero in generos:
    # Crear tendencia con más variabilidad y aleatoriedad
    base = random.uniform(0.4, 0.7)
    tendencia = []
    for i in range(14):  # 2010-2023
        # Añadir más fluctuación aleatoria para evitar patrones reconocibles
        cambio = random.uniform(-0.15, 0.15)
        # Suma de componentes aleatorias para simular cambios no predecibles
        nuevo_valor = base + cambio + 0.02 * i * random.uniform(-1, 1)
        tendencia.append(max(0.3, min(1.0, nuevo_valor)))  # Limitar entre 0.3 y 1.0
    tendencias_genero[genero] = tendencia

print("Generando perfiles para artistas...")
# Asignar características específicas a artistas
perfiles_artistas = {}
for artista in tqdm(artistas, desc="Creando perfiles de artistas"):
    # Asignar géneros principales y secundarios
    n_generos = random.randint(1, 4)  # Cada artista puede tener hasta 4 géneros
    generos_artista = random.sample(generos, n_generos)
    genero_principal = generos_artista[0]
    
    # Mejor categorización de BPM por género CON MÁS RUIDO
    if genero_principal in ['Electrónica', 'House', 'Techno', 'Dance']:
        bpm_min, bpm_max = 120, 180
    elif genero_principal in ['Trap', 'Hip Hop', 'Drill', 'Rap']:
        bpm_min, bpm_max = 60, 160
    elif genero_principal in ['Pop', 'K-Pop', 'Pop Latino']:
        bpm_min, bpm_max = 90, 130
    elif genero_principal in ['Reggaeton', 'Urbano', 'Latino']:
        bpm_min, bpm_max = 85, 110
    elif genero_principal in ['Rock', 'Metal', 'Punk']:
        bpm_min, bpm_max = 100, 200
    elif genero_principal in ['R&B', 'Soul', 'Jazz']:
        bpm_min, bpm_max = 70, 110
    else:
        bpm_min, bpm_max = 70, 140
        
    # AÑADIR RUIDO A LOS RANGOS DE BPM para evitar patrones claros
    bpm_min = max(60, bpm_min + random.randint(-10, 10))
    bpm_max = min(200, bpm_max + random.randint(-15, 15))
    
    # Características personalizadas por artista (más variadas)
    estilos_vocales = ['Falsete', 'Grave', 'Medio', 'Alto', 'Rap', 'Melódico', 
                      'Susurro', 'Potente', 'Ronco', 'Suave', 'Agudo', 'Nasal']
    estilo_vocal = random.choice(estilos_vocales)
    
    # Más variabilidad en duración
    duracion_min = random.randint(90, 240)
    duracion_max = random.randint(duracion_min, 500)
    
    # Probabilidad de colaboración más realista y con MÁS VARIACIÓN
    prob_colaboracion = random.uniform(0.1, 0.7)
    
    # Distribución más realista de seguidores (distribución log-normal con MÁS VARIANZA)
    seguidores_base = max(1, np.random.lognormal(2, 1.5))  # Aumentar varianza
    
    # Crecimiento anual variable (algunos crecen rápido, otros más lento)
    crecimiento_anual = random.uniform(1.05, 1.8) * random.uniform(0.8, 1.2)
    
    # Escalas musicales preferidas
    escalas_preferidas = random.sample(escalas, random.randint(3, len(escalas)))
    
    # Probabilidad de lanzar single vs álbum
    prob_single = random.uniform(0.4, 0.9) * random.uniform(0.9, 1.1)
    
    # Factor de experimentación 
    factor_experimentacion = random.uniform(0.05, 0.4) * random.uniform(0.7, 1.3)
    
    # Tendencia temporal más IMPREDECIBLE
    tendencia_temporal = random.choice(["ascendente", "estable", "descendente", "fluctuante", "inestable"])
    
    # Probabilidad de hit viral 
    prob_hit_viral = random.uniform(0.01, 0.15) * random.uniform(0.6, 1.5)
    
    # Factor de inconsistencia
    factor_inconsistencia = random.uniform(0.05, 0.3)
    
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
        'factor_inconsistencia': factor_inconsistencia
    }

def generar_dataset_canciones(artistas, perfiles_artistas, n_canciones_por_artista=100, 
                              fecha_inicio='2010-01-01', fecha_fin='2023-12-31'):
    """
    Genera un dataset sintético ampliado de canciones
    """
    fecha_inicio = datetime.strptime(fecha_inicio, '%Y-%m-%d')
    fecha_fin = datetime.strptime(fecha_fin, '%Y-%m-%d')
    rango_dias = (fecha_fin - fecha_inicio).days
    
    datos = []
    
    for artista in tqdm(artistas, desc="Generando canciones"):
        perfil = perfiles_artistas[artista]
        
        # Determinar número de canciones con variabilidad
        factor_productividad = np.random.lognormal(0, 0.8)
        n_canciones = int(n_canciones_por_artista * factor_productividad)
        n_canciones = max(30, min(300, n_canciones))
        
        # Seguidores base que crecen con el tiempo
        seguidores = perfil['seguidores_base']
        
        # Factor de popularidad inicial del artista
        factor_popularidad_artista = random.uniform(0.2, 1.0) * random.uniform(0.8, 1.2)
        
        # Fechas de lanzamiento
        fechas_lanzamiento = []
        ultima_fecha = fecha_inicio + timedelta(days=random.randint(0, 365))
        
        for _ in range(n_canciones):
            dias_hasta_siguiente = random.randint(20, 250)
            
            if random.random() < 0.2:
                dias_hasta_siguiente = random.randint(7, 30)
            
            ultima_fecha += timedelta(days=dias_hasta_siguiente)
            
            if ultima_fecha > fecha_fin:
                break
                
            fechas_lanzamiento.append(ultima_fecha)
        
        fechas_lanzamiento.sort()
        
        for idx, fecha_lanzamiento in enumerate(fechas_lanzamiento):
            year_idx = min(13, max(0, fecha_lanzamiento.year - 2010))
            
            # Actualizar seguidores con irregularidades aleatorias
            if idx > 0:
                evento_aleatorio = random.random() < 0.15
                efecto_evento = random.uniform(0.7, 1.4) if evento_aleatorio else 1.0
                
                if perfil['tendencia_temporal'] == "ascendente":
                    seguidores *= perfil['crecimiento_anual'] * random.uniform(0.9, 1.15) * efecto_evento
                elif perfil['tendencia_temporal'] == "descendente":
                    seguidores *= min(1.0, perfil['crecimiento_anual'] * random.uniform(0.75, 0.95)) * efecto_evento
                elif perfil['tendencia_temporal'] == "fluctuante":
                    seguidores *= perfil['crecimiento_anual'] * random.uniform(0.85, 1.15) * efecto_evento
                elif perfil['tendencia_temporal'] == "inestable":
                    seguidores *= perfil['crecimiento_anual'] * random.uniform(0.6, 1.4) * efecto_evento
                else:  # estable
                    seguidores *= perfil['crecimiento_anual'] * random.uniform(0.95, 1.05) * efecto_evento
            
            # Determinar estación
            mes = fecha_lanzamiento.month
            if 3 <= mes <= 5:
                estacion = 'Primavera'
            elif 6 <= mes <= 8:
                estacion = 'Verano'
            elif 9 <= mes <= 11:
                estacion = 'Otoño'
            else:
                estacion = 'Invierno'
            
            # Seleccionar género con experimentación impredecible
            prob_experimentacion = perfil['factor_experimentacion'] * random.uniform(0.6, 1.4)
            if random.random() < min(0.75, prob_experimentacion):
                if random.random() < 0.6 and len(perfil['generos']) > 1:
                    generos_alternativos = [g for g in perfil['generos'] if g != perfil['genero_principal']]
                    genero = random.choice(generos_alternativos)
                else:
                    genero = random.choice(generos)
            else:
                genero = perfil['genero_principal']
            
            # BPM con menor predictibilidad
            bpm_base = random.randint(perfil['bpm_range'][0], perfil['bpm_range'][1])
            bpm = max(60, min(200, int(bpm_base + random.randint(-20, 20))))
            
            # Duración con más varianza
            factor_duracion = random.uniform(0.7, 1.3)
            duracion_seg = random.randint(
                int(perfil['duracion_range'][0] * factor_duracion),
                int(perfil['duracion_range'][1] * factor_duracion)
            )
            
            # Clave musical con menos predictibilidad
            if random.random() < 0.6:
                escala = random.choice(perfil['escalas_preferidas'])
            else:
                escala = random.choice(escalas)
            clave_musical = escalas.index(escala)
            
            # Modo (mayor o menor)
            if random.random() < 0.3:
                modo = random.randint(0, 1)
            elif genero in ['Trap', 'Hip Hop', 'Drill', 'R&B', 'Reggaeton'] and year_idx > 5:
                modo = 0 if random.random() < 0.6 else 1
            elif genero in ['Pop', 'Dance', 'Electrónica']:
                modo = 1 if random.random() < 0.55 else 0
            else:
                modo = random.randint(0, 1)
            
            # Instrumental
            if random.random() < 0.2:
                instrumental = random.randint(0, 1)
            elif genero in ['Electrónica', 'House', 'Techno', 'Lofi']:
                instrumental = 1 if random.random() < 0.25 else 0
            else:
                instrumental = 1 if random.random() < 0.15 else 0
            
            # Con voz (complemento de instrumental)
            con_voz = 0 if instrumental else 1
            
            # Colaboración
            prob_colab_base = perfil['prob_colaboracion'] * random.uniform(0.8, 1.2)
            
            if random.random() < 0.25:
                colaboracion = random.randint(0, 1)
            else:
                prob_colab_ajustada = prob_colab_base * (1 + year_idx * 0.03 * random.uniform(0.5, 1.5))
                
                if genero in ['Reggaeton', 'Trap', 'Hip Hop', 'Pop Latino', 'Urbano']:
                    prob_colab_ajustada *= random.uniform(1.2, 1.7)
                    
                colaboracion = 1 if random.random() < min(0.9, prob_colab_ajustada) else 0
            
            # Artista colaborador
            if colaboracion:
                if random.random() < 0.3:
                    artista_colaborador = random.choice([a for a in artistas if a != artista])
                else:
                    artistas_mismo_genero = [a for a in artistas if a != artista and 
                                           any(g == genero for g in perfiles_artistas[a]['generos'])]
                    
                    if artistas_mismo_genero and random.random() < 0.6:
                        artista_colaborador = random.choice(artistas_mismo_genero)
                    else:
                        artista_colaborador = random.choice([a for a in artistas if a != artista])
                
                # Factor de popularidad
                colab_popularidad = perfiles_artistas[artista_colaborador]['seguidores_base']
                factor_colaboracion = 1.0 + min(0.5, (colab_popularidad / (seguidores + 1)) * random.uniform(0.5, 1.5))
            else:
                artista_colaborador = None
                factor_colaboracion = 1.0
            
            # Tipo de lanzamiento
            if random.random() < 0.3:
                tipo_lanzamiento = random.choice(['Single', 'Album'])
            else:
                prob_single_ajustada = min(0.9, perfil['prob_single'] + year_idx * 0.01 * random.uniform(0.5, 1.5))
                tipo_lanzamiento = 'Single' if random.random() < prob_single_ajustada else 'Album'
            
            # Factor de novedad
            dias_desde_inicio = (fecha_lanzamiento - fecha_inicio).days
            factor_novedad = 0.5 + 0.5 * (dias_desde_inicio / rango_dias) * random.uniform(0.7, 1.3)
            
            # Factor de estacionalidad
            factores_estacionales = {
                'Primavera': 1.0 + random.uniform(-0.2, 0.2),
                'Verano': 1.2 + random.uniform(-0.25, 0.25),
                'Otoño': 0.9 + random.uniform(-0.2, 0.2),
                'Invierno': 0.8 + random.uniform(-0.25, 0.25)
            }
            factor_estacional = factores_estacionales[estacion] * random.uniform(0.8, 1.2)
            
            # Ajuste adicional para diciembre
            if fecha_lanzamiento.month == 12:
                factor_estacional *= random.uniform(1.0, 1.3)
            
            # Tendencia del género en ese año
            try:
                tendencia_base = tendencias_genero[genero][year_idx]
                tendencia_genero = tendencia_base * random.uniform(0.7, 1.3)
            except (KeyError, IndexError):
                tendencia_genero = random.uniform(0.5, 0.9)
            
            # Factor de energía
            factor_energia = max(0.1, min(1.0, ((bpm - 60) / 140) * random.uniform(0.7, 1.3)))
            
            # Calcular popularidad base
            popularidad_base = seguidores * random.uniform(0.5, 1.5) * 1000
            
            # Hits virales ocasionales
            es_hit_viral = random.random() < (perfil['prob_hit_viral'] * random.uniform(0.7, 1.3))
            factor_viral = random.uniform(2.0, 3.5) if es_hit_viral else 1.0
            
            # Reproducciones previas
            canciones_artista_previas = [d for d in datos if d['artista'] == artista]
            if canciones_artista_previas and random.random() < 0.7:
                if random.random() < 0.3:
                    canciones_recientes = canciones_artista_previas[-3:]
                    reproducciones_previas = np.mean([c['popularidad'] for c in canciones_recientes]) * random.uniform(0.7, 1.3)
                else:
                    pesos = np.linspace(0.3, 1.0, len(canciones_artista_previas)) * np.random.uniform(0.6, 1.4, len(canciones_artista_previas))
                    reproducciones_previas = np.average([c['popularidad'] for c in canciones_artista_previas], weights=pesos)
                
                factor_consistencia = 0.7 + (reproducciones_previas / 100) * 0.3 * random.uniform(0.5, 1.5)
            else:
                reproducciones_previas = random.uniform(0, 30)
                factor_consistencia = random.uniform(0.8, 1.2)
            
            # Factor de inconsistencia
            if random.random() < perfil['factor_inconsistencia'] * random.uniform(0.8, 1.2):
                factor_inconsistencia = random.uniform(0.3, 2.5)
            else:
                factor_inconsistencia = 1.0
            
            # Ajustar popularidad
            popularidad = popularidad_base * factor_popularidad_artista * factor_novedad * \
                          factor_estacional * tendencia_genero * (1 + 0.2 * factor_energia) * \
                          factor_colaboracion * factor_viral * factor_consistencia * factor_inconsistencia
            
            # Añadir más ruido
            popularidad *= np.random.lognormal(0, 0.5)
            
            # Factores impredecibles
            if random.random() < 0.1:
                popularidad *= random.uniform(0.2, 5.0)
            
            # Normalizar popularidad
            popularidad = min(100, max(1, popularidad / 10000))
            
            # Añadir outliers ocasionales
            if random.random() < 0.05:
                if random.random() < 0.5:
                    popularidad = random.uniform(1, 10)
                else:
                    popularidad = random.uniform(80, 100)
            
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
                'artista_colaborador': artista_colaborador,
                'fecha_lanzamiento': fecha_lanzamiento,
                'año': fecha_lanzamiento.year,
                'mes': fecha_lanzamiento.month,
                'estacion': estacion,
                'tipo_lanzamiento': tipo_lanzamiento,
                'popularidad': popularidad,
                'reproducciones_previas': reproducciones_previas,
                'hit_viral': 1 if es_hit_viral else 0,
                'dia_semana': fecha_lanzamiento.weekday(),
                'trimestre': (fecha_lanzamiento.month-1)//3 + 1
            }
            
            datos.append(cancion)
    
    # Convertir a DataFrame
    df = pd.DataFrame(datos)
    
    # Ordenar por fecha de lanzamiento
    df = df.sort_values(['fecha_lanzamiento', 'artista'])
    
    # Renumerar el índice
    df = df.reset_index(drop=True)
    
    print(f"Dataset generado: {len(df):,} canciones para {len(artistas)} artistas")
    
    return df

# Crear directorio para outputs si no existe
output_dir = 'dataset_output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Generar el dataset con más canciones por artista
print("\nGenerando dataset de canciones con mayor aleatoriedad (anti-overfitting)...")
df_canciones = generar_dataset_canciones(artistas, perfiles_artistas, n_canciones_por_artista=150)

# Mostrar dimensiones del dataset
print(f"Dataset generado con {len(df_canciones):,} canciones")
print(f"Número de artistas: {df_canciones['artista'].nunique()}")
print(f"Número de géneros: {df_canciones['genero'].nunique()}")
print(f"Rango de fechas: {df_canciones['fecha_lanzamiento'].min().date()} a {df_canciones['fecha_lanzamiento'].max().date()}")

# Guardar dataset
ruta_dataset = os.path.join(output_dir, 'dataset_canciones_ampliado.csv')
df_canciones.to_csv(ruta_dataset, index=False)
print(f"\nDataset guardado como '{ruta_dataset}'")

# Tiempo de ejecución
tiempo_total = time.time() - start_time
print(f"\nTiempo total de ejecución: {tiempo_total:.2f} segundos ({tiempo_total/60:.2f} minutos)")

print("\nProceso de generación de dataset anti-overfitting completado correctamente.")