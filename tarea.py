import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# Configurar semilla para reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)

print("="*80)
print("PREDICCIÓN DE POPULARIDAD MUSICAL CON REDES NEURONALES RECURRENTES")
print("="*80)
print("\nTensorFlow versión:", tf.__version__)

# ==============================
# 1. CARGA DE DATOS
# ==============================

print("\n" + "="*50)
print("1. CARGA DE DATOS")
print("="*50)

# Buscar archivos de datos preprocesados
dataset_dir = 'dataset_output'
dataset_csv = os.path.join(dataset_dir, 'dataset_canciones_mejorado.csv')

try:
    # Cargar el dataset
    print(f"Cargando dataset desde {dataset_csv}...")
    df_canciones = pd.read_csv(dataset_csv, parse_dates=['fecha_lanzamiento'])
    print(f"Dataset cargado: {len(df_canciones)} canciones")
except Exception as e:
    print(f"ERROR al cargar el dataset: {e}")
    print("Asegúrate de ejecutar primero el script generarDataset.py")
    exit(1)

# ==============================
# 2. VISUALIZACIÓN EXPLORATORIA 
# ==============================

print("\n" + "="*50)
print("2. VISUALIZACIÓN EXPLORATORIA")
print("="*50)

# Configurar estilo de gráficos
plt.style.use('seaborn-v0_8')

# Estadísticas descriptivas
print("\nEstadísticas descriptivas de popularidad:")
print(df_canciones['popularidad'].describe())

# Gráficos de exploración
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Distribución de popularidad
sns.histplot(df_canciones['popularidad'], kde=True, ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title('Distribución de Popularidad de Canciones', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Popularidad')
axes[0, 0].set_ylabel('Frecuencia')
axes[0, 0].grid(True, alpha=0.3)

# 2. Top 10 artistas por popularidad media
top_artistas = df_canciones.groupby('artista')['popularidad'].mean().sort_values(ascending=False).head(10)
sns.barplot(y=top_artistas.index, x=top_artistas.values, palette='viridis', ax=axes[0, 1])
axes[0, 1].set_title('Top 10 Artistas por Popularidad Media', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Popularidad Media')
axes[0, 1].set_ylabel('Artista')
axes[0, 1].grid(True, alpha=0.3)

# 3. Evolución temporal de la popularidad
df_tiempo = df_canciones.groupby(pd.Grouper(key='fecha_lanzamiento', freq='3M'))['popularidad'].mean().reset_index()
axes[1, 0].plot(df_tiempo['fecha_lanzamiento'], df_tiempo['popularidad'], 'o-', linewidth=2, color='darkblue')
axes[1, 0].set_title('Evolución Temporal de Popularidad (Trimestral)', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Fecha')
axes[1, 0].set_ylabel('Popularidad Media')
axes[1, 0].grid(True, alpha=0.3)

# 4. Matriz de correlación de características numéricas
cols_numericas = ['popularidad', 'bpm', 'duracion_seg', 'clave_musical', 'modo', 'colaboracion', 
                  'año', 'mes', 'instrumental', 'hit_viral']
corr = df_canciones[cols_numericas].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=axes[1, 1])
axes[1, 1].set_title('Matriz de Correlación entre Características', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# ==============================
# 3. PREPARACIÓN DE LOS DATOS
# ==============================

print("\n" + "="*50)
print("3. PREPARACIÓN DE LOS DATOS")
print("="*50)

def crear_secuencias_por_artista(df, lookback=5):
    """
    Crea secuencias de canciones por artista para entrenamiento de RNN
    """
    print(f"Creando secuencias con ventana temporal de {lookback} canciones...")
    
    # Seleccionar características relevantes
    caracteristicas = [
        'popularidad', 'bpm', 'duracion_seg', 'clave_musical', 'modo', 
        'colaboracion', 'instrumental', 'hit_viral', 'año', 'mes'
    ]
    
    X = []  # Secuencias de entrada
    y = []  # Valores a predecir
    
    # Agrupar por artista
    for artista, canciones in df.groupby('artista'):
        if len(canciones) <= lookback:
            continue
        
        # Ordenar canciones por fecha
        canciones = canciones.sort_values('fecha_lanzamiento')
        
        # Crear secuencias para este artista
        for i in range(lookback, len(canciones)):
            # Secuencia de canciones anteriores
            secuencia = canciones.iloc[i-lookback:i][caracteristicas].values
            # Popularidad de la próxima canción
            siguiente_popularidad = canciones.iloc[i]['popularidad']
            
            X.append(secuencia)
            y.append(siguiente_popularidad)
    
    return np.array(X), np.array(y)

# Crear secuencias para entrenamiento
lookback = 5  # Usar 5 canciones anteriores para predecir la siguiente
X, y = crear_secuencias_por_artista(df_canciones, lookback)

print(f"Secuencias creadas: X shape: {X.shape}, y shape: {y.shape}")

# Dividir en conjuntos de entrenamiento, validación y prueba
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Conjunto de entrenamiento: {X_train.shape[0]} muestras")
print(f"Conjunto de validación: {X_val.shape[0]} muestras")
print(f"Conjunto de prueba: {X_test.shape[0]} muestras")

# Normalización de datos
# Escalar features (para cada feature a lo largo de todas las secuencias)
scaler = MinMaxScaler()
n_samples, n_timesteps, n_features = X_train.shape

# Reshape para normalizar (combinar muestras y timesteps)
X_train_reshaped = X_train.reshape(-1, n_features)
X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)

X_val_reshaped = X_val.reshape(-1, n_features) 
X_val_scaled = scaler.transform(X_val_reshaped).reshape(X_val.shape)

X_test_reshaped = X_test.reshape(-1, n_features)
X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)

# Normalizar target (popularidad)
y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).flatten()  
y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

print("Normalización de datos completada")

# ==============================
# 4. DEFINICIÓN DE MODELOS
# ==============================

print("\n" + "="*50)
print("4. DEFINICIÓN DE MODELOS")
print("="*50)

def crear_modelo_rnn(input_shape, units=64, dropout_rate=0.2):
    """Crea un modelo RNN simple (Vanilla RNN)"""
    model = Sequential([
        SimpleRNN(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        SimpleRNN(units//2),
        Dropout(dropout_rate),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def crear_modelo_lstm(input_shape, units=64, dropout_rate=0.2):
    """Crea un modelo LSTM"""
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(units//2),
        Dropout(dropout_rate),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def crear_modelo_gru(input_shape, units=64, dropout_rate=0.2):
    """Crea un modelo GRU"""
    model = Sequential([
        GRU(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        GRU(units//2),
        Dropout(dropout_rate),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

# Definir forma de entrada para los modelos
input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
print(f"Forma de entrada para los modelos: {input_shape}")

# Crear modelos
print("Creando modelos...")
modelos = {
    'RNN': crear_modelo_rnn(input_shape),
    'LSTM': crear_modelo_lstm(input_shape),
    'GRU': crear_modelo_gru(input_shape)
}

# Mostrar arquitecturas
for nombre, modelo in modelos.items():
    print(f"\nArquitectura del modelo {nombre}:")
    modelo.summary()

# ==============================
# 5. ENTRENAMIENTO DE MODELOS
# ==============================

print("\n" + "="*50)
print("5. ENTRENAMIENTO DE MODELOS")
print("="*50)

def entrenar_modelo(modelo, nombre, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """
    Entrena un modelo y retorna el historial de entrenamiento
    """
    print(f"\n{'='*60}")
    print(f"ENTRENANDO MODELO: {nombre}")
    print(f"{'='*60}")
    
    # Callbacks para entrenamiento
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Entrenar modelo
    historia = modelo.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )
    
    return historia

# Entrenar cada modelo
historias = {}
epochs = 50
batch_size = 32

for nombre, modelo in modelos.items():
    historias[nombre] = entrenar_modelo(
        modelo, nombre, 
        X_train_scaled, y_train_scaled, 
        X_val_scaled, y_val_scaled,
        epochs=epochs,
        batch_size=batch_size
    )

print("\nTodos los modelos han sido entrenados")

# ==============================
# 6. EVALUACIÓN DE MODELOS
# ==============================

print("\n" + "="*50)
print("6. EVALUACIÓN DE MODELOS")
print("="*50)

def evaluar_modelo(modelo, nombre, X_test, y_test):
    """
    Evalúa un modelo en el conjunto de prueba y calcula métricas de rendimiento
    """
    print(f"\nEvaluando modelo: {nombre}")
    
    # Hacer predicciones
    y_pred_scaled = modelo.predict(X_test, verbose=0)
    
    # Desnormalizar predicciones y valores reales
    y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()
    y_true = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Calcular métricas
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calcular MAPE con filtro para valores significativos
    mask = y_true > 5.0  # Evitar división por valores cercanos a cero
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape
    }
    
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    return metrics, y_pred, y_true

# Evaluar cada modelo
resultados = {}
predicciones = {}

for nombre, modelo in modelos.items():
    metrics, y_pred, y_true = evaluar_modelo(
        modelo, nombre, X_test_scaled, y_test_scaled
    )
    
    resultados[nombre] = metrics
    predicciones[nombre] = {'pred': y_pred, 'true': y_true}

# Crear tabla comparativa
resultados_df = pd.DataFrame(resultados).T
print("\nTABLA COMPARATIVA DE MÉTRICAS:")
print("="*80)
print(resultados_df.round(4))

# Identificar mejor modelo según diferentes métricas
mejor_rmse = resultados_df['RMSE'].idxmin()
mejor_mae = resultados_df['MAE'].idxmin()
mejor_r2 = resultados_df['R²'].idxmax()
mejor_mape = resultados_df['MAPE'].idxmin()

print("\n" + "="*50)
print("MEJOR MODELO POR MÉTRICA:")
print(f"Mejor modelo por RMSE: {mejor_rmse} ({resultados_df.loc[mejor_rmse, 'RMSE']:.4f})")
print(f"Mejor modelo por MAE: {mejor_mae} ({resultados_df.loc[mejor_mae, 'MAE']:.4f})")
print(f"Mejor modelo por R²: {mejor_r2} ({resultados_df.loc[mejor_r2, 'R²']:.4f})")
print(f"Mejor modelo por MAPE: {mejor_mape} ({resultados_df.loc[mejor_mape, 'MAPE']:.4f}%)")

# Seleccionar el mejor modelo general (basado en RMSE)
mejor_modelo_general = mejor_rmse
print(f"\nEl mejor modelo general es: {mejor_modelo_general}")

# ==============================
# 7. VISUALIZACIÓN DE RESULTADOS
# ==============================

print("\n" + "="*50)
print("7. VISUALIZACIÓN DE RESULTADOS")
print("="*50)

# Gráfico 1: Historial de entrenamiento
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Loss durante entrenamiento
axes[0,0].set_title('Pérdida Durante Entrenamiento', fontsize=14, fontweight='bold')
for name, history in historias.items():
    axes[0,0].plot(history.history['loss'], label=f'{name} - Train')
    axes[0,0].plot(history.history['val_loss'], label=f'{name} - Val', linestyle='--')
axes[0,0].set_xlabel('Época')
axes[0,0].set_ylabel('MSE Loss')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# MAE durante entrenamiento
axes[0,1].set_title('MAE Durante Entrenamiento', fontsize=14, fontweight='bold')
for name, history in historias.items():
    axes[0,1].plot(history.history['mae'], label=f'{name} - Train')
    axes[0,1].plot(history.history['val_mae'], label=f'{name} - Val', linestyle='--')
axes[0,1].set_xlabel('Época')
axes[0,1].set_ylabel('MAE')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Comparación de métricas
metrics_names = ['RMSE', 'MAE', 'MAPE']
x = np.arange(len(metrics_names))
width = 0.25

axes[1,0].set_title('Comparación de Métricas de Error', fontsize=14, fontweight='bold')
for i, (name, metrics) in enumerate(resultados.items()):
    values = [metrics['RMSE'], metrics['MAE'], metrics['MAPE'] / 10]  # Escalar MAPE para visualización
    axes[1,0].bar(x + i*width, values, width, label=name, alpha=0.8)

axes[1,0].set_xlabel('Métricas')
axes[1,0].set_ylabel('Valor')
axes[1,0].set_xticks(x + width)
axes[1,0].set_xticklabels(metrics_names)
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# R² Score
axes[1,1].set_title('R² Score (Coeficiente de Determinación)', fontsize=14, fontweight='bold')
r2_scores = [resultados[name]['R²'] for name in modelos.keys()]
bars = axes[1,1].bar(modelos.keys(), r2_scores, color=['skyblue', 'orange', 'lightgreen'], alpha=0.8)
axes[1,1].set_ylabel('R² Score')
axes[1,1].grid(True, alpha=0.3)

# Añadir valores en las barras
for bar, score in zip(bars, r2_scores):
    height = bar.get_height()
    axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# Gráfico 2: Predicciones vs Valores Reales
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Predicciones completas
axes[0,0].set_title('Predicciones vs Valores Reales (primeros 100 puntos)', fontsize=14, fontweight='bold')
num_samples = min(100, len(predicciones[list(predicciones.keys())[0]]['true']))
x_range = range(num_samples)
for name in modelos.keys():
    axes[0,0].plot(x_range, predicciones[name]['pred'][:num_samples], label=f'{name} - Predicción', alpha=0.8)
axes[0,0].plot(x_range, predicciones['RNN']['true'][:num_samples], label='Valor Real', color='black', linewidth=2)
axes[0,0].set_xlabel('Muestra')
axes[0,0].set_ylabel('Popularidad')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Scatter plots para cada modelo
colors = ['blue', 'orange', 'green']
for i, (name, color) in enumerate(zip(modelos.keys(), colors)):
    row = (i + 1) // 2
    col = (i + 1) % 2
    
    y_true = predicciones[name]['true']
    y_pred = predicciones[name]['pred']
    
    axes[row, col].scatter(y_true, y_pred, alpha=0.6, color=color, s=20)
    
    # Línea diagonal perfecta
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    axes[row, col].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    axes[row, col].set_title(f'{name}: Predicción vs Real', fontsize=14, fontweight='bold')
    axes[row, col].set_xlabel('Valor Real')
    axes[row, col].set_ylabel('Predicción')
    axes[row, col].grid(True, alpha=0.3)
    
    # Añadir R² en el gráfico
    r2 = resultados[name]['R²']
    axes[row, col].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[row, col].transAxes,
                       fontsize=12, verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.show()

# Gráfico 3: Análisis de residuos
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (name, color) in enumerate(zip(modelos.keys(), colors)):
    y_true = predicciones[name]['true']
    y_pred = predicciones[name]['pred']
    residuals = y_true - y_pred
    
    # Histograma de residuos
    axes[i].hist(residuals, bins=30, alpha=0.7, color=color, edgecolor='black')
    axes[i].set_title(f'Distribución de Residuos - {name}', fontsize=14, fontweight='bold')
    axes[i].set_xlabel('Residuo')
    axes[i].set_ylabel('Frecuencia')
    axes[i].grid(True, alpha=0.3)
    
    # Estadísticas de residuos
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    axes[i].axvline(mean_residual, color='red', linestyle='--', linewidth=2, label=f'Media: {mean_residual:.3f}')
    axes[i].axvline(mean_residual + std_residual, color='orange', linestyle=':', alpha=0.7, label=f'±1σ: {std_residual:.3f}')
    axes[i].axvline(mean_residual - std_residual, color='orange', linestyle=':', alpha=0.7)
    axes[i].legend()

plt.tight_layout()
plt.show()

# ==============================
# 8. ANÁLISIS COMPARATIVO FINAL
# ==============================

print("\n" + "="*80)
print("ANÁLISIS COMPARATIVO FINAL")
print("="*80)

print("\n1. RENDIMIENTO GENERAL:")
print("-" * 40)
print(f"   MEJOR MODELO (menor RMSE): {mejor_rmse}")
print(f"   RMSE: {resultados[mejor_rmse]['RMSE']:.4f}")
print(f"   R²: {resultados[mejor_rmse]['R²']:.4f}")

print("\n2. COMPARACIÓN POR MÉTRICA:")
print("-" * 40)
for metric in ['RMSE', 'MAE', 'MAPE', 'R²']:
    if metric == 'R²':
        best = max(resultados.keys(), key=lambda x: resultados[x][metric])
        print(f"{metric:8s}: {best:4s} ({resultados[best][metric]:7.4f})")
    else:
        best = min(resultados.keys(), key=lambda x: resultados[x][metric])
        print(f"{metric:8s}: {best:4s} ({resultados[best][metric]:7.4f})")

print("\n3. CARACTERÍSTICAS DE CADA MODELO:")
print("-" * 40)

print("RNN (Simple):")
print("  ✓ Más rápido de entrenar")
print("  ✗ Problemas con dependencias largas")
print("  ✗ Desvanecimiento del gradiente")

print("\nLSTM:")
print("  ✓ Maneja dependencias a largo plazo")
print("  ✓ Controla el flujo de información")
print("  ✗ Más parámetros (más lento)")

print("\nGRU:")
print("  ✓ Compromiso entre RNN y LSTM")
print("  ✓ Menos parámetros que LSTM")
print("  ✓ Rendimiento similar a LSTM")

print("\n4. NÚMERO DE PARÁMETROS:")
print("-" * 40)
for name, model in modelos.items():
    params = model.count_params()
    print(f"{name:4s}: {params:,} parámetros")

print("\n" + "="*80)
print(f"¡ANÁLISIS COMPLETADO! El modelo recomendado para predecir popularidad musical es: {mejor_modelo_general}")
print("="*80)