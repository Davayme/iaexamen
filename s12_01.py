
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
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

print("TensorFlow version:", tf.__version__)
print("Librerías importadas exitosamente")

def generate_synthetic_timeseries(n_samples=1000, noise_level=0.1):
    """
    Genera una serie temporal sintética con múltiples componentes:
    - Tendencia lineal
    - Componente estacional (múltiples frecuencias)
    - Componente cíclica
    - Ruido gaussiano
    
    Parámetros:
    - n_samples: número de puntos de datos
    - noise_level: nivel de ruido (0-1)
    
    Retorna:
    - DataFrame con la serie temporal
    """
    
    # Crear índice temporal
    time_index = np.arange(n_samples)
    
    # Componente de tendencia (crecimiento con cambios)
    trend = 0.02 * time_index + 0.0001 * time_index**1.5
    
    # Componente estacional múltiple
    seasonal_1 = 3 * np.sin(2 * np.pi * time_index / 50)  # Ciclo de 50 períodos
    seasonal_2 = 1.5 * np.cos(2 * np.pi * time_index / 20)  # Ciclo de 20 períodos
    seasonal_3 = 0.8 * np.sin(2 * np.pi * time_index / 7)   # Ciclo semanal
    
    # Componente cíclica de largo plazo
    cyclical = 2 * np.sin(2 * np.pi * time_index / 200) * np.exp(-time_index/800)
    
    # Componente no lineal
    nonlinear = 0.5 * np.sin(time_index/10) * np.cos(time_index/30)
    
    # Ruido gaussiano
    noise = np.random.normal(0, noise_level, n_samples)
    
    # Serie temporal final
    y = trend + seasonal_1 + seasonal_2 + seasonal_3 + cyclical + nonlinear + noise
    
    # Crear características adicionales (variables exógenas)
    # Simulan factores externos que podrían influir en la serie
    external_factor_1 = 0.5 * np.sin(2 * np.pi * time_index / 30) + np.random.normal(0, 0.1, n_samples)
    external_factor_2 = np.cumsum(np.random.normal(0, 0.05, n_samples))  # Random walk
    
    # Crear DataFrame
    data = pd.DataFrame({
        'timestamp': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
        'target': y,
        'external_1': external_factor_1,
        'external_2': external_factor_2,
        'day_of_week': np.tile(range(7), n_samples//7 + 1)[:n_samples],
        'month': pd.date_range('2020-01-01', periods=n_samples, freq='D').month
    })
    
    return data

# Generar datos sintéticos
print("Generando datos sintéticos...")
data = generate_synthetic_timeseries(n_samples=1000, noise_level=0.2)

# Guardar datos en CSV
data.to_csv('synthetic_timeseries_data.csv', index=False)
print("Datos guardados en 'synthetic_timeseries_data.csv'")

# Mostrar información básica
print(f"\nDimensiones de los datos: {data.shape}")
print("\nPrimeras 5 filas:")
print(data.head())

print("\nEstadísticas descriptivas:")
print(data.describe())

# ================================
# 3. VISUALIZACIÓN EXPLORATORIA
# ================================

# Configurar estilo de gráficos
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Gráfico 1: Serie temporal completa
axes[0,0].plot(data['timestamp'], data['target'], linewidth=1, alpha=0.8)
axes[0,0].set_title('Serie Temporal Sintética Completa', fontsize=14, fontweight='bold')
axes[0,0].set_xlabel('Fecha')
axes[0,0].set_ylabel('Valor')
axes[0,0].grid(True, alpha=0.3)

# Gráfico 2: Últimos 100 puntos (detalle)
axes[0,1].plot(data['timestamp'][-100:], data['target'][-100:], 'o-', linewidth=2, markersize=3)
axes[0,1].set_title('Detalle: Últimos 100 Puntos', fontsize=14, fontweight='bold')
axes[0,1].set_xlabel('Fecha')
axes[0,1].set_ylabel('Valor')
axes[0,1].grid(True, alpha=0.3)

# Gráfico 3: Distribución de valores
axes[1,0].hist(data['target'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
axes[1,0].set_title('Distribución de Valores', fontsize=14, fontweight='bold')
axes[1,0].set_xlabel('Valor')
axes[1,0].set_ylabel('Frecuencia')
axes[1,0].grid(True, alpha=0.3)

# Gráfico 4: Correlación entre variables
correlation_matrix = data[['target', 'external_1', 'external_2']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,1])
axes[1,1].set_title('Matriz de Correlación', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()


# ================================
# 4. PREPARACIÓN DE DATOS
# ================================

def create_sequences(data, target_col, feature_cols, lookback_window=30, forecast_horizon=1):
    """
    Convierte datos de series temporales en secuencias para modelos RNN.
    
    Parámetros:
    - data: DataFrame con los datos
    - target_col: nombre de la columna objetivo
    - feature_cols: lista de columnas de características
    - lookback_window: ventana de tiempo para predicción
    - forecast_horizon: horizonte de predicción
    
    Retorna:
    - X: secuencias de entrada
    - y: valores objetivo
    """
    
    # Seleccionar datos relevantes
    relevant_data = data[feature_cols + [target_col]].values
    
    X, y = [], []
    
    for i in range(lookback_window, len(relevant_data) - forecast_horizon + 1):
        # Secuencia de entrada (características + target histórico)
        X.append(relevant_data[i-lookback_window:i])
        # Valor objetivo a predecir
        y.append(relevant_data[i+forecast_horizon-1, -1])  # -1 es el target_col
    
    return np.array(X), np.array(y)

# Definir parámetros
LOOKBACK_WINDOW = 30  # Usar 30 días anteriores
FORECAST_HORIZON = 1  # Predecir 1 día adelante
FEATURE_COLS = ['external_1', 'external_2', 'day_of_week', 'month', 'target']

print("Preparando secuencias de datos...")

# Crear secuencias
X, y = create_sequences(
    data, 
    target_col='target',
    feature_cols=FEATURE_COLS,
    lookback_window=LOOKBACK_WINDOW,
    forecast_horizon=FORECAST_HORIZON
)

print(f"Forma de X (secuencias): {X.shape}")
print(f"Forma de y (objetivos): {y.shape}")

# División en conjuntos de entrenamiento, validación y prueba
train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))

X_train = X[:train_size]
y_train = y[:train_size]
X_val = X[train_size:train_size+val_size]
y_val = y[train_size:train_size+val_size]
X_test = X[train_size+val_size:]
y_test = y[train_size+val_size:]

print(f"\nConjunto de entrenamiento: {X_train.shape[0]} muestras")
print(f"Conjunto de validación: {X_val.shape[0]} muestras")
print(f"Conjunto de prueba: {X_test.shape[0]} muestras")

# Normalización de datos
scaler = MinMaxScaler()

# Reshape para normalizar
X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)

X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
X_val_scaled = scaler.transform(X_val_reshaped).reshape(X_val.shape)

X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)

# Normalizar targets
y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

print("Normalización completada")

# ================================
# 5. DEFINICIÓN DE MODELOS
# ================================

def create_rnn_model(input_shape, units=50, dropout_rate=0.2):
    """
    Crea un modelo RNN simple (Vanilla RNN)
    """
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

def create_lstm_model(input_shape, units=50, dropout_rate=0.2):
    """
    Crea un modelo LSTM
    """
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

def create_gru_model(input_shape, units=50, dropout_rate=0.2):
    """
    Crea un modelo GRU
    """
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

# Definir forma de entrada
input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
print(f"Forma de entrada para los modelos: {input_shape}")

# ================================
# 6. ENTRENAMIENTO DE MODELOS
# ================================

def train_model(model, model_name, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """
    Entrena un modelo y retorna el historial de entrenamiento
    """
    print(f"\n{'='*50}")
    print(f"ENTRENANDO MODELO: {model_name}")
    print(f"{'='*50}")
    
    # Callback para early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    # Entrenar modelo
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )
    
    return history

# Crear modelos
print("Creando modelos...")
rnn_model = create_rnn_model(input_shape)
lstm_model = create_lstm_model(input_shape)
gru_model = create_gru_model(input_shape)

# Mostrar arquitecturas
print("\n" + "="*60)
print("ARQUITECTURA RNN:")
rnn_model.summary()

print("\n" + "="*60)
print("ARQUITECTURA LSTM:")
lstm_model.summary()

print("\n" + "="*60)
print("ARQUITECTURA GRU:")
gru_model.summary()

# Entrenar modelos
histories = {}

# Entrenar RNN
histories['RNN'] = train_model(
    rnn_model, 'RNN', 
    X_train_scaled, y_train_scaled, 
    X_val_scaled, y_val_scaled
)

# Entrenar LSTM
histories['LSTM'] = train_model(
    lstm_model, 'LSTM', 
    X_train_scaled, y_train_scaled, 
    X_val_scaled, y_val_scaled
)

# Entrenar GRU
histories['GRU'] = train_model(
    gru_model, 'GRU', 
    X_train_scaled, y_train_scaled, 
    X_val_scaled, y_val_scaled
)
# ================================
# 7. EVALUACIÓN DE MODELOS
# ================================

def evaluate_model(model, model_name, X_test, y_test, y_scaler):
    """
    Evalúa un modelo y calcula métricas de rendimiento
    """
    print(f"\nEvaluando modelo: {model_name}")
    
    # Hacer predicciones
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    # Desnormalizar predicciones
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Calcular métricas
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calcular MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape
    }
    
    return metrics, y_pred, y_true

# Evaluar todos los modelos
models = {
    'RNN': rnn_model,
    'LSTM': lstm_model,
    'GRU': gru_model
}

results = {}
predictions = {}

print("\n" + "="*60)
print("EVALUACIÓN DE MODELOS EN CONJUNTO DE PRUEBA")
print("="*60)

for name, model in models.items():
    metrics, y_pred, y_true = evaluate_model(
        model, name, X_test_scaled, y_test_scaled, y_scaler
    )
    results[name] = metrics
    predictions[name] = {'pred': y_pred, 'true': y_true}

# Crear tabla comparativa de resultados
results_df = pd.DataFrame(results).T
print("\nTABLA COMPARATIVA DE MÉTRICAS:")
print("="*60)
print(results_df.round(4))

# ================================
# 8. VISUALIZACIÓN DE RESULTADOS
# ================================

# Gráfico 1: Historial de entrenamiento
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Loss durante entrenamiento
axes[0,0].set_title('Pérdida Durante Entrenamiento', fontsize=14, fontweight='bold')
for name, history in histories.items():
    axes[0,0].plot(history.history['loss'], label=f'{name} - Train')
    axes[0,0].plot(history.history['val_loss'], label=f'{name} - Val', linestyle='--')
axes[0,0].set_xlabel('Época')
axes[0,0].set_ylabel('MSE Loss')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# MAE durante entrenamiento
axes[0,1].set_title('MAE Durante Entrenamiento', fontsize=14, fontweight='bold')
for name, history in histories.items():
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
for i, (name, metrics) in enumerate(results.items()):
    values = [metrics['RMSE'], metrics['MAE'], metrics['MAPE']]
    axes[1,0].bar(x + i*width, values, width, label=name, alpha=0.8)

axes[1,0].set_xlabel('Métricas')
axes[1,0].set_ylabel('Valor')
axes[1,0].set_xticks(x + width)
axes[1,0].set_xticklabels(metrics_names)
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# R² Score
axes[1,1].set_title('R² Score (Coeficiente de Determinación)', fontsize=14, fontweight='bold')
r2_scores = [results[name]['R²'] for name in models.keys()]
bars = axes[1,1].bar(models.keys(), r2_scores, color=['skyblue', 'orange', 'lightgreen'], alpha=0.8)
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
axes[0,0].set_title('Predicciones vs Valores Reales (Completo)', fontsize=14, fontweight='bold')
for name in models.keys():
    axes[0,0].plot(predictions[name]['pred'][:100], label=f'{name} - Predicción', alpha=0.8)
axes[0,0].plot(predictions['RNN']['true'][:100], label='Valor Real', color='black', linewidth=2)
axes[0,0].set_xlabel('Muestra')
axes[0,0].set_ylabel('Valor')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Scatter plots para cada modelo
colors = ['blue', 'orange', 'green']
for i, (name, color) in enumerate(zip(models.keys(), colors)):
    row = (i + 1) // 2
    col = (i + 1) % 2
    
    y_true = predictions[name]['true']
    y_pred = predictions[name]['pred']
    
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
    r2 = results[name]['R²']
    axes[row, col].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[row, col].transAxes,
                       fontsize=12, verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.show()

# ================================
# 9. ANÁLISIS DE RESIDUOS
# ================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, name in enumerate(models.keys()):
    y_true = predictions[name]['true']
    y_pred = predictions[name]['pred']
    residuals = y_true - y_pred
    
    # Histograma de residuos
    axes[i].hist(residuals, bins=30, alpha=0.7, color=colors[i], edgecolor='black')
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

# ================================
# 10. ANÁLISIS COMPARATIVO FINAL
# ================================

print("\n" + "="*80)
print("ANÁLISIS COMPARATIVO FINAL")
print("="*80)

print("\n1. RENDIMIENTO GENERAL:")
print("-" * 40)
best_model = min(results.keys(), key=lambda x: results[x]['RMSE'])
print(f"   MEJOR MODELO (menor RMSE): {best_model}")
print(f"   RMSE: {results[best_model]['RMSE']:.4f}")
print(f"   R²: {results[best_model]['R²']:.4f}")

print("\n2. COMPARACIÓN POR MÉTRICA:")
print("-" * 40)
for metric in ['RMSE', 'MAE', 'MAPE', 'R²']:
    if metric == 'R²':
        best = max(results.keys(), key=lambda x: results[x][metric])
        print(f"{metric:8s}: {best:4s} ({results[best][metric]:7.4f})")
    else:
        best = min(results.keys(), key=lambda x: results[x][metric])
        print(f"{metric:8s}: {best:4s} ({results[best][metric]:7.4f})")

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
for name, model in models.items():
    params = model.count_params()
    print(f"{name:4s}: {params:,} parámetros")

print("\n5. RECOMENDACIONES:")
print("-" * 40)
print("• Para series temporales complejas con patrones a largo plazo: LSTM o GRU")
print("• Para aplicaciones con restricciones computacionales: GRU")
print("• Para patrones simples y velocidad: RNN simple")
print("• Para este dataset específico: El modelo con mejor RMSE es el recomendado")

print("\n" + "="*80)
print("ANÁLISIS COMPLETADO")
print("="*80)

# ================================
# 11. GUARDAR MODELOS Y RESULTADOS
# ================================

# Guardar modelos
print("\nGuardando modelos entrenados...")
rnn_model.save('rnn_model.h5')
lstm_model.save('lstm_model.h5')
gru_model.save('gru_model.h5')

# Guardar resultados
results_df.to_csv('model_comparison_results.csv')
print("Modelos y resultados guardados exitosamente")

print("\n NOTEBOOK COMPLETADO EXITOSAMENTE ")
print("Archivos generados:")
print("- synthetic_timeseries_data.csv")
print("- model_comparison_results.csv")
print("- rnn_model.h5")
print("- lstm_model.h5") 
print("- gru_model.h5")
