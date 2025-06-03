# ============================================================================
# DEMOSTRACIÓN DE REDES NEURONALES CONVOLUCIONALES CON CIFAR-10
# Programa educativo optimizado para entrenamiento rápido
# ============================================================================

# Importación de las bibliotecas necesarias
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import time
import warnings
warnings.filterwarnings('ignore')

# Configurar TensorFlow para mejor rendimiento
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU disponible para aceleración")
else:
    print("Ejecutando en CPU")

print("TensorFlow versión:", tf.__version__)
print("¿GPU disponible?", tf.config.list_physical_devices('GPU'))

# ============================================================================
# PASO 1: CARGA Y EXPLORACIÓN DE LOS DATOS
# ============================================================================

print("\n" + "="*60)
print("PASO 1: CARGANDO Y EXPLORANDO LOS DATOS CIFAR-10")
print("="*60)

# CIFAR-10 contiene 60,000 imágenes de 32x32 en 10 clases
# Las clases son: avión, automóvil, pájaro, gato, ciervo, perro, rana, caballo, barco, camión
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Nombres de las clases para hacer más interpretables los resultados
nombres_clases = ['Avión', 'Automóvil', 'Pájaro', 'Gato', 'Ciervo', 
                  'Perro', 'Rana', 'Caballo', 'Barco', 'Camión']

print(f"Forma de datos de entrenamiento: {x_train.shape}")
print(f"Forma de etiquetas de entrenamiento: {y_train.shape}")
print(f"Forma de datos de prueba: {x_test.shape}")
print(f"Forma de etiquetas de prueba: {y_test.shape}")
print(f"Número de clases: {len(nombres_clases)}")

# Visualización de algunas imágenes de ejemplo (opcional - comentar para acelerar)
"""
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
fig.suptitle('Ejemplos de imágenes CIFAR-10', fontsize=16)

for i in range(10):
    fila = i // 5
    col = i % 5
    axes[fila, col].imshow(x_train[i])
    axes[fila, col].set_title(f'{nombres_clases[y_train[i][0]]}')
    axes[fila, col].axis('off')

plt.tight_layout()
plt.show()
"""

# ============================================================================
# PASO 2: PREPROCESAMIENTO DE DATOS
# ============================================================================

print("\n" + "="*60)
print("PASO 2: PREPROCESAMIENTO DE LOS DATOS")
print("="*60)

# Normalización de píxeles: convertir de rango [0, 255] a [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

print("Datos normalizados al rango [0, 1]")

# Conversión de etiquetas a codificación categórica (one-hot encoding)
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

print(f"Etiquetas convertidas a formato one-hot")

# Configurar tamaño de batch para equilibrar velocidad y memoria
BATCH_SIZE = 128  # Aumentado para acelerar entrenamiento
VALIDATION_SPLIT = 0.2

# ============================================================================
# PASO 3: CONSTRUCCIÓN DEL MODELO CNN
# ============================================================================

print("\n" + "="*60)
print("PASO 3: CONSTRUYENDO LA RED NEURONAL CONVOLUCIONAL OPTIMIZADA")
print("="*60)

def crear_modelo_cnn_eficiente():
    """
    Construye una CNN optimizada para velocidad y rendimiento
    """
    modelo = keras.Sequential([
        # Primera capa convolucional - entrada
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        
        # Segunda capa convolucional
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Tercera capa convolucional
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Capa de clasificación
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),  # Solo un dropout al final
        layers.Dense(10, activation='softmax')  # 10 clases
    ])
    
    return modelo

# Crear el modelo
modelo = crear_modelo_cnn_eficiente()

# Mostrar resumen del modelo
print("Arquitectura del modelo optimizado:")
modelo.summary()

# ============================================================================
# PASO 4: COMPILACIÓN DEL MODELO
# ============================================================================

print("\n" + "="*60)
print("PASO 4: COMPILANDO EL MODELO")
print("="*60)

# Configuración del optimizador
modelo.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),  # Learning rate estándar para equilibrio
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Modelo compilado exitosamente!")
print("Optimizador: Adam (learning_rate=0.001)")
print("Función de pérdida: categorical_crossentropy")
print("Métricas: accuracy")

# ============================================================================
# PASO 5: CONFIGURACIÓN DE CALLBACKS
# ============================================================================

print("\n" + "="*60)
print("PASO 5: CONFIGURANDO CALLBACKS PARA ENTRENAMIENTO EFICIENTE")
print("="*60)

# Callback para reducir la tasa de aprendizaje cuando el rendimiento se estanca
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=3,
    min_lr=0.0001,
    verbose=1
)

# Callback para detener el entrenamiento temprano
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Lista de callbacks
callbacks = [reduce_lr, early_stopping]

print("Callbacks configurados:")
print("- ReduceLROnPlateau: reduce tasa de aprendizaje cuando la precisión se estanca")
print("- EarlyStopping: detiene el entrenamiento cuando no hay más mejoras")

# ============================================================================
# PASO 6: ENTRENAMIENTO DEL MODELO
# ============================================================================

print("\n" + "="*60)
print("PASO 6: ENTRENANDO EL MODELO")
print("="*60)

# Configuración del entrenamiento
EPOCHS = 25  # Máximo de épocas

print(f"Configuración del entrenamiento:")
print(f"- Épocas máximas: {EPOCHS}")
print(f"- Tamaño del lote: {BATCH_SIZE}")
print(f"- División de validación: {VALIDATION_SPLIT}")

print("\nIniciando entrenamiento...")
start_time = time.time()

# Entrenar el modelo
history = modelo.fit(
    x_train, y_train_cat,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    callbacks=callbacks,
    verbose=1
)

end_time = time.time()
training_time = end_time - start_time

print("\n¡Entrenamiento completado!")
print(f"Tiempo total de entrenamiento: {training_time:.2f} segundos")
print(f"Tiempo promedio por época: {training_time / len(history.history['loss']):.2f} segundos")

# ============================================================================
# PASO 7: VISUALIZACIÓN DEL ENTRENAMIENTO
# ============================================================================

print("\n" + "="*60)
print("PASO 7: ANALIZANDO EL PROGRESO DEL ENTRENAMIENTO")
print("="*60)

def plot_training_history(history):
    """Visualiza las métricas de entrenamiento a lo largo de las épocas"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Gráfico de precisión
    ax1.plot(history.history['accuracy'], label='Precisión de entrenamiento', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Precisión de validación', linewidth=2)
    ax1.set_title('Evolución de la Precisión del Modelo', fontsize=14)
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Precisión')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico de pérdida
    ax2.plot(history.history['loss'], label='Pérdida de entrenamiento', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Pérdida de validación', linewidth=2)
    ax2.set_title('Evolución de la Pérdida del Modelo', fontsize=14)
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Pérdida')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Mostrar gráficos de entrenamiento
plot_training_history(history)

# Estadísticas finales del entrenamiento
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]

print(f"Resultados finales del entrenamiento:")
print(f"- Precisión de entrenamiento: {final_train_acc:.4f}")
print(f"- Precisión de validación: {final_val_acc:.4f}")

# ============================================================================
# PASO 8: EVALUACIÓN EN DATOS DE PRUEBA
# ============================================================================

print("\n" + "="*60)
print("PASO 8: EVALUACIÓN FINAL EN DATOS DE PRUEBA")
print("="*60)

# Evaluar el modelo en el conjunto de prueba
test_loss, test_accuracy = modelo.evaluate(x_test, y_test_cat, verbose=0)
print(f"Precisión en datos de prueba: {test_accuracy:.4f}")
print(f"Pérdida en datos de prueba: {test_loss:.4f}")

# Generar predicciones para análisis detallado
y_pred_probs = modelo.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test_cat, axis=1)

# Reporte de clasificación detallado
print("\nReporte de clasificación detallado:")
print(classification_report(y_true, y_pred, 
                          target_names=nombres_clases, digits=4))

# ============================================================================
# PASO 9: MATRIZ DE CONFUSIÓN
# ============================================================================

print("\n" + "="*60)
print("PASO 9: ANÁLISIS DE ERRORES CON MATRIZ DE CONFUSIÓN")
print("="*60)

# Calcular matriz de confusión
cm = confusion_matrix(y_true, y_pred)

# Visualizar matriz de confusión
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=nombres_clases, yticklabels=nombres_clases)
plt.title('Matriz de Confusión - Resultados de Clasificación', fontsize=16)
plt.xlabel('Predicciones')
plt.ylabel('Valores Reales')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Análisis de precisión por clase
precisiones_por_clase = cm.diagonal() / cm.sum(axis=1)
print("Precisión por clase:")
for i, nombre in enumerate(nombres_clases):
    print(f"{nombre}: {precisiones_por_clase[i]:.4f}")

# ============================================================================
# PASO 10: RESUMEN Y CONCLUSIONES
# ============================================================================

print("\n" + "="*60)
print("PASO 10: RESUMEN Y CONCLUSIONES")
print("="*60)

print("🎉 ¡DEMOSTRACIÓN COMPLETADA EXITOSAMENTE!")
print("\nResumen de resultados del modelo optimizado:")
print(f"✅ Precisión final en datos de prueba: {test_accuracy:.1%}")
print(f"✅ Tiempo total de entrenamiento: {training_time:.2f} segundos")
print(f"✅ Tiempo promedio por época: {training_time / len(history.history['loss']):.2f} segundos")
print(f"✅ Número de parámetros del modelo: {modelo.count_params():,}")
print(f"✅ Entrenamiento completado en {len(history.history['loss'])} épocas")

print("\nOptimizaciones aplicadas:")
print("• Arquitectura CNN más eficiente")
print("• Batch size aumentado a 128")
print("• Menos capas de regularización")
print("• Estrategias eficientes de entrenamiento")

print("\n" + "="*60)
print("¡Gracias por explorar las redes neuronales convolucionales!")
print("="*60)