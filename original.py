# ============================================================================
# DEMOSTRACIÓN DE REDES NEURONALES CONVOLUCIONALES CON CIFAR-10
# Programa educativo para Google Colab
# ============================================================================

# Importación de las bibliotecas necesarias
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

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

# Visualización de algunas imágenes de ejemplo
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

# Análisis de la distribución de clases
plt.figure(figsize=(10, 6))
clases_unicas, conteos = np.unique(y_train, return_counts=True)
plt.bar([nombres_clases[i] for i in clases_unicas], conteos, color='skyblue')
plt.title('Distribución de clases en el conjunto de entrenamiento')
plt.xlabel('Clases')
plt.ylabel('Número de imágenes')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================================
# PASO 2: PREPROCESAMIENTO DE DATOS
# ============================================================================

print("\n" + "="*60)
print("PASO 2: PREPROCESAMIENTO DE LOS DATOS")
print("="*60)

# Normalización de píxeles: convertir de rango [0, 255] a [0, 1]
# Esto ayuda a que la red neuronal converja más rápidamente
x_train_norm = x_train.astype('float32') / 255.0
x_test_norm = x_test.astype('float32') / 255.0

print("Antes de normalización - Rango de píxeles:", x_train.min(), "a", x_train.max())
print("Después de normalización - Rango de píxeles:", x_train_norm.min(), "a", x_train_norm.max())

# Conversión de etiquetas a codificación categórica (one-hot encoding)
# Esto transforma etiquetas como [3] a [0,0,0,1,0,0,0,0,0,0]
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

print(f"Forma de etiquetas antes: {y_train.shape}")
print(f"Forma de etiquetas después: {y_train_cat.shape}")
print(f"Ejemplo de etiqueta original: {y_train[0]} -> Categórica: {y_train_cat[0]}")

# ============================================================================
# PASO 3: CONSTRUCCIÓN DEL MODELO CNN
# ============================================================================

print("\n" + "="*60)
print("PASO 3: CONSTRUYENDO LA RED NEURONAL CONVOLUCIONAL")
print("="*60)

def crear_modelo_cnn():
    """
    Construye una CNN con arquitectura moderna y técnicas de regularización.
    
    La arquitectura incluye:
    - Capas convolucionales para extracción de características
    - Capas de pooling para reducir dimensionalidad
    - Batch normalization para estabilizar el entrenamiento
    - Dropout para prevenir sobreajuste
    - Capas densas para clasificación final
    """
    modelo = keras.Sequential([
        # Primer bloque convolucional
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Segundo bloque convolucional
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Tercer bloque convolucional
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # Capas de clasificación
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')  # 10 clases de salida
    ])
    
    return modelo

# Crear el modelo
modelo = crear_modelo_cnn()

# Mostrar la arquitectura del modelo
print("Arquitectura del modelo:")
modelo.summary()

# Visualizar la arquitectura
tf.keras.utils.plot_model(modelo, show_shapes=True, show_layer_names=True, 
                          rankdir="TB", dpi=150)

# ============================================================================
# PASO 4: COMPILACIÓN DEL MODELO
# ============================================================================

print("\n" + "="*60)
print("PASO 4: COMPILANDO EL MODELO")
print("="*60)

# Configuración del optimizador y funciones de pérdida
modelo.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),  # Optimizador Adam con tasa de aprendizaje
    loss='categorical_crossentropy',                        # Función de pérdida para clasificación multiclase
    metrics=['accuracy']                                    # Métrica para monitorear durante el entrenamiento
)

print("Modelo compilado exitosamente!")
print("Optimizador: Adam (learning_rate=0.001)")
print("Función de pérdida: categorical_crossentropy")
print("Métricas: accuracy")

# ============================================================================
# PASO 5: CONFIGURACIÓN DE CALLBACKS
# ============================================================================

print("\n" + "="*60)
print("PASO 5: CONFIGURANDO CALLBACKS PARA EL ENTRENAMIENTO")
print("="*60)

# Callback para reducir la tasa de aprendizaje cuando el rendimiento se estanque
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.0001,
    verbose=1
)

# Callback para detener el entrenamiento temprano si no hay mejora
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

callbacks = [reduce_lr, early_stopping]
print("Callbacks configurados:")
print("- ReduceLROnPlateau: reduce tasa de aprendizaje si no hay mejora")
print("- EarlyStopping: detiene entrenamiento temprano para evitar sobreajuste")

# ============================================================================
# PASO 6: ENTRENAMIENTO DEL MODELO
# ============================================================================

print("\n" + "="*60)
print("PASO 6: ENTRENANDO EL MODELO")
print("="*60)

# Configuración del entrenamiento
EPOCHS = 50          # Número máximo de épocas
BATCH_SIZE = 32      # Tamaño del lote para entrenamiento
VALIDATION_SPLIT = 0.2  # Porcentaje de datos para validación

print(f"Configuración del entrenamiento:")
print(f"- Épocas máximas: {EPOCHS}")
print(f"- Tamaño del lote: {BATCH_SIZE}")
print(f"- División de validación: {VALIDATION_SPLIT}")
print(f"- Datos de entrenamiento: {len(x_train_norm) * (1-VALIDATION_SPLIT):.0f} imágenes")
print(f"- Datos de validación: {len(x_train_norm) * VALIDATION_SPLIT:.0f} imágenes")

print("\nIniciando entrenamiento...")

# Entrenar el modelo
history = modelo.fit(
    x_train_norm, y_train_cat,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    callbacks=callbacks,
    verbose=1
)

print("\n¡Entrenamiento completado!")

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
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]

print(f"Resultados finales del entrenamiento:")
print(f"- Precisión de entrenamiento: {final_train_acc:.4f}")
print(f"- Precisión de validación: {final_val_acc:.4f}")
print(f"- Pérdida de entrenamiento: {final_train_loss:.4f}")
print(f"- Pérdida de validación: {final_val_loss:.4f}")

# ============================================================================
# PASO 8: EVALUACIÓN EN DATOS DE PRUEBA
# ============================================================================

print("\n" + "="*60)
print("PASO 8: EVALUACIÓN FINAL EN DATOS DE PRUEBA")
print("="*60)

# Evaluar el modelo en el conjunto de prueba
test_loss, test_accuracy = modelo.evaluate(x_test_norm, y_test_cat, verbose=0)
print(f"Precisión en datos de prueba: {test_accuracy:.4f}")
print(f"Pérdida en datos de prueba: {test_loss:.4f}")

# Generar predicciones para análisis detallado
y_pred = modelo.predict(x_test_norm)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test_cat, axis=1)

# Reporte de clasificación detallado
print("\nReporte de clasificación detallado:")
print(classification_report(y_true_classes, y_pred_classes, 
                          target_names=nombres_clases, digits=4))

# ============================================================================
# PASO 9: MATRIZ DE CONFUSIÓN
# ============================================================================

print("\n" + "="*60)
print("PASO 9: ANÁLISIS DE ERRORES CON MATRIZ DE CONFUSIÓN")
print("="*60)

# Calcular matriz de confusión
cm = confusion_matrix(y_true_classes, y_pred_classes)

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
# PASO 10: EJEMPLOS DE PREDICCIONES
# ============================================================================

print("\n" + "="*60)
print("PASO 10: VISUALIZACIÓN DE PREDICCIONES INDIVIDUALES")
print("="*60)

def mostrar_predicciones(modelo, x_test, y_test, nombres_clases, num_ejemplos=12):
    """Muestra ejemplos de predicciones con sus probabilidades"""
    
    # Seleccionar ejemplos aleatorios
    indices = np.random.choice(len(x_test), num_ejemplos, replace=False)
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Ejemplos de Predicciones del Modelo', fontsize=16)
    
    for i, idx in enumerate(indices):
        fila = i // 4
        col = i % 4
        
        # Realizar predicción
        pred = modelo.predict(x_test[idx:idx+1])
        pred_class = np.argmax(pred[0])
        pred_prob = np.max(pred[0])
        true_class = y_test[idx][0]
        
        # Mostrar imagen
        axes[fila, col].imshow(x_test[idx])
        
        # Color del título según si la predicción es correcta
        color = 'green' if pred_class == true_class else 'red'
        
        axes[fila, col].set_title(
            f'Real: {nombres_clases[true_class]}\n'
            f'Pred: {nombres_clases[pred_class]} ({pred_prob:.2f})',
            color=color, fontsize=10
        )
        axes[fila, col].axis('off')
    
    plt.tight_layout()
    plt.show()

# Mostrar ejemplos de predicciones
mostrar_predicciones(modelo, x_test, y_test, nombres_clases)

# ============================================================================
# PASO 11: ANÁLISIS DE CARACTERÍSTICAS APRENDIDAS
# ============================================================================

print("\n" + "="*60)
print("PASO 11: VISUALIZACIÓN DE FILTROS CONVOLUCIONALES")
print("="*60)

def visualizar_filtros(modelo, capa_idx=0, num_filtros=8):
    """Visualiza los filtros aprendidos por una capa convolucional"""
    
    # Obtener los pesos de la primera capa convolucional
    filtros = modelo.layers[capa_idx].get_weights()[0]
    
    # Normalizar filtros para visualización
    f_min, f_max = filtros.min(), filtros.max()
    filtros = (filtros - f_min) / (f_max - f_min)
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle(f'Filtros de la Capa Convolucional {capa_idx + 1}', fontsize=14)
    
    for i in range(min(num_filtros, filtros.shape[3])):
        fila = i // 4
        col = i % 4
        axes[fila, col].imshow(filtros[:, :, :, i])
        axes[fila, col].set_title(f'Filtro {i+1}')
        axes[fila, col].axis('off')
    
    plt.tight_layout()
    plt.show()

# Visualizar filtros de la primera capa
visualizar_filtros(modelo, capa_idx=0)

# ============================================================================
# PASO 12: RESUMEN Y CONCLUSIONES
# ============================================================================

print("\n" + "="*60)
print("PASO 12: RESUMEN Y CONCLUSIONES")
print("="*60)

print("🎉 ¡DEMOSTRACIÓN COMPLETADA EXITOSAMENTE!")
print("\nResumen de resultados:")
print(f"✅ Modelo entrenado con {len(x_train)} imágenes")
print(f"✅ Precisión final en prueba: {test_accuracy:.1%}")
print(f"✅ Arquitectura CNN con {modelo.count_params():,} parámetros")
print(f"✅ Entrenamiento completado en {len(history.history['loss'])} épocas")

print("\n📚 Conceptos demostrados:")
print("• Carga y preprocesamiento de datos de imagen")
print("• Construcción de arquitectura CNN moderna")
print("• Técnicas de regularización (Dropout, BatchNorm)")
print("• Entrenamiento con callbacks inteligentes")
print("• Evaluación exhaustiva de resultados")
print("• Visualización de filtros y predicciones")

print("\n🔍 Puntos clave aprendidos:")
print("• Las CNNs extraen características jerárquicas automáticamente")
print("• La normalización de datos mejora la convergencia")
print("• Los callbacks previenen el sobreajuste efectivamente")
print("• La matriz de confusión revela patrones de error específicos")
print("• Los filtros convolucionales aprenden detectores de características")

print("\n🚀 Próximos pasos sugeridos:")
print("• Experimentar con diferentes arquitecturas (ResNet, VGG)")
print("• Probar técnicas de aumento de datos (data augmentation)")
print("• Implementar transfer learning con modelos preentrenados")
print("• Explorar otros conjuntos de datos (CIFAR-100, ImageNet)")

print("\n" + "="*60)
print("¡Gracias por explorar las redes neuronales convolucionales!")
print("="*60)