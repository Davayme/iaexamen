import numpy as np
import matplotlib.pyplot as plt
import math
import random
from IPython.display import display, clear_output
import time

# Función objetivo (la misma del PDF)
def f(x):
    return np.sin(x) * np.exp(-0.1 * x)

# Configuración inicial
x_actual = random.uniform(0, 10)  # Punto inicial aleatorio
T = 1000  # Temperatura inicial
paso = 1.0  # Tamaño del paso para generar vecinos
historial = {'x': [], 'f_x': [], 'T': [], 'aceptado': []}

# Crear la figura
plt.figure(figsize=(12, 6))
x_vals = np.linspace(0, 20, 400)
y_vals = f(x_vals)

# Bucle del Recocido Simulado (con animación)
for iteracion in range(100):
    # Generar vecino aleatorio
    x_vecino = x_actual + random.uniform(-paso, paso)
    
    # Asegurar que el vecino esté dentro del dominio
    x_vecino = max(0, min(x_vecino, 20))
    
    # Calcular cambio en la función
    delta = f(x_vecino) - f(x_actual)
    
    # Decidir si aceptar el vecino
    if delta > 0 or random.random() < math.exp(delta / T):
        x_actual = x_vecino
        aceptado = True
    else:
        aceptado = False
    
    # Guardar historial para la gráfica
    historial['x'].append(x_actual)
    historial['f_x'].append(f(x_actual))
    historial['T'].append(T)
    historial['aceptado'].append(aceptado)
    
    # Actualizar la gráfica en cada iteración
    clear_output(wait=True)
    plt.clf()
    
    # Graficar la función
    plt.plot(x_vals, y_vals, label="f(x) = sin(x) * e^(-0.1x)", color='blue')
    
    # Graficar los pasos del algoritmo
    colores = ['green' if a else 'red' for a in historial['aceptado']]
    plt.scatter(historial['x'], historial['f_x'], c=colores, s=50, alpha=0.6, 
                label="Aceptado (verde) / Rechazado (rojo)")
    
    # Destacar el punto actual
    plt.scatter(x_actual, f(x_actual), color='black', s=100, label="Posición actual")
    
    # Añadir detalles
    plt.title(f"Recocido Simulado\nIteración: {iteracion + 1}, Temperatura: {T:.2f}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid()
    plt.show()
    
    # Pausa para ver la animación
    time.sleep(0.5)
    
    # Enfriar la temperatura
    T *= 0.95

print(f"Solución final: x = {x_actual:.2f}, f(x) = {f(x_actual):.2f}")