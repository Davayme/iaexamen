import numpy as np
import matplotlib.pyplot as plt
import math
import random

# Función objetivo
def f(x):
    return np.sin(x) * np.exp(-0.1 * x)

# Recocido Simulado
def recocido_simulado(f, x_inicial, temp_inicial=1000, enfriamiento=0.95, iter_por_temp=100):
    x_actual = x_inicial
    temp = temp_inicial
    historial = []  # Para graficar
    
    while temp > 1e-3:  # Condición de parada (temp cercana a 0)
        for _ in range(iter_por_temp):
            # Generar vecino aleatorio
            x_vecino = x_actual + random.uniform(-1, 1)
            delta = f(x_vecino) - f(x_actual)
            
            # Criterio de aceptación
            if delta > 0 or random.random() < math.exp(delta / temp):
                x_actual = x_vecino
            historial.append(x_actual)
        
        temp *= enfriamiento  # Enfriar
    
    return x_actual, historial

# Ejecución
x_opt, historial = recocido_simulado(f, x_inicial=0.0)

# Graficar
x_vals = np.linspace(0, 20, 100)
plt.plot(x_vals, f(x_vals), label="f(x)")
plt.scatter(historial, [f(x) for x in historial], c='red', s=5, alpha=0.5, label="Búsqueda")
plt.scatter(x_opt, f(x_opt), c='green', s=100, label="Óptimo")
plt.legend()
plt.show()

print(f"Máximo encontrado: x = {x_opt:.2f}, f(x) = {f(x_opt):.2f}")