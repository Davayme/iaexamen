import numpy as np
import matplotlib.pyplot as plt

# Función objetivo
def f(x):
    return -x**2 - 4*x

# Hill Climbing
def hill_climbing(f, x_inicial, paso=0.1, max_iter=100):
    x_actual = x_inicial
    historial = [x_actual]  # Para guardar los pasos
    
    for _ in range(max_iter):
        # Generar vecinos
        vecinos = [x_actual + paso, x_actual - paso] 
        # Evaluar vecinos y seleccionar el mejor
        x_siguiente = max(vecinos, key=f)
        
        if f(x_siguiente) <= f(x_actual):
            break  # ¡Óptimo local encontrado!
        
        x_actual = x_siguiente
        historial.append(x_actual)
    
    return x_actual, historial

# Ejecutar
x_opt, pasos = hill_climbing(f, x_inicial=0.0)

# Graficar
x_vals = np.linspace(-5, 1, 100)
plt.plot(x_vals, f(x_vals), label="f(x) = -x² - 4x")
plt.scatter(pasos, [f(x) for x in pasos], color='red', label="Pasos de Hill Climbing")
plt.scatter(x_opt, f(x_opt), color='green', s=100, label="Óptimo local")
plt.legend()
plt.grid()
plt.show()

print(f"Máximo encontrado: x = {x_opt:.2f}, f(x) = {f(x_opt):.2f}")