import numpy as np
import matplotlib.pyplot as plt
import math
import random

# Función objetivo que queremos maximizar
# En este caso es una función senoidal amortiguada
def f(x):
    return np.sin(x) * np.exp(-0.1 * x)  # Combinación de seno y exponencial decreciente

# Algoritmo de Recocido Simulado
def recocido_simulado(f, x_inicial, temp_inicial=1000, enfriamiento=0.95, iter_por_temp=100):
    """
    Implementación del algoritmo de recocido simulado para maximización.
    
    Parámetros:
    - f: función objetivo que queremos maximizar
    - x_inicial: punto de partida de la búsqueda
    - temp_inicial: temperatura inicial (alta = más probabilidad de aceptar soluciones peores)
    - enfriamiento: factor de reducción de temperatura (0-1)
    - iter_por_temp: número de iteraciones a realizar en cada temperatura
    """
    # Inicialización
    x_actual = x_inicial       # Solución actual
    temp = temp_inicial        # Temperatura actual
    historial = []             # Lista para almacenar las posiciones visitadas
    
    # Bucle principal - continúa hasta que la temperatura sea muy baja
    while temp > 1e-3:         # Criterio de parada: temperatura casi cero
        # Para cada temperatura, realizamos varias iteraciones
        for _ in range(iter_por_temp):
            # PASO 1: Generar una solución vecina aleatoria
            # Tomamos un paso aleatorio entre -1 y 1 desde la posición actual
            x_vecino = x_actual + random.uniform(-1, 1)
            
            # PASO 2: Calcular el cambio en la función objetivo
            # Delta positivo = el vecino es mejor (mayor valor en f)
            # Delta negativo = el vecino es peor (menor valor en f)
            delta = f(x_vecino) - f(x_actual)
            
            # PASO 3: Decidir si aceptamos el movimiento
            # Criterio de Metropolis:
            # - Si delta > 0: aceptamos siempre (encontramos mejor solución)
            # - Si delta < 0: aceptamos con probabilidad exp(delta/temp)
            #   (más probable aceptar malos movimientos al inicio cuando temp es alta)
            if delta > 0 or random.random() < math.exp(delta / temp):
                x_actual = x_vecino  # Aceptamos el movimiento
            
            # Guardamos la posición actual para visualización
            historial.append(x_actual)
        
        # PASO 4: Enfriar el sistema (reducir temperatura)
        # Cada ciclo reducimos la temperatura multiplicando por el factor de enfriamiento
        temp *= enfriamiento  # Por ejemplo: 1000 → 950 → 902.5 → ...
    
    # Devolvemos la mejor solución encontrada y el historial de búsqueda
    return x_actual, historial

# Ejecución del algoritmo
x_opt, historial = recocido_simulado(f, x_inicial=0.0)

# Visualización gráfica de los resultados
# Creamos un rango de valores x para mostrar la función completa
x_vals = np.linspace(0, 20, 100)

# Graficamos la función objetivo
plt.plot(x_vals, f(x_vals), label="f(x)")

# Graficamos los puntos visitados durante la búsqueda
plt.scatter(historial, [f(x) for x in historial], c='red', s=5, alpha=0.5, label="Búsqueda")

# Destacamos la solución óptima encontrada
plt.scatter(x_opt, f(x_opt), c='green', s=100, label="Óptimo")

plt.legend()
plt.show()

# Mostramos el resultado final
print(f"Máximo encontrado: x = {x_opt:.2f}, f(x) = {f(x_opt):.2f}")