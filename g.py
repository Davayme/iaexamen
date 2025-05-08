import random

# --------------------------
# CONFIGURACIÓN (¡Tú la ajustas!)
# --------------------------
GENES = ["0", "1"]  # Ej: bits, letras, números...
LONGITUD_INDIVIDUO = 5  # Tamaño de la solución
TAMANO_POBLACION = 10
PROB_MUTACION = 0.1
GENERACIONES = 10

# Función de fitness (¡Tú la defines!)
def fitness(individuo):
    # Ejemplo 1: Maximizar x² (para binario)
    x = int("".join(individuo), 2)
    return x ** 2
    
    # Ejemplo 2: Minimizar distancia (para rutas)
    # return calcular_distancia(individuo)

# --------------------------
# ALGORITMO GENÉTICO (Fijo)
# --------------------------
def crear_individuo():
    return [random.choice(GENES) for _ in range(LONGITUD_INDIVIDUO)]

def seleccion_torneo(poblacion, k=3):
    participantes = random.sample(poblacion, k)
    return max(participantes, key=fitness)

def cruza(padre1, padre2):
    punto = random.randint(1, LONGITUD_INDIVIDUO - 1)
    hijo = padre1[:punto] + padre2[punto:]
    return hijo

def mutar(individuo):
    for i in range(LONGITUD_INDIVIDUO):
        if random.random() < PROB_MUTACION:
            individuo[i] = random.choice(GENES)
    return individuo

# Población inicial
poblacion = [crear_individuo() for _ in range(TAMANO_POBLACION)]

# Evolución
for generacion in range(GENERACIONES):
    # Selección
    padres = [seleccion_torneo(poblacion) for _ in range(TAMANO_POBLACION)]
    
    # Cruza
    hijos = []
    for i in range(0, TAMANO_POBLACION, 2):
        hijo1 = cruza(padres[i], padres[i+1])
        hijo2 = cruza(padres[i+1], padres[i])
        hijos.extend([hijo1, hijo2])
    
    # Mutación
    hijos = [mutar(hijo) for hijo in hijos]
    
    # Reemplazo
    poblacion = hijos
    
    # Mostrar progreso
    mejor = max(poblacion, key=fitness)
    print(f"Gen {generacion}: {mejor} -> Fitness: {fitness(mejor)}")