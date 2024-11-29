import numpy as np
import control as ctrl
from deap import base, creator, tools, algorithms

# Sistema de ejemplo: Planta G(s) = 1 / (s^2 + 3s + 2)
numerador = [1]
denominador = [1, 3, 2]
planta = ctrl.TransferFunction(numerador, denominador)

# Función objetivo: Evaluar rendimiento del controlador PID
def evaluar_pid(individuo):
    Kp, Ki, Kd = individuo
    # Controlador PID
    controlador = ctrl.TransferFunction([Kd, Kp, Ki], [1, 0])
    # Lazo cerrado
    sistema_lazo_cerrado = ctrl.feedback(controlador * planta)
    # Simulación de la respuesta al escalón
    t, yout = ctrl.step_response(sistema_lazo_cerrado)
    # Error: Deseamos que la salida siga un escalón unitario
    error = 1 - yout
    # ISE: Integral del error al cuadrado
    ise = np.sum(error**2) * (t[1] - t[0])
    return ise,

# Configuración del algoritmo genético
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimizar ISE
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

# Definir individuos (Kp, Ki, Kd)
toolbox.register("attr_float", np.random.uniform, 0, 10)  # Rango de [0, 10]
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 3)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Operadores genéticos
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluar_pid)

# Proceso evolutivo
population = toolbox.population(n=50)
ngen = 20  # Número de generaciones
cxpb = 0.7  # Probabilidad de cruzamiento
mutpb = 0.2  # Probabilidad de mutación

# Ejecutar algoritmo genético
resultados = algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, verbose=True)

# Obtener los mejores parámetros PID
mejor_individuo = tools.selBest(population, k=1)[0]
print("Mejores ganancias PID encontradas:")
print(f"Kp = {mejor_individuo[0]}, Ki = {mejor_individuo[1]}, Kd = {mejor_individuo[2]}")
