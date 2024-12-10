import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lti, step, TransferFunction
from deap import base, creator, tools, algorithms

# --- Limpiar definiciones previas en creator ---
if "FitnessMin" in creator.__dict__:
    del creator.FitnessMin
if "Individual" in creator.__dict__:
    del creator.Individual

# Crear el espacio de soluciones
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, 0, 10)  # Rango de búsqueda para Kp, Ki, Kd
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=3)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", lambda ind: fitness(ind))
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# --- Definir las ganancias según valores iniciales ---
Kp_m = 120  # Ganancia proporcional
Ki_m = 160  # Ganancia integral
Kd_m = 120  # Ganancia derivativa

# --- Parámetros del sistema ---
M = 2  # Masa del carro (kg)
m = 0.5  # Masa del péndulo (kg)
l = 1  # Longitud del péndulo (m)
g = 9.81  # Aceleración de la gravedad (m/s²)

# Definir numerador y denominador de la planta
num = [1]
den = [M * l, 0, -(M + m) * g]
plant = TransferFunction(num, den)

# --- Simulación del sistema cerrado con un PID ---
def simulate_pid(kp, ki, kd, plant, time):
    """Simula la respuesta de la planta controlada por un PID."""
    # PID Transfer Function: G_pid(s) = kp + ki/s + kd*s
    num_pid = [kd, kp, ki]  # Numerador del PID
    den_pid = [1, 0]  # Denominador del PID
    pid = TransferFunction(num_pid, den_pid)
    
    # Sistema cerrado: G_cl(s) = (G_pid(s) * Plant(s)) / (1 + G_pid(s) * Plant(s))
    system_closed = lti(np.convolve(pid.num, plant.num),
                        np.polyadd(np.convolve(pid.num, plant.num),
                                   np.convolve(pid.den, plant.den)))
    
    # Simulación del escalón
    t, y = step(system_closed, T=time)
    return t, y

# --- Definir la función objetivo ---
def fitness(individual):
    """Función objetivo para el algoritmo genético."""
    kp, ki, kd = individual
    time = np.linspace(0, 10, 1000)  # Tiempo de simulación
    t, y = simulate_pid(kp, ki, kd, plant, time)
    
    # Métrica: Minimizar el error cuadrático medio (MSE) y el overshoot
    mse = np.mean((y - 1)**2)  # Error respecto al escalón unitario
    overshoot = max(y) - 1 if max(y) > 1 else 0  # Penalización por sobreimpulso
    
    # Penalización si el sistema es inestable (explosión de la salida)
    if np.isnan(y).any() or max(y) > 10:
        return 1e6,  # Valor muy alto para descartar
    
    return mse + overshoot,  # Fitness a minimizar

# --- Ejecución del algoritmo genético ---
def main():
    np.random.seed(42)  # Para reproducibilidad
    pop = toolbox.population(n=100)  # Tamaño de la población
    hof = tools.HallOfFame(1)  # Mejor individuo encontrado
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=1, mutpb=0.2, ngen=50, stats=stats, halloffame=hof, verbose=True)
    
    # Mejor conjunto de parámetros encontrados
    best = hof[0]
    print(f"Mejor individuo (Kp, Ki, Kd): {best}")
    
    # Simulación final con los parámetros encontrados
    time = np.linspace(0, 10, 1000)
    t, y = simulate_pid(best[0], best[1], best[2], plant, time)
    t_m, y_m = simulate_pid(Kp_m, Ki_m, Kd_m, plant, time)
    
    # Graficar resultado
    plt.figure()
    plt.plot(t, y, label="Gráfica PID ajustado con algoritmos genéticos")
    plt.plot(t_m, y_m, label="Gráfica PID ajustado con ajuste manual")
    plt.axhline(1, color='r', linestyle='--', label="Referencia (Setpoint)")
    plt.title("Respuesta del sistema con PID optimizado")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Salida")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()