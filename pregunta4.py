import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lti, step, lsim, TransferFunction

# Parámetros del sistema
M = 2       # Masa del carro (kg)
m = 0.5     # Masa del péndulo (kg)
l = 1       # Longitud del péndulo (m)
g = 9.81    # Aceleración de la gravedad (m/s²)

# Definimos la función de transferencia de la planta
numerator = [1]             # Numerador de G(s)
denominator = [(M*l), 0, -(M+m)*g]
plant = TransferFunction(numerator, denominator)

# Parámetros del controlador PID
Kp = 523.40   # Ganancia proporcional300
Ki = 195.6    # Ganancia integral70
Kd = 654.24   # Ganancia derivativa10

# Función de transferencia del controlador PID
numerator_pid = [Kd, Kp, Ki]    # Numerador: Kd * s^2 + Kp * s + Ki
denominator_pid = [1, 0]        # Denominador: s
controller = TransferFunction(numerator_pid, denominator_pid)

# Lazo cerrado: C(s) * G(s) / (1 + C(s) * G(s))
# Multiplicamos controlador y planta
num_open = np.polymul(controller.num, plant.num)
den_open = np.polymul(controller.den, plant.den)

# Sumamos 1 al denominador para cerrar el lazo
den_closed = np.polyadd(num_open, den_open)
closed_loop = TransferFunction(num_open, den_closed)

# Tiempo de simulación
t = np.linspace(0, 10, 1000)

# Entrada (Setpoint: escalón unitario)
step_input = np.ones_like(t)

# Perturbación: un impulso en t=5 s
perturbation = np.zeros_like(t)
perturbation[np.argmin(np.abs(t - 5))] = 1  # Impulso en el instante t=5 s

# Simular respuesta del sistema para la entrada y la perturbación
_, y_step, _ = lsim(closed_loop, U=step_input, T=t)
_, y_perturbation, _ = lsim(closed_loop, U=perturbation, T=t)

# Combinar ambas señales (respuesta al escalón + efecto de perturbación)
response = y_step + y_perturbation

# Graficar resultados
plt.figure(figsize=(10, 6))
plt.plot(t, response, label="Salida del sistema (PID con perturbación)", color='blue')
plt.plot(t, step_input, 'r--', label="Referencia (Setpoint)")
plt.axvline(5, color='k', linestyle=':', label="Perturbación (impulso en t=5s)")
plt.title("Respuesta del sistema controlado por PID con perturbación")
plt.xlabel("Tiempo (s)")
plt.ylabel("Salida")
plt.legend()
plt.grid()
plt.show()

