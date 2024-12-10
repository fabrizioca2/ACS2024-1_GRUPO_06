import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lsim, tf2ss, TransferFunction, StateSpace

# Parámetros del sistema
M = 2       # Masa del carro (kg)
m = 0.5     # Masa del péndulo (kg)
l = 1       # Longitud del péndulo (m)
g = 9.81    # Aceleración de la gravedad (m/s²)

# Definir la función de transferencia de la planta
numerator = [1]             # Numerador de G(s)
denominator = [(M*l), 0, -(M+m)*g]
plant = TransferFunction(numerator, denominator)

# Parámetros del controlador PID
Kp = 265.20   # Ganancia proporcional
Ki = 34.39    # Ganancia integral
Kd = 81.34    # Ganancia derivativa

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

# Convertir la función de transferencia a espacio de estados
A, B, C, D = tf2ss(num_open, den_closed)
system_ss = StateSpace(A, B, C, D)

# Inspeccionar dimensiones de A para ajustar x0
print("Matriz A:", A)
print("Dimensión de A:", A.shape)

# Tiempo de simulación
t = np.linspace(0, 10, 1000)

# Ajustar condiciones iniciales según la dimensión del sistema
x0 = [0.1, 0, 0]  # Dimensión debe coincidir con la cantidad de estados en A

# Simular respuesta del sistema con condiciones iniciales
_, y, x_states = lsim(system_ss, U=0, T=t, X0=x0)

# Graficar resultados
plt.figure(figsize=(10, 6))
plt.plot(t, y, label="Salida del sistema con condiciones iniciales", color='blue')
plt.axhline(0, color='red', linestyle='--', label="Referencia (Setpoint)")
plt.title("Respuesta del sistema controlado por PID con condiciones iniciales")
plt.xlabel("Tiempo (s)")
plt.ylabel("Salida (posición)")
plt.legend()
plt.grid()
plt.show()