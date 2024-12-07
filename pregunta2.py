import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

# Parámetros del sistema
M = 2  # Masa del carro (kg)
m = 0.5  # Masa del péndulo (kg)
l = 1  # Longitud del péndulo (m)
g = 9.81 # Aceleración de la gravedad (m/s²)


# Definir las ganancias
Kp = 120  # Ganancia proporcional
Ki = 160 # Ganancia integral
Kd = 120 # Ganancia derivativa

# Sistema G(s)
numerador_G = [1]
denominador_G = [(M*l), 0, -(M+m)*g]
G = ctrl.TransferFunction(numerador_G, denominador_G)
print(G)
# Controladores P, PI, PD, PID
C_P = ctrl.TransferFunction([Kp], [1])
C_PI = ctrl.TransferFunction([Kp, Ki], [1, 0])
C_PD = ctrl.TransferFunction([Kd, Kp], [1])
C_PID = ctrl.TransferFunction([Kd, Kp, Ki], [1, 0])

# Lazo cerrado
T_P = ctrl.feedback(G * C_P)
T_PI = ctrl.feedback(G * C_PI)
T_PD = ctrl.feedback(G * C_PD)
T_PID = ctrl.feedback(G * C_PID)

# Tiempo de simulación
t = np.linspace(0, 5, 1000)

# Respuestas al escalón
time_P, yout_P = ctrl.step_response(T_P, t)
time_PI, yout_PI = ctrl.step_response(T_PI, t)
time_PD, yout_PD = ctrl.step_response(T_PD, t)
time_PID, yout_PID = ctrl.step_response(T_PID, t)

# Graficar las respuestas
plt.figure(figsize=(10, 6))
plt.plot(time_P, yout_P, label="Controlador P")
plt.plot(time_PI, yout_PI, label="Controlador PI")
plt.plot(time_PD, yout_PD, label="Controlador PD")
plt.plot(time_PID, yout_PID, label="Controlador PID")
plt.title("Respuestas al Escalón de Controladores P, PI, PD y PID")
plt.xlabel("Tiempo (s)")
plt.ylabel("Salida")
plt.axhline(y=1, color='r', linestyle='--', label="Referencia (setpoint)")
plt.legend()
plt.grid(True)
plt.ylim(-5, 5)
plt.show()