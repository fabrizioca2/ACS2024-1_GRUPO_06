import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

# Parámetros del sistema
M = 2  # Masa del carro (kg)
m = 0.5  # Masa del péndulo (kg)
l = 1  # Longitud del péndulo (m)
g = 9.81 # Aceleración de la gravedad (m/s²)

# Definir las ganancias
Kp = 30    # Ganancia proporcional
Ki = 30  # Ganancia integral
Kd = 30  # Ganancia derivativa

# Definir las ganancias
Kp_ = 37.985    # Ganancia proporcional
Ki_ = 16.4227  # Ganancia integral
Kd_ = 9.780  # Ganancia derivativa

# Sistema G(s)
numerador_G = [1]
denominador_G = [(M*l), 0, -(M+m)*g]
G = ctrl.TransferFunction(numerador_G, denominador_G)
print(G)
# Controladores PID
C_PID = ctrl.TransferFunction([Kd, Kp, Ki], [1, 0])
C_PID_ = ctrl.TransferFunction([Kd_, Kp_, Ki_], [1, 0])

# Lazo cerrado
T_PID = ctrl.feedback(G * C_PID)
T_PID_ = ctrl.feedback(G * C_PID_)

# Tiempo de simulación
t = np.linspace(0, 10, 1000)

# Respuestas al escalón
time_PID, yout_PID = ctrl.step_response(T_PID, t)
time_PID_, yout_PID_ = ctrl.step_response(T_PID_, t)

# Graficar las respuestas
plt.figure(figsize=(10, 6))
plt.plot(time_PID, yout_PID, label="Controlador PID no ajustado")
plt.plot(time_PID_, yout_PID_, label="Controlador PID ajustado por algortimos genéticos")
plt.title("Respuesta al Escalón del Controlador PID")
plt.xlabel("Tiempo (s)")
plt.ylabel("Salida")
plt.axhline(y=1, color='r', linestyle='--', label="Referencia (setpoint)")
plt.legend()
plt.grid(True)
plt.ylim(-2, 2)
plt.show()
