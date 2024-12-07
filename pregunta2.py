import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

# Parámetros del sistema
M = 0.8  # Masa del carro (kg)
m = 0.23  # Masa del péndulo (kg)
L = 0.3  # Longitud del péndulo (m)
g = 9.81  # Aceleración de la gravedad (m/s²)

# Título de la aplicación
st.title("Controladores P, PI, PD y PID")

# Sliders para Kp, Ki y Kd
Kp = st.slider("Kp (Proporcional)", min_value=0.0, max_value=200.0, value=80.0)
Ki = st.slider("Ki (Integral)", min_value=0.0, max_value=200.0, value=80.0)
Kd = st.slider("Kd (Derivativo)", min_value=0.0, max_value=200.0, value=3.0)

# Sistema G(s)
numerador_G = [1]
denominador_G = [M * L, 0, -(M + m) * g]
G = ctrl.TransferFunction(numerador_G, denominador_G)

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
t = np.linspace(0, 1, 1000)

# Respuestas al escalón
_, yout_P = ctrl.step_response(T_P, t)
_, yout_PI = ctrl.step_response(T_PI, t)
_, yout_PD = ctrl.step_response(T_PD, t)
_, yout_PID = ctrl.step_response(T_PID, t)

# Graficar las respuestas
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(t, yout_P, label="Controlador P")
ax.plot(t, yout_PI, label="Controlador PI")
ax.plot(t, yout_PD, label="Controlador PD")
ax.plot(t, yout_PID, label="Controlador PID")

ax.set_title("Respuestas al Escalón de Controladores P, PI, PD y PID")
ax.set_xlabel("Tiempo (s)")
ax.set_ylabel("Salida")
ax.legend()
ax.grid(True)

# Mostrar el gráfico
st.pyplot(fig)