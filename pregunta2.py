import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import control as ctrl

# Parámetros del sistema
M = 0.5  # Masa del carro (kg)
m = 0.2  # Masa del péndulo (kg)
L = 0.3  # Longitud del péndulo (m)
g = 9.81 # Aceleración de la gravedad (m/s²)

# Función que actualiza el gráfico
def update_plot():
    # Obtener los valores de los sliders
    Kp = slider_Kp.get()
    Ki = slider_Ki.get()
    Kd = slider_Kd.get()

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
    time_P, yout_P = ctrl.step_response(T_P, t)
    time_PI, yout_PI = ctrl.step_response(T_PI, t)
    time_PD, yout_PD = ctrl.step_response(T_PD, t)
    time_PID, yout_PID = ctrl.step_response(T_PID, t)

    # Limpiar el gráfico antes de dibujar uno nuevo
    ax.clear()

    # Graficar las respuestas
    ax.plot(time_P, yout_P, label="Controlador P")
    ax.plot(time_PI, yout_PI, label="Controlador PI")
    ax.plot(time_PD, yout_PD, label="Controlador PD")
    ax.plot(time_PID, yout_PID, label="Controlador PID")
    
    ax.set_title("Respuestas al Escalón de Controladores P, PI, PD y PID")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Salida")
    ax.legend()
    ax.grid(True)

    # Actualizar el gráfico en la ventana de Tkinter
    canvas.draw()

# Crear la ventana principal
root = tk.Tk()
root.title("Controladores P, PI, PD, PID")

# Crear los sliders para Kp, Ki, Kd
slider_Kp = tk.Scale(root, from_=0, to_=200, orient="horizontal", label="Kp", length=400)
slider_Kp.set(80)  # Valor inicial
slider_Kp.pack()

slider_Ki = tk.Scale(root, from_=0, to_=200, orient="horizontal", label="Ki", length=400)
slider_Ki.set(80)  # Valor inicial
slider_Ki.pack()

slider_Kd = tk.Scale(root, from_=0, to_=20, orient="horizontal", label="Kd", length=400)
slider_Kd.set(3)  # Valor inicial
slider_Kd.pack()

# Crear la figura de Matplotlib y agregarla a la ventana de Tkinter
fig, ax = plt.subplots(figsize=(10, 6))

# Crear el canvas de Matplotlib y agregarlo a la ventana de Tkinter
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

# Botón para actualizar el gráfico
button = tk.Button(root, text="Actualizar Gráfico", command=update_plot)
button.pack()

# Ejecutar la ventana de Tkinter
root.mainloop()