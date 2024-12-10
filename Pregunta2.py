import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Parámetros del sistema
M = 2  # Masa del carro (kg)
m = 0.5  # Masa del péndulo (kg)
l = 1  # Longitud del péndulo (m)
g = 9.81  # Aceleración de la gravedad (m/s²)

# Sistema G(s)
numerador_G = [1]
denominador_G = [(M * l), 0, -(M + m) * g]
G = ctrl.TransferFunction(numerador_G, denominador_G)

# Función para actualizar el gráfico
def update_plot():
    # Obtener valores de las barras deslizantes
    Kp = slider_kp.get()
    Ki = slider_ki.get()
    Kd = slider_kd.get()

    # Definir los controladores
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

    # Limpiar el gráfico actual
    ax.clear()

    # Graficar las respuestas
    ax.plot(time_P, yout_P, label="Controlador P")
    ax.plot(time_PI, yout_PI, label="Controlador PI")
    ax.plot(time_PD, yout_PD, label="Controlador PD")
    ax.plot(time_PID, yout_PID, label="Controlador PID")

    # Personalizar gráfico
    ax.axhline(y=1, color='r', linestyle='--', label="Referencia (setpoint)")
    ax.set_title("Respuestas al Escalón de Controladores P, PI, PD y PID")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Salida")
    ax.legend()
    ax.grid(True)
    ax.set_ylim(-5, 5)

    # Dibujar el gráfico actualizado
    canvas.draw()

# Crear la ventana principal
root = tk.Tk()
root.title("Simulador de Controladores PID")

# Crear un marco para los sliders
frame_controls = tk.Frame(root)
frame_controls.pack(side=tk.LEFT, padx=10, pady=10)

# Crear sliders para Kp, Ki, Kd
slider_kp = tk.Scale(frame_controls, from_=0, to_=500, resolution=1, orient=tk.HORIZONTAL, label="Kp")
slider_kp.set(120)
slider_kp.pack()

slider_ki = tk.Scale(frame_controls, from_=0, to_=500, resolution=1, orient=tk.HORIZONTAL, label="Ki")
slider_ki.set(160)
slider_ki.pack()

slider_kd = tk.Scale(frame_controls, from_=0, to_=500, resolution=1, orient=tk.HORIZONTAL, label="Kd")
slider_kd.set(120)
slider_kd.pack()

# Botón para actualizar el gráfico
btn_update = tk.Button(frame_controls, text="Actualizar Gráfico", command=update_plot)
btn_update.pack(pady=10)

# Crear el gráfico
fig, ax = plt.subplots(figsize=(8, 6))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Mostrar la ventana principal
update_plot()  # Mostrar el gráfico inicial
root.mainloop()
