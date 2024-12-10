from flask import Flask, request, jsonify, render_template
import numpy as np
import control as ctrl

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/simulate', methods=['POST'])
def simulate():
    # Recibir parámetros del usuario
    data = request.json
    Kp = float(data.get('Kp', 120))
    Ki = float(data.get('Ki', 160))
    Kd = float(data.get('Kd', 120))

    # Parámetros del sistema
    M = 2
    m = 0.5
    l = 1
    g = 9.81

    # Sistema G(s)
    numerador_G = [1]
    denominador_G = [(M*l), 0, -(M+m)*g]
    G = ctrl.TransferFunction(numerador_G, denominador_G)

    # Controlador PID
    C_PID = ctrl.TransferFunction([Kd, Kp, Ki], [1, 0])
    T_PID = ctrl.feedback(G * C_PID)

    # Simulación
    t = np.linspace(0, 5, 1000)
    _, yout_PID = ctrl.step_response(T_PID, t)

    # Enviar datos al cliente
    return jsonify({
        'time': t.tolist(),
        'response': yout_PID.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
