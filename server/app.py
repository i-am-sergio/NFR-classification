from flask import Flask, request, jsonify
from flask_cors import CORS
from prediction import Prediction

# Server
app = Flask(__name__)
CORS(app)

# Clasificador
predictor = Prediction(12,"LR")

@app.route('/predict12', methods=['POST'])
def predict_12class():
    return predict(12)

@app.route('/predict2', methods=['POST'])
def predict_2class():
    return predict(2)

def predict(num_classes):
    try:
        data = request.get_json()

        if not data or not isinstance(data, dict) or 'message' not in data:
            return jsonify({'error': 'Se requiere el campo "message" en la solicitud'}), 400

        message = data['message']
        selected_model = data.get('selectedModel')  # Obtiene el modelo seleccionado, si está presente
        selected_language = data.get('selectedLanguage')  # Obtiene el idioma seleccionado, si está presente

        print(f"Mensaje recibido: {message}")
        print(f"Modelo seleccionado: {selected_model}")
        print(f"Idioma seleccionado: {selected_language}")

        # Aquí puedes usar 'message', 'selected_model' y 'selected_language' para tu lógica de predicción

        # Ejemplo de cómo usar los datos:
        if selected_model == 'modelo_a':
            # Lógica para el modelo A
            prediction_result = f"Predicción del modelo A para: {message}"
        elif selected_model == 'modelo_b':
            # Lógica para el modelo B
            prediction_result = f"Predicción del modelo B para: {message}"
        else:
            # Lógica por defecto si no se selecciona un modelo específico
            predicted_class = predictor.process_requirement(message)
            prediction_result = f"[{predicted_class}: {predictor.get_name_of_class(predicted_class)}]"

        response = {
            'modelUsed': f'English - LR - {num_classes} clases',
            'predictionResult' : prediction_result,
            'predictionDetails' : {
                "F" : "12%",
                "O" : "16%",
                "PE" : "98%" 
            }
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)