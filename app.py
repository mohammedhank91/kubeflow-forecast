from flask import Flask, request, jsonify
from function import *
import numpy as np
import os 
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get JSON data from the request
    input_data = data.get('input_data')
    version = data.get('version')

    # Call your forecast function here with the input_data and version
    try:
        predictions = forecast(
            X_train=X_train,
            y_train=y_train,
            batch_size=batch_size,
            epochs=epochs,
            n_neurons=n_neurons,
            learning_rate=learning_rate,
            cycle=cycle,
            n_cycles=n_cycles,
            full_data=full_data,
            length=length,
            data_copy=data_copy,
            folder=folder,
            version=version,
        )

        return jsonify({'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=8090)
