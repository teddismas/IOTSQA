from flask import Flask, request, jsonify
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Conv1D, MaxPooling1D, Flatten
from sklearn.metrics import mean_squared_error, mean_absolute_error
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

app = Flask(__name__)

# Load the logistic regression model for crop prediction
MODEL_PATH = os.getenv('MODEL_PATH')
with open(MODEL_PATH, 'rb') as file:
    logistic_model = pickle.load(file)

# Define the DQN model with Bi-LSTM for soil quality prediction
time_steps = 7  # Adjust as per your data
num_features = 1  # Number of features (N, P, K, temperature, humidity, ph, rainfall)

class DQN_BiLSTM:
    def __init__(self, input_shape):
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=input_shape))
        self.model.add(Bidirectional(LSTM(50)))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train(self, X, y, epochs=10, batch_size=32):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)

    def predict(self, X):
        return self.model.predict(X)

dqn_model = DQN_BiLSTM((time_steps, num_features))

# Define RNN Model
class RNN_Model:
    def __init__(self, input_shape):
        self.model = Sequential()
        self.model.add(LSTM(50, input_shape=input_shape))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train(self, X, y, epochs=10, batch_size=32):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)

    def predict(self, X):
        return self.model.predict(X)

rnn_model = RNN_Model((time_steps, num_features))
# Optionally load pretrained weights
# rnn_model.model.load_weights('rnn_model_weights.h5')

# Define CNN Model
class CNN_Model:
    def __init__(self, input_shape):
        self.model = Sequential()
        self.model.add(Conv1D(64, kernel_size=2, activation='relu', input_shape=input_shape))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Flatten())
        self.model.add(Dense(50, activation='relu'))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train(self, X, y, epochs=10, batch_size=32):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)

    def predict(self, X):
        return self.model.predict(X)

cnn_model = CNN_Model((time_steps, num_features))
# Optionally load pretrained weights
# cnn_model.model.load_weights('cnn_model_weights.h5')

# Function to calculate evaluation metrics
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    return rmse, mae, mape

# Function to calculate EC based on NPK values
def calculate_ec(n, p, k):
    n_contribution = 0.00014
    p_contribution = 0.0001
    k_contribution = 0.0012
    n_ec = n * n_contribution
    p_ec = p * p_contribution
    k_ec = k * k_contribution
    total_ec = n_ec + p_ec + k_ec
    return total_ec

# Function to determine soil health and provide improvement suggestions
def determine_soil_health_and_improvements(n, p, k, ec):
    improvements = []
    ranges = {
        'Nitrogen': (0.2, 0.4),
        'Phosphorus': (0.3, 0.5),
        'Potassium': (0.15, 0.2),
        'Electrical Conductivity': (0.15, 0.3)
    }
    soil_health_status = {
        'Nitrogen': n,
        'Phosphorus': p,
        'Potassium': k,
        'Electrical Conductivity': ec
    }
    
    if not (ranges['Nitrogen'][0] <= n <= ranges['Nitrogen'][1]):
        improvements.append("Add nitrogen-rich fertilizer to increase nitrogen levels.")
    if not (ranges['Phosphorus'][0] <= p <= ranges['Phosphorus'][1]):
        improvements.append("Add phosphorus-rich fertilizer to increase phosphorus levels.")
    if not (ranges['Potassium'][0] <= k <= ranges['Potassium'][1]):
        improvements.append("Add potassium-rich fertilizer to increase potassium levels.")
    
    if ranges['Nitrogen'][0] <= n <= ranges['Nitrogen'][1] and ranges['Phosphorus'][0] <= p <= ranges['Phosphorus'][1] and ranges['Potassium'][0] <= k <= ranges['Potassium'][1]:
        if ec < ranges['Electrical Conductivity'][0]:
            soil_health = "bad"
        elif ranges['Electrical Conductivity'][0] <= ec <= ranges['Electrical Conductivity'][1]:
            soil_health = "moderate"
        else:
            soil_health = "good"
    else:
        soil_health = "bad"
    
    return soil_health, soil_health_status, improvements

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing field: {field}'}), 400
    
    n = data['N']
    p = data['P']
    k = data['K']
    temperature = data['temperature']
    humidity = data['humidity']
    ph = data['ph']
    rainfall = data['rainfall']
    
    if not (20 <= n <= 40):
        return jsonify({"error": "Nitrogen value must be between 20 and 40 ppm"}), 400
    if not (30 <= p <= 50):
        return jsonify({"error": "Phosphorus value must be between 30 and 50 ppm"}), 400
    if not (150 <= k <= 200):
        return jsonify({"error": "Potassium value must be between 150 and 200 ppm"}), 400
    
    feature_array = np.array([n, p, k, temperature, humidity, ph, rainfall]).reshape(1, time_steps, num_features)
    
    # Predict the crop using the logistic regression model
    crop_prediction = logistic_model.predict(feature_array.reshape(1, -1))
    
    soil_quality_dqn = dqn_model.predict(feature_array)
    soil_quality_rnn = rnn_model.predict(feature_array)
    soil_quality_cnn = cnn_model.predict(feature_array)
    
    # Calculate the EC
    ec = calculate_ec(n, p, k)
    
    # Determine soil health and get improvement suggestions
    soil_health, soil_health_status, improvements = determine_soil_health_and_improvements(n, p, k, ec)
    
    # Convert any float32 to Python float
    soil_quality_dqn = float(soil_quality_dqn[0][0])
    soil_quality_rnn = float(soil_quality_rnn[0][0])
    soil_quality_cnn = float(soil_quality_cnn[0][0])
    
    # Return the predicted crop, EC, soil quality, soil health, and improvements as a JSON response
    return jsonify({
        'predicted_crop': crop_prediction[0],
        'electrical_conductivity': ec,
        'soil_quality_dqn': soil_quality_dqn,
        'soil_quality_rnn': soil_quality_rnn,
        'soil_quality_cnn': soil_quality_cnn,
        'soil_health': soil_health,
        'soil_health_status': soil_health_status,
                'improvements': improvements
    })

if __name__ == '__main__':
    flask_host = os.getenv('FLASK_HOST')
    flask_port = int(os.getenv('FLASK_PORT'))
    flask_debug = os.getenv('FLASK_DEBUG').lower() == 'true'
    app.run(host=flask_host, port=flask_port, debug=flask_debug)

