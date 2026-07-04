import os
import joblib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Global model container
model_data = None

def load_model():
    global model_data
    model_path = os.path.join(os.path.dirname(__file__), "car_price_model.pkl")
    if os.path.exists(model_path):
        try:
            model_data = joblib.load(model_path)
            print("Model loaded successfully from:", model_path)
        except Exception as e:
            print("Error loading model:", e)
    else:
        print("Warning: car_price_model.pkl not found. Run train_model.py first.")

# Load model at startup
load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model_data
    # Reload model if it wasn't loaded successfully before
    if model_data is None:
        load_model()
        if model_data is None:
            return jsonify({'error': 'Prediction model is currently unavailable. Please train the model.'}), 500
            
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No input data provided.'}), 400
            
        # Parse features
        present_price = float(data.get('present_price', 0))
        kms_driven = float(data.get('kms_driven', 0))
        fuel_type = int(data.get('fuel_type', 0))
        seller_type = int(data.get('seller_type', 0))
        transmission = int(data.get('transmission', 0))
        owner = int(data.get('owner', 0))
        age = float(data.get('age', 0))
        
        # Prepare input vector
        features = [[present_price, kms_driven, fuel_type, seller_type, transmission, owner, age]]
        
        # Predict using loaded Scikit-Learn RandomForest model
        prediction = model_data['model'].predict(features)[0]
        
        # Avoid any potential negative predictions (clip at 0)
        prediction = max(0.0, prediction)
        
        return jsonify({
            'success': True,
            'prediction': round(prediction, 2)
        })
        
    except ValueError as val_err:
        return jsonify({'error': f'Invalid input format: {str(val_err)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    # Start the server on port 5000
    app.run(debug=True, host='0.0.0.0', port=5000)
