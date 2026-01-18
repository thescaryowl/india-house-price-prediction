from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load the trained model
MODEL_PATH = 'models/model.pkl'
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
    else:
        print(f"❌ Model not found at {MODEL_PATH}")
        print("Please train the model first by running: python src/train.py")

@app.route('/')
def home():
    return jsonify({
        'message': 'India House Price Prediction API',
        'endpoints': {
            '/predict': 'POST - Make price predictions',
            '/health': 'GET - Check API health'
        },
        'model_loaded': model is not None
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.'
        }), 500
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Expected features (update based on your actual features)
        # These should match the training data columns (excluding target)
        expected_features = [
            'State', 'City', 'Property_Type', 'BHK', 'Size_in_SqFt',
            'Price_per_SqFt', 'Year_Built', 'Furnished_Status', 'Floor_No',
            'Total_Floors', 'Age_of_Property', 'Nearby_Schools',
            'Nearby_Hospitals', 'Public_Transport_Accessibility',
            'Parking_Space', 'Security', 'Facing', 'Owner_Type',
            'Availability_Status', 'Room_Size_Ratio', 'Floor_Position'
        ]
        
        # Create a dictionary with default values
        input_data = {}
        
        # Map user-friendly inputs to encoded values
        # Note: In production, you'd load the encoders and use them
        state_map = {'Andhra Pradesh': 0, 'Assam': 1, 'Bihar': 2, 'Chhattisgarh': 3, 
                     'Delhi': 4, 'Gujarat': 5, 'Haryana': 6, 'Jharkhand': 7, 
                     'Karnataka': 8, 'Kerala': 9, 'Madhya Pradesh': 10, 'Maharashtra': 11, 
                     'Odisha': 12, 'Punjab': 13, 'Rajasthan': 14, 'Tamil Nadu': 15, 
                     'Telangana': 16, 'Uttar Pradesh': 17, 'Uttarakhand': 18, 'West Bengal': 19}
        city_map = {'Ahmedabad': 0, 'Amritsar': 1, 'Bangalore': 2, 'Bhopal': 3, 'Bhubaneswar': 4,
                    'Bilaspur': 5, 'Chennai': 6, 'Coimbatore': 7, 'Cuttack': 8, 'Dehradun': 9,
                    'Durgapur': 10, 'Dwarka': 11, 'Faridabad': 12, 'Gaya': 13, 'Gurgaon': 14,
                    'Guwahati': 15, 'Haridwar': 16, 'Hyderabad': 17, 'Indore': 18, 'Jaipur': 19,
                    'Jamshedpur': 20, 'Jodhpur': 21, 'Kochi': 22, 'Kolkata': 23, 'Lucknow': 24,
                    'Ludhiana': 25, 'Mangalore': 26, 'Mumbai': 27, 'Mysore': 28, 'Nagpur': 29,
                    'New Delhi': 30, 'Noida': 31, 'Patna': 32, 'Pune': 33, 'Raipur': 34,
                    'Ranchi': 35, 'Silchar': 36, 'Surat': 37, 'Trivandrum': 38, 'Vijayawada': 39,
                    'Vishakhapatnam': 40, 'Warangal': 41}
        property_map = {'Apartment': 0, 'Independent House': 1, 'Villa': 2}
        furnished_map = {'Furnished': 0, 'Semi-furnished': 1, 'Unfurnished': 2}
        facing_map = {'North': 0, 'South': 1, 'East': 2, 'West': 3}
        owner_map = {'Owner': 0, 'Builder': 1, 'Broker': 2}
        availability_map = {'Ready_to_Move': 0, 'Under_Construction': 1}
        transport_map = {'High': 0, 'Low': 1, 'Medium': 2}
        
        # Parse input
        input_data['State'] = state_map.get(data.get('state', 'Maharashtra'), 1)
        input_data['City'] = city_map.get(data.get('city', 'Mumbai'), 10)
        input_data['Property_Type'] = property_map.get(data.get('property_type', 'Apartment'), 0)
        input_data['BHK'] = data.get('bhk', 2)
        input_data['Size_in_SqFt'] = data.get('size', 1000)
        
        # Calculate Price_per_SqFt (user can provide estimated market rate)
        input_data['Price_per_SqFt'] = data.get('price_per_sqft', 0.1)
        
        input_data['Year_Built'] = data.get('year_built', 2015)
        input_data['Furnished_Status'] = furnished_map.get(data.get('furnished_status', 'Semi-furnished'), 1)
        input_data['Floor_No'] = data.get('floor_no', 5)
        input_data['Total_Floors'] = data.get('total_floors', 10)
        input_data['Age_of_Property'] = 2025 - input_data['Year_Built']
        input_data['Nearby_Schools'] = data.get('nearby_schools', 5)
        input_data['Nearby_Hospitals'] = data.get('nearby_hospitals', 3)
        input_data['Public_Transport_Accessibility'] = transport_map.get(data.get('transport', 'Medium'), 2)
        input_data['Parking_Space'] = 1 if data.get('parking', False) else 0
        input_data['Security'] = 1 if data.get('security', False) else 0
        input_data['Facing'] = facing_map.get(data.get('facing', 'North'), 0)
        input_data['Owner_Type'] = owner_map.get(data.get('owner_type', 'Owner'), 0)
        input_data['Availability_Status'] = availability_map.get(data.get('availability', 'Ready_to_Move'), 0)
        
        # Calculate derived features
        input_data['Room_Size_Ratio'] = input_data['Size_in_SqFt'] / input_data['BHK']
        input_data['Floor_Position'] = input_data['Floor_No'] / (input_data['Total_Floors'] + 1)
        
        # Create DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        return jsonify({
            'predicted_price_lakhs': round(prediction, 2),
            'predicted_price_inr': round(prediction * 100000, 2),
            'model': 'Random Forest',
            'input_features': input_data
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Error making prediction. Please check your input data.'
        }), 400

if __name__ == '__main__':
    load_model()
    print("\n" + "="*60)
    print("🚀 Starting Flask API Server...")
    print("="*60)
    print("API will be available at: http://localhost:8000")
    print("Endpoints:")
    print("  - GET  /        : API info")
    print("  - GET  /health  : Health check")
    print("  - POST /predict : Make predictions")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=8000)