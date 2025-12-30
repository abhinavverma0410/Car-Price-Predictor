from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np
import logging
import requests

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
cors = CORS(app)

def get_euro_to_inr_rate():
    try:
        response = requests.get('https://api.exchangerate-api.com/v4/latest/EUR', timeout=5)
        data = response.json()
        return data['rates']['INR']
    except Exception as e:
        logger.warning(f"Failed to fetch exchange rate: {e}, using fallback rate 102")
        return 102  # Fallback rate

# Try to load the model
try:
    model = pickle.load(open("CarPricePrediction.pkl", 'rb'))
    logger.info("Model loaded successfully")
    logger.info(f"Model type: {type(model)}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

# Load the dataset
try:
    car = pd.read_csv("CleanedCarPrices.csv")
    logger.info("Dataset loaded successfully")
    
    # DEBUG: Print column names to see what we're working with
    logger.info(f"Dataset columns: {car.columns.tolist()}")
    logger.info(f"Dataset shape: {car.shape}")
    
except Exception as e:
    logger.error(f"Error loading dataset: {e}")
    # Set car to empty DataFrame if loading fails
    car = pd.DataFrame()

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        # DEBUG: Check what columns we have
        logger.info(f"Available columns: {car.columns.tolist()}")
        
        # Get unique values for dropdowns
        vehicle_type_col = 'VehicleType'
        brand_col = 'Brand'
        model_col = 'Model'
        
        # Get unique values using columns
        vehicle_types = sorted(car[vehicle_type_col].unique()) if vehicle_type_col in car.columns else []
        gearbox_types = sorted(car['Gearbox'].unique()) if 'Gearbox' in car.columns else []
        fuel_types = sorted(car['FuelType'].unique()) if 'FuelType' in car.columns else []
        brands = sorted(car[brand_col].unique()) if brand_col in car.columns else []
        models = sorted(car[model_col].unique()) if model_col in car.columns else []
        repaired_options = sorted(car['Repaired'].unique()) if 'Repaired' in car.columns else []

        # Create vehicle type -> brand -> model mapping
        vehicle_type_data = {}
        if vehicle_type_col in car.columns and brand_col in car.columns and model_col in car.columns:
            for vehicle_type in vehicle_types:
                vehicle_type_data[vehicle_type] = {}
                # Get brands for this vehicle type
                brands_for_type = car[car[vehicle_type_col] == vehicle_type][brand_col].unique()
                for brand in brands_for_type:
                    # Get models for this vehicle type and brand
                    models_for_brand = car[(car[vehicle_type_col] == vehicle_type) & 
                                            (car[brand_col] == brand)][model_col].unique()
                    vehicle_type_data[vehicle_type][brand] = sorted(models_for_brand.tolist())

        logger.info(f"Created mapping with {len(vehicle_type_data)} vehicle types")

        return render_template("index.html",
                                vehicle_types=vehicle_types,
                                gearbox_types=gearbox_types,
                                fuel_types=fuel_types,
                                brands=brands,
                                models=models,
                                repaired_options=repaired_options,
                                vehicle_type_data=vehicle_type_data)

    except Exception as e:
        logger.error(f"Error in index route: {e}")
        # Return empty data if there's an error
        return render_template("index.html",
                                vehicle_types=[],
                                gearbox_types=[],
                                fuel_types=[],
                                brands=[],
                                models=[],
                                repaired_options=[],
                                vehicle_type_data={})

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        # Get current exchange rate
        EURO_TO_INR_RATE = get_euro_to_inr_rate()
        logger.info(f"Using exchange rate: 1 EUR = {EURO_TO_INR_RATE} INR")
        
        if model is None:
            logger.warning("Model is None, using fallback prediction")
            # Fallback: return a dummy prediction
            power = float(request.form.get('power', 100))
            kilometer = float(request.form.get('kilometer', 50000))
            registration_year = 2020  # default
            
            # Try to get registration year from form
            registration_year_month = request.form.get('registration_year')
            if registration_year_month:
                try:
                    registration_year = int(registration_year_month.split('-')[0])
                except:
                    registration_year = 2020
            
            # More realistic dummy prediction formula
            base_price = 5000  # base price in EUR
            power_factor = power * 80  # higher power increases price
            km_factor = kilometer * -0.1  # more km decreases price
            age_factor = (2024 - registration_year) * -500  # older decreases price
            
            dummy_price_eur = base_price + power_factor + km_factor + age_factor
            dummy_price_eur = max(dummy_price_eur, 1000)  # minimum price
            dummy_price_inr = dummy_price_eur * EURO_TO_INR_RATE
            
            logger.info(f"Fallback prediction: {dummy_price_eur} EUR = {dummy_price_inr} INR")
            
            return jsonify({
                'price_eur': np.round(dummy_price_eur, 2),
                'price_inr': np.round(dummy_price_inr, 2),
                'exchange_rate': EURO_TO_INR_RATE,
                'message': 'Using fallback prediction (model not loaded)'
            })

        # Get form data with proper error handling
        try:
            vehicle_type = request.form.get('vehicle_type')
            gearbox = request.form.get('gearbox')
            fuel_type = request.form.get('fuel_type')
            power = float(request.form.get('power'))
            kilometer = float(request.form.get('kilometer'))
            brand = request.form.get('brand')
            model_name = request.form.get('model')
            repaired = request.form.get('repaired')
            postal_code = request.form.get('postal_code')
            registration_year_month = request.form.get('registration_year')
            
            # Convert year-month format to year
            if registration_year_month:
                registration_year = int(registration_year_month.split('-')[0])
            else:
                registration_year = 2020

            logger.info(f"Received prediction request: {vehicle_type}, {brand}, {model_name}, Power: {power}, KM: {kilometer}, Year: {registration_year}")

        except Exception as e:
            logger.error(f"Error parsing form data: {e}")
            return jsonify({
                'error': f"Invalid form data: {str(e)}"
            }), 400

        # Create input data for prediction
        input_data = pd.DataFrame({
            'VehicleType': [vehicle_type],
            'Gearbox': [gearbox],
            'FuelType': [fuel_type],
            'Power': [power],
            'Kilometer': [kilometer],
            'Brand': [brand],
            'Model': [model_name],
            'Repaired': [repaired],
            'PostalCode': [postal_code],
            'RegistrationYear': [registration_year]
        })

        logger.info(f"Input data for prediction: {input_data}")

        # Make prediction (this returns price in Euros)
        try:
            prediction_euros = model.predict(input_data)
            logger.info(f"Raw prediction result: {prediction_euros}")
            logger.info(f"Prediction type: {type(prediction_euros)}")
            logger.info(f"Prediction shape: {getattr(prediction_euros, 'shape', 'No shape attribute')}")
            
            # Handle different prediction output formats
            if hasattr(prediction_euros, '__len__') and len(prediction_euros) > 0:
                prediction_euros_value = prediction_euros[0]
            else:
                prediction_euros_value = float(prediction_euros)
                
        except Exception as e:
            logger.error(f"Error during model prediction: {e}")
            return jsonify({
                'error': f"Model prediction failed: {str(e)}"
            }), 500
        
        # Convert Euros to INR
        prediction_inr = prediction_euros_value * EURO_TO_INR_RATE

        logger.info(f"Prediction successful: {prediction_euros_value} Euros = {prediction_inr} INR")

        return jsonify({
            'price_eur': np.round(prediction_euros_value, 2),
            'price_inr': np.round(prediction_inr, 2),
            'exchange_rate': EURO_TO_INR_RATE,
            'message': 'Success'
        })

    except Exception as e:
        logger.error(f"Unexpected error in prediction: {e}", exc_info=True)
        return jsonify({
            'error': f"Unexpected error: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(debug=True)