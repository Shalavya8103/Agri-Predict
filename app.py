from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import pickle
from PIL import Image
from tensorflow.keras.models import load_model
from keras_preprocessing.image import img_to_array
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PowerTransformer
import pandas as pd

app = Flask(__name__)

# Load models and transformers
try:
    # Load the disease detection model and label binarizer
    disease_model = load_model('ML/mod2.h5')
    label_binarizer = pickle.load(open('ML/label_transform.pkl', 'rb'))

    # Load the yield prediction model and transformer
    yield_model = pickle.load(open('ML/rf_yield_model.pkl', 'rb'))
    yield_transformer = pickle.load(open('ML/yield_transformer.pkl', 'rb'))
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading model or transformer: {e}")

# Set default image size for disease detection
default_image_size = (256, 256)

# Helper function for disease detection
def convert_image_to_array(image):
    try:
        image = cv2.resize(image, default_image_size)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        return image / 255.0
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Disease Detection Route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        try:
            image = Image.open(file)
            image = image.convert('RGB')  
            image = np.array(image)
            image_array = convert_image_to_array(image)

            if image_array is None:
                return jsonify({'error': 'Error processing image'}), 400

            predictions = disease_model.predict(image_array)
            predicted_class = label_binarizer.classes_[np.argmax(predictions)]

            return jsonify({'prediction': predicted_class})

        except Exception as e:
            print(f"Error during prediction: {e}")
            return jsonify({'error': 'Error processing image for prediction'}), 500
    return jsonify({'error': 'Invalid file format'}), 400


# Yield Prediction Route
@app.route('/predict_yield', methods=['POST'])
def predict_yield():
    try:
        # Extract data from the JSON request
        data = request.get_json()
        crop = data.get('crop')
        season = data.get('season')
        state = data.get('state')
        area = float(data.get('area'))
        production = float(data.get('production'))
        annual_rainfall = float(data.get('annualRainfall'))
        fertilizer = float(data.get('fertilizer'))
        pesticide = float(data.get('pesticide'))

        # Prepare data as a DataFrame to match model input format
        input_data = pd.DataFrame([[crop, season, state, area, production, annual_rainfall, fertilizer, pesticide]],
                                  columns=['Crop', 'Season', 'State', 'Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide'])

        # One-hot encoding to match training data format
        input_data = pd.get_dummies(input_data, drop_first=True)
        
        # Ensure input data matches model features (add missing columns with 0)
        missing_cols = set(yield_transformer.feature_names_in_) - set(input_data.columns)
        for col in missing_cols:
            input_data[col] = 0
        input_data = input_data[yield_transformer.feature_names_in_]

        # Transform data using the fitted transformer
        transformed_data = yield_transformer.transform(input_data)

        # Predict yield
        yield_prediction = yield_model.predict(transformed_data)[0]
        return jsonify({'prediction': yield_prediction})
    except Exception as e:
        print(f"Error in yield prediction: {e}")
        return jsonify({'error': f'Error in yield prediction: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)  
