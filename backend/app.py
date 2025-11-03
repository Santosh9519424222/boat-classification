"""
Backend Application for Boat Type Classification
=================================================
This Flask application serves as the backend for our boat classifier.
It loads the trained model and provides an API endpoint for predictions.
"""

# Import necessary libraries
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io

# Initialize the Flask application
app = Flask(__name__)

# Enable CORS (Cross-Origin Resource Sharing) to allow the frontend to communicate with the backend
CORS(app)

# --- Configuration ---
# Define the path to the saved model
MODEL_PATH = 'boat_classifier_mobilenet.h5'

# --- Model Loading ---
# Load the trained model when the server starts
# This is done once at startup to avoid loading the model on every request
try:
    model = load_model(MODEL_PATH)
    print(f"✓ Model loaded successfully from {MODEL_PATH}")
    print(f"✓ Model is ready to make predictions")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None

# --- Define Class Names ---
# These are the boat types our model can identify
# They must be in the same order as during training
CLASS_NAMES = [
    'buoy',           # Floating markers in water
    'cruise_ship',    # Large passenger ships
    'ferry_boat',     # Boats that transport people/vehicles
    'freight_boat',   # Cargo ships
    'gondola',        # Traditional Venetian boats
    'inflatable_boat',# Rubber boats
    'kayak',          # Small paddle boats
    'paper_boat',     # Origami-style boats
    'sailboat'        # Boats with sails
]

def preprocess_image(img_bytes):
    """
    Preprocess an image for model prediction.
    
    This function performs the following steps:
    1. Loads the image from bytes into a PIL format
    2. Resizes the image to 224x224 pixels (required by MobileNetV2)
    3. Converts the image to a NumPy array
    4. Adds a batch dimension (the model expects a batch of images)
    5. Rescales pixel values from [0, 255] to [0, 1]
    
    Parameters:
    -----------
    img_bytes : bytes
        The raw image data in bytes format
    
    Returns:
    --------
    numpy.ndarray
        Preprocessed image array ready for model prediction
    """
    # Load the image from bytes and resize it to 224x224 pixels
    img = image.load_img(io.BytesIO(img_bytes), target_size=(224, 224))
    
    # Convert the PIL image to a numpy array
    img_array = image.img_to_array(img)
    
    # Add a batch dimension (model expects shape: [batch_size, height, width, channels])
    img_array_expanded = np.expand_dims(img_array, axis=0)
    
    # Rescale pixel values from [0, 255] to [0, 1]
    # This matches the preprocessing used during training
    return img_array_expanded / 255.0

# --- API Routes ---

@app.route('/', methods=['GET'])
def home():
    """
    Home endpoint - provides information about the API.
    This is useful for checking if the server is running.
    """
    return jsonify({
        'message': 'Boat Type Classification API',
        'status': 'running',
        'model_loaded': model is not None,
        'endpoints': {
            '/': 'API information',
            '/predict': 'POST - Upload an image to get boat type prediction'
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint - receives an image and returns the predicted boat type.
    
    This function:
    1. Checks if the model is loaded
    2. Validates the uploaded file
    3. Preprocesses the image
    4. Makes a prediction using the model
    5. Returns the predicted class and confidence score
    
    Returns:
    --------
    JSON response with:
        - prediction: The predicted boat type
        - confidence: How confident the model is (as a percentage)
        - all_predictions: Confidence scores for all classes
    """
    # Check if the model is loaded
    if model is None:
        return jsonify({
            'error': 'Model is not loaded. Please check the server logs.'
        }), 500

    # Check if a file was included in the request
    if 'file' not in request.files:
        return jsonify({
            'error': 'No file provided. Please upload an image.'
        }), 400
    
    file = request.files['file']

    # Check if the user selected a file
    if file.filename == '':
        return jsonify({
            'error': 'No file selected. Please choose an image file.'
        }), 400

    try:
        # Read the image file as bytes
        img_bytes = file.read()
        
        # Preprocess the image
        processed_image = preprocess_image(img_bytes)
        
        # Make a prediction using the model
        # The model returns probabilities for each class
        prediction = model.predict(processed_image, verbose=0)
        
        # Get the index of the class with the highest probability
        predicted_class_index = np.argmax(prediction[0])
        
        # Get the corresponding class name
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        
        # Get the confidence score (probability of the predicted class)
        confidence = float(prediction[0][predicted_class_index])
        
        # Create a dictionary of all predictions
        all_predictions = {
            CLASS_NAMES[i]: float(prediction[0][i]) 
            for i in range(len(CLASS_NAMES))
        }
        
        # Sort predictions by confidence (highest first)
        sorted_predictions = dict(
            sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
        )
        
        # Return the results as JSON
        return jsonify({
            'success': True,
            'prediction': predicted_class_name,
            'confidence': f'{confidence:.2%}',
            'confidence_raw': confidence,
            'all_predictions': sorted_predictions
        })
    
    except Exception as e:
        # Handle any errors that occur during prediction
        return jsonify({
            'error': f'Error processing image: {str(e)}'
        }), 500

# --- Health Check Endpoint ---
@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint - useful for monitoring the server status.
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

# --- Main Execution ---
if __name__ == '__main__':
    """
    Start the Flask application.
    
    Configuration:
    - host='0.0.0.0': Makes the server accessible from other devices on the network
    - port=5000: The port number the server will listen on
    - debug=True: Enables debug mode (auto-reload on code changes)
    """
    print("\n" + "="*60)
    print("🚢 Boat Type Classification Backend Server")
    print("="*60)
    print(f"Server starting on http://localhost:5000")
    print(f"Model status: {'✓ Loaded' if model else '✗ Not loaded'}")
    print("="*60 + "\n")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
