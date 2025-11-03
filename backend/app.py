"""
🚢 Boat Type Classification - Backend API
==========================================
Flask REST API for serving boat classification predictions.

Features:
- Loads trained MobileNetV2 model on startup
- Accepts image uploads via POST request
- Returns predicted boat type with confidence scores
- CORS enabled for frontend communication

Author: Boat Classification Project
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io

# ============================================================================
# FLASK APPLICATION SETUP
# ============================================================================

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for frontend communication

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = 'boat_classifier_mobilenet.h5'  # Path to the trained model file

# Define boat type classes (MUST match training order)
CLASS_NAMES = [
    'buoy',            # Floating water markers
    'cruise_ship',     # Large passenger vessels
    'ferry_boat',      # Passenger/vehicle transport boats
    'freight_boat',    # Cargo ships
    'gondola',         # Traditional Venetian boats
    'inflatable_boat', # Rubber/inflatable boats
    'kayak',           # Small paddle-powered boats
    'paper_boat',      # Origami-style paper boats
    'sailboat'         # Wind-powered sail boats
]

# ============================================================================
# MODEL LOADING
# ============================================================================

try:
    # Load the trained model once at server startup (not on each request)
    model = load_model(MODEL_PATH)
    print(f"✓ Model loaded successfully from: {MODEL_PATH}")
    print(f"✓ Ready to classify {len(CLASS_NAMES)} boat types")
except FileNotFoundError:
    print(f"✗ ERROR: Model file not found at: {MODEL_PATH}")
    print("  Please run the training notebook and move the .h5 file to backend/")
    model = None
except Exception as e:
    print(f"✗ ERROR loading model: {str(e)}")
    model = None

# ============================================================================
# IMAGE PREPROCESSING FUNCTION
# ============================================================================

def preprocess_image(img_bytes):
    """
    Prepare an uploaded image for model prediction.
    
    Process:
    1. Load image from raw bytes
    2. Resize to 224x224 (MobileNetV2 input size)
    3. Convert to NumPy array
    4. Add batch dimension [1, 224, 224, 3]
    5. Normalize pixel values to [0, 1] range
    
    Args:
        img_bytes (bytes): Raw image data from upload
    
    Returns:
        np.ndarray: Preprocessed image array shape (1, 224, 224, 3)
    """
    # Step 1: Load and resize image to model's expected input size
    img = image.load_img(io.BytesIO(img_bytes), target_size=(224, 224))
    
    # Step 2: Convert PIL image to NumPy array [224, 224, 3]
    img_array = image.img_to_array(img)
    
    # Step 3: Add batch dimension [1, 224, 224, 3]
    # Model expects batches, even if we're only predicting one image
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Step 4: Normalize pixel values from [0-255] to [0-1]
    # This matches the rescaling used during training
    normalized_img = img_batch / 255.0
    
    return normalized_img

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/', methods=['GET'])
def home():
    """
    Home endpoint - API information and status check.
    
    Returns:
        JSON with API details and model status
    """
    return jsonify({
        'message': 'Boat Type Classification API v1.0',
        'status': 'running',
        'model_loaded': model is not None,
        'supported_classes': len(CLASS_NAMES),
        'endpoints': {
            'GET /': 'API information',
            'POST /predict': 'Upload image for classification',
            'GET /health': 'Server health check'
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint - classifies uploaded boat images.
    
    Request:
        - Method: POST
        - Content-Type: multipart/form-data
        - Body: 'file' field with image (jpg/png)
    
    Response:
        JSON with:
        - success: True/False
        - prediction: Predicted boat type (string)
        - confidence: Confidence as percentage (string)
        - confidence_raw: Raw confidence score 0-1 (float)
        - all_predictions: Dict of all class probabilities
    
    Error Codes:
        - 500: Model not loaded or prediction failed
        - 400: No file uploaded or invalid request
    """
    
    # === VALIDATION CHECKS ===
    
    # Check 1: Is model loaded?
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Check server logs for details.'
        }), 500
    
    # Check 2: Was a file uploaded?
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No file field in request. Use key "file" for upload.'
        }), 400
    
    file = request.files['file']
    
    # Check 3: Is the file empty?
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'Empty filename. Please select an image file.'
        }), 400
    
    # === PREDICTION PROCESS ===
    
    try:
        # Step 1: Read uploaded image as bytes
        img_bytes = file.read()
        
        # Step 2: Preprocess image for model
        processed_img = preprocess_image(img_bytes)
        
        # Step 3: Run model prediction
        # Returns array of probabilities for each class
        predictions = model.predict(processed_img, verbose=0)
        
        # Step 4: Extract prediction results
        predicted_class_idx = np.argmax(predictions[0])          # Index of highest probability
        predicted_class = CLASS_NAMES[predicted_class_idx]       # Class name
        confidence = float(predictions[0][predicted_class_idx])  # Confidence score
        
        # Step 5: Create dictionary of all class probabilities
        all_predictions = {
            CLASS_NAMES[i]: float(predictions[0][i]) 
            for i in range(len(CLASS_NAMES))
        }
        
        # Step 6: Sort predictions by confidence (highest first)
        sorted_predictions = dict(
            sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
        )
        
        # Step 7: Return results
        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'confidence': f'{confidence:.2%}',      # e.g., "85.50%"
            'confidence_raw': confidence,            # e.g., 0.8550
            'all_predictions': sorted_predictions,
            'filename': file.filename
        })
    
    except Exception as e:
        # Handle any unexpected errors during prediction
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint for monitoring server status.
    
    Returns:
        JSON with server and model status
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'classes': len(CLASS_NAMES)
    })

# ============================================================================
# SERVER STARTUP
# ============================================================================

if __name__ == '__main__':
    """
    Start the Flask development server.
    
    Configuration:
    - host='0.0.0.0': Accept connections from any IP (local network access)
    - port=5000: Default Flask port
    - debug=True: Enable auto-reload and detailed error messages
    
    WARNING: debug=True should be False in production!
    """
    
    print("\n" + "=" * 70)
    print("🚢 BOAT TYPE CLASSIFICATION API SERVER")
    print("=" * 70)
    print(f"   Server URL: http://localhost:5000")
    print(f"   Model Status: {'✓ Loaded (' + str(len(CLASS_NAMES)) + ' classes)' if model else '✗ NOT LOADED'}")
    print(f"   CORS: Enabled")
    print(f"   Debug Mode: ON")
    print("=" * 70)
    print("\n📡 API Endpoints:")
    print("   GET  /         - API information")
    print("   POST /predict  - Upload image for classification")
    print("   GET  /health   - Health check")
    print("\n🔧 To test: Upload an image using frontend/index.html")
    print("=" * 70 + "\n")
    
    # Get port from environment variable or use default 5000
    port = int(os.environ.get('PORT', 5000))
    
    # Start the server
    app.run(host='0.0.0.0', port=port, debug=True)
