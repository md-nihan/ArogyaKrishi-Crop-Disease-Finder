from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import pickle
import os
import base64

# Conditional import for TensorFlow to avoid IDE warnings
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables to track model status
model_loaded = False
using_mock_model = False

# Load the pre-trained models
if TF_AVAILABLE:
    try:
        crop_disease_model = tf.keras.models.load_model('leaf_disease_model.keras')
        with open('class_names.pkl', 'rb') as f:
            DISEASE_CLASSES = pickle.load(f)
        model_loaded = True
        using_mock_model = False
        print("Model and class names loaded successfully!")
        print(f"Model can classify {len(DISEASE_CLASSES)} different plant diseases")
    except Exception as e:
        print(f"Error loading model: {e}")
        # Fallback to mock model and classes
        crop_disease_model = None
        DISEASE_CLASSES = [
            'Apple___Apple_scab',
            'Apple___Black_rot',
            'Apple___Cedar_apple_rust',
            'Apple___healthy',
            'Blueberry___healthy',
            'Cherry_(including_sour)___Powdery_mildew',
            'Cherry_(including_sour)___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
            'Corn_(maize)___Common_rust_',
            'Corn_(maize)___Northern_Leaf_Blight',
            'Corn_(maize)___healthy',
            'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)',
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)',
            'Peach___Bacterial_spot',
            'Peach___healthy',
            'Pepper,_bell___Bacterial_spot',
            'Pepper,_bell___healthy',
            'Potato___Early_blight',
            'Potato___Late_blight',
            'Potato___healthy',
            'Raspberry___healthy',
            'Soybean___healthy',
            'Squash___Powdery_mildew',
            'Strawberry___Leaf_scorch',
            'Strawberry___healthy',
            'Tomato___Bacterial_spot',
            'Tomato___Early_blight',
            'Tomato___Late_blight',
            'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]
        model_loaded = False
        using_mock_model = True
        print("Using mock model - run training to get a real model")
else:
    crop_disease_model = None
    DISEASE_CLASSES = [
        'Apple___Apple_scab',
        'Apple___Black_rot',
        'Apple___Cedar_apple_rust',
        'Apple___healthy',
        'Blueberry___healthy',
        'Cherry_(including_sour)___Powdery_mildew',
        'Cherry_(including_sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
        'Corn_(maize)___Common_rust_',
        'Corn_(maize)___Northern_Leaf_Blight',
        'Corn_(maize)___healthy',
        'Grape___Black_rot',
        'Grape___Esca_(Black_Measles)',
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
        'Grape___healthy',
        'Orange___Haunglongbing_(Citrus_greening)',
        'Peach___Bacterial_spot',
        'Peach___healthy',
        'Pepper,_bell___Bacterial_spot',
        'Pepper,_bell___healthy',
        'Potato___Early_blight',
        'Potato___Late_blight',
        'Potato___healthy',
        'Raspberry___healthy',
        'Soybean___healthy',
        'Squash___Powdery_mildew',
        'Strawberry___Leaf_scorch',
        'Strawberry___healthy',
        'Tomato___Bacterial_spot',
        'Tomato___Early_blight',
        'Tomato___Late_blight',
        'Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'
    ]
    model_loaded = False
    using_mock_model = True
    print("TensorFlow not available - using mock model")

@app.route('/')
def home():
    if model_loaded and not using_mock_model:
        model_status = "Trained model loaded"
    elif using_mock_model:
        model_status = "Using mock model (training not completed)"
    else:
        model_status = "No model loaded"
    
    return jsonify({
        "message": "AI Server is running",
        "model_status": model_status,
        "disease_classes_count": len(DISEASE_CLASSES),
        "using_mock_model": using_mock_model
    })

@app.route('/predict', methods=['POST'])
def predict_disease():
    try:
        # Check if image is sent as file or in JSON data
        if 'image' in request.files:
            # Handle file upload directly
            image_file = request.files['image']
            image = Image.open(image_file.stream)
        elif request.is_json and request.json and 'image' in request.json:
            # Handle base64 encoded image from JSON
            image_data = request.json.get('image')
            if image_data:
                try:
                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(io.BytesIO(image_bytes))
                except Exception as e:
                    return jsonify({
                        "success": False,
                        "error": f"Failed to decode image data: {str(e)}"
                    }), 400
            else:
                return jsonify({
                    "success": False,
                    "error": "No image data provided"
                }), 400
        else:
            return jsonify({
                "success": False,
                "error": "No image provided"
            }), 400
        
        # Process the image
        processed_image = preprocess_image(image)
        
        # Make prediction using the actual model (if loaded)
        if model_loaded and not using_mock_model and crop_disease_model is not None:
            try:
                prediction = crop_disease_model.predict(processed_image)
                predicted_class_idx = np.argmax(prediction[0])
                confidence = float(prediction[0][predicted_class_idx])
                predicted_class = DISEASE_CLASSES[predicted_class_idx]
                
                # Add additional information about the model being used
                model_info = "Trained model"
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": f"Model prediction failed: {str(e)}"
                }), 500
        else:
            # Fallback to mock prediction
            predicted_class = "Corn_(maize)___Common_rust_"
            confidence = 0.32  # Lower confidence for mock predictions
            model_info = "Mock model"
        
        return jsonify({
            "success": True,
            "disease": predicted_class,
            "confidence": confidence,
            "model_used": model_info
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Internal error: {str(e)}"
        }), 500

def preprocess_image(image):
    """
    Preprocess the image for model prediction
    """
    # Convert to RGB if necessary (handle RGBA, P, etc.)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image to match model input size (224x224 for MobileNetV2)
    image = image.resize((224, 224))
    # Convert to array and normalize
    image_array = np.array(image) / 255.0
    # Expand dimensions to match model input shape
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)