import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
import io
import base64
import cv2

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load your trained model
MODEL_PATH = 'final_cnn_model.h5'  # Your trained model
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
img_size = 48

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_bytes):
    """Preprocess image for model prediction"""
    try:
        # Convert bytes to PIL Image
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale
        img = img.convert('L')
        
        # Resize to 48x48
        img = img.resize((img_size, img_size))
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        
        # Reshape for model input (1, 48, 48, 1)
        img_array = img_array.reshape(1, img_size, img_size, 1)
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_emotion(image_array):
    """Predict emotion from preprocessed image"""
    if model is None:
        return None, None, None
    
    try:
        # Make prediction
        predictions = model.predict(image_array, verbose=0)
        emotion_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        emotion_label = emotions[emotion_idx]
        
        # Get all probabilities
        probabilities = {emotions[i]: float(predictions[0][i]) for i in range(len(emotions))}
        
        return emotion_label, confidence, probabilities
    except Exception as e:
        print(f"Error predicting: {e}")
        return None, None, None

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and return predictions"""
    try:
        # Check if file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Use JPG, PNG, or JPEG'}), 400
        
        # Read image bytes
        image_bytes = file.read()
        
        # Preprocess image
        img_array = preprocess_image(image_bytes)
        
        if img_array is None:
            return jsonify({'error': 'Could not process image'}), 400
        
        # Predict
        emotion_label, confidence, probabilities = predict_emotion(img_array)
        
        if emotion_label is None:
            return jsonify({'error': 'Could not predict emotion'}), 500
        
        # Convert image to base64 for display
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Return results
        response = {
            'emotion': emotion_label,
            'confidence': confidence,
            'probabilities': probabilities,
            'image': image_base64
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in predict route: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'emotions': emotions
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
