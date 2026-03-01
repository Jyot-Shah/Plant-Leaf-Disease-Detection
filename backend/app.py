from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
import base64
import os
from chatbot import initialize_chat, chat_with_gpt

app = Flask(__name__)
CORS(app)
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), '..', 'frontend')

# Load YOLO model
try:
    model = YOLO('assets/best.pt')
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    model = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify backend is running"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    }), 200

@app.route('/', methods=['GET'])
def serve_frontend():
    """Serve frontend entry page"""
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/<path:path>', methods=['GET'])
def serve_frontend_assets(path):
    """Serve frontend static assets"""
    asset_path = os.path.join(FRONTEND_DIR, path)
    if os.path.exists(asset_path) and os.path.isfile(asset_path):
        return send_from_directory(FRONTEND_DIR, path)
    return jsonify({'error': f'Incorrect route: /{path}'}), 404

@app.errorhandler(404)
def handle_not_found(error):
    """Return JSON for unknown routes"""
    return jsonify({'error': f'Incorrect route: {request.path}'}), 404

@app.route('/predict_json', methods=['POST'])
def predict_json():
    """Detect plant diseases from uploaded leaf image"""
    if not model:
        return jsonify({'error': 'Model not loaded. Please check server configuration.'}), 503
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Validate file type
    allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
    file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    if file_ext not in allowed_extensions:
        allowed = ', '.join(sorted(allowed_extensions))
        return jsonify({'error': f'Invalid file type. Allowed: {allowed}'}), 400
    
    try:
        image_bytes = file.read()
        
        # Validate file size (max 10MB)
        if len(image_bytes) > 10 * 1024 * 1024:
            return jsonify({'error': 'File too large. Maximum size is 10MB.'}), 400
        
        # Open and validate image
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except Exception as img_error:
            return jsonify({'error': 'Invalid image file. Please upload a valid image.'}), 400
        
        # Run YOLO prediction
        results = model.predict(image)[0]
        
        # Extract unique disease names from detections
        diseases = []
        if results.boxes and hasattr(results.boxes, "cls"):
            diseases = list({model.names[int(c)] for c in results.boxes.cls.tolist()})
        
        # Generate annotated image
        annotated = results.plot()
        buf = io.BytesIO()
        Image.fromarray(annotated).save(buf, format="JPEG")
        buf.seek(0)
        image_b64 = base64.b64encode(buf.read()).decode("utf-8")
        
        # Initialize chatbot with detected disease
        if diseases:
            initialize_chat(diseases[0])
        
        return jsonify({
            "diseases": diseases,
            "image_b64": image_b64,
            "status": "success"
        }), 200
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chatbot conversation about detected disease"""
    data = request.get_json(silent=True)
    
    if not data:
        return jsonify({"reply": "Invalid request format."}), 400
    
    message = (data.get('message') or "").strip()
    
    if not message:
        return jsonify({"reply": "Please enter a message."}), 400
    
    if len(message) > 500:
        return jsonify({"reply": "Message too long. Please keep it under 500 characters."}), 400
    
    try:
        reply = chat_with_gpt(message)
        return jsonify({"reply": reply, "status": "success"}), 200
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({"reply": "An error occurred. Please try again."}), 500

if __name__ == '__main__':
    app.run(debug=True)