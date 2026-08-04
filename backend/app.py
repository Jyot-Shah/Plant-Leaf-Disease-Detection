from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import traceback
from chatbot import initialize_chat, chat_with_gpt
from predictor import predict_disease, is_model_loaded

app = Flask(__name__)
CORS(app)
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), '..', 'frontend')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify backend is running"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': is_model_loaded()
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
    if not is_model_loaded():
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
        
        diseases, image_b64 = predict_disease(image_bytes)
        
        # Initialize chatbot with detected disease
        session_id = None
        if diseases:
            session_id = initialize_chat(diseases[0])
        
        return jsonify({
            "diseases": diseases,
            "image_b64": image_b64,
            "session_id": session_id,
            "status": "success"
        }), 200
    
    except ValueError as val_err:
        return jsonify({'error': str(val_err)}), 400
    except Exception as e:
        err_msg = traceback.format_exc()
        print(f"Prediction error:\\n{err_msg}")
        return jsonify({'error': f'Prediction failed: {str(e)}', 'traceback': err_msg}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chatbot conversation about detected disease"""
    data = request.get_json(silent=True)
    
    if not data:
        return jsonify({"reply": "Invalid request format."}), 400
    
    message = (data.get('message') or "").strip()
    session_id = data.get('session_id')
    
    if not message:
        return jsonify({"reply": "Please enter a message."}), 400
    
    if len(message) > 500:
        return jsonify({"reply": "Message too long. Please keep it under 500 characters."}), 400
    
    try:
        reply = chat_with_gpt(session_id, message)
        return jsonify({"reply": reply, "status": "success"}), 200
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({"reply": "An error occurred. Please try again."}), 500

if __name__ == '__main__':
    app.run(debug=True)