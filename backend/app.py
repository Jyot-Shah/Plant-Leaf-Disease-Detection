from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
import base64

from chatbot import initialize_chat, chat_with_gpt

app = Flask(__name__)
CORS(app)

# Load the YOLO model
model = YOLO('assets/best.pt')

@app.route('/predict_json', methods=['POST'])
def predict_json():
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': 'No file provided'}), 400
    try:
        image_bytes = request.files['file'].read()
        image = Image.open(io.BytesIO(image_bytes))
        results = model.predict(image)[0]

        diseases = []
        if results.boxes and hasattr(results.boxes, "cls"):
            diseases = list({model.names[int(c)] for c in results.boxes.cls.tolist()})

        annotated = results.plot()
        buf = io.BytesIO()
        Image.fromarray(annotated).save(buf, format="JPEG")
        buf.seek(0)
        image_b64 = base64.b64encode(buf.read()).decode("utf-8")

        # Initialize chatbot with detected disease
        detected_disease = diseases[0] if diseases else ""
        if detected_disease:
            initialize_chat(detected_disease)

        return jsonify({"diseases": diseases, "image_b64": image_b64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get('message') or "").strip()
    if not message:
        return jsonify({"reply": "Please enter a message."}), 400
    reply = chat_with_gpt(message)
    return jsonify({"reply": reply})


if __name__ == '__main__':
    app.run(debug=True)