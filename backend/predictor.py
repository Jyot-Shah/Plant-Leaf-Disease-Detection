import os
import io
import base64
from PIL import Image
from ultralytics import YOLO

# Resolve absolute path to the model so it works regardless of cwd
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'assets', 'best.pt')

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error loading YOLO model from {MODEL_PATH}: {e}")
    model = None

def is_model_loaded() -> bool:
    return model is not None

def predict_disease(image_bytes: bytes) -> tuple:
    """
    Validates the image and predicts the disease using YOLO.
    Returns a tuple: (list_of_diseases, base64_image_string)
    """
    if not model:
        raise ValueError("Model is not loaded on the server.")
    
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception as e:
        raise ValueError("Invalid image file. Could not parse image bytes.") from e
        
    try:
        results = model.predict(image)[0]
    except Exception as e:
        raise RuntimeError(f"YOLO prediction failed: {e}") from e
    
    # Extract unique diseases
    diseases = []
    if results.boxes and hasattr(results.boxes, "cls"):
        diseases = list({model.names[int(c)] for c in results.boxes.cls.tolist()})
        
    # Generate annotated image in b64
    try:
        annotated = results.plot()
        buf = io.BytesIO()
        Image.fromarray(annotated).save(buf, format="JPEG")
        buf.seek(0)
        image_b64 = base64.b64encode(buf.read()).decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to generate annotated image: {e}") from e
        
    return diseases, image_b64
