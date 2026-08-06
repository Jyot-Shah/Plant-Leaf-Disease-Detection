import os
import io
import base64
from PIL import Image
from ultralytics import YOLO

# Resolve absolute path to the model so it works regardless of cwd
BASE_DIR = os.path.dirname(__file__)
PT_PATH = os.path.join(BASE_DIR, 'assets', 'best.pt')

try:
    model = YOLO(PT_PATH, task='detect')
except Exception as e:
    print(f"Error loading YOLO model from {PT_PATH}: {e}")
    model = None

def is_model_loaded() -> bool:
    return model is not None

def predict_disease(image_bytes: bytes) -> tuple:
    """
    Validates the image and predicts the disease using YOLO.
    Returns a tuple: (list_of_diseases, base64_image_string, highest_conf)
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
    
    # Extract unique diseases sorted by highest confidence
    disease_map = {}
    if results.boxes and hasattr(results.boxes, "cls"):
        for c, conf in zip(results.boxes.cls.tolist(), results.boxes.conf.tolist()):
            name = model.names[int(c)]
            if name not in disease_map or conf > disease_map[name]:
                disease_map[name] = conf
                
    sorted_items = sorted(disease_map.items(), key=lambda x: x[1], reverse=True)
    diseases = [item[0] for item in sorted_items]
    highest_conf = round((sorted_items[0][1] * 100), 1) if sorted_items else 0.0
        
    # Generate annotated image in b64
    try:
        annotated = results.plot(conf=True, labels=True)
        buf = io.BytesIO()
        Image.fromarray(annotated[..., ::-1]).save(buf, format="JPEG")
        buf.seek(0)
        image_b64 = base64.b64encode(buf.read()).decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to generate annotated image: {e}") from e
        
    return diseases, image_b64, highest_conf
