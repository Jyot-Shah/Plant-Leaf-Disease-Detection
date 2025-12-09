# 🌿 Plant Leaf Disease Detection & Treatment System

An AI-powered web application that detects plant leaf diseases using YOLOv11 object detection and provides personalized treatment recommendations through an intelligent chatbot powered by Google's Gemini.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-red.svg)
![Gemini](https://img.shields.io/badge/Gemini-AI-orange.svg)
![Roboflow](https://img.shields.io/badge/Roboflow-Dataset-purple.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Table of Contents
- [Features](#-features)
- [Demo](#-demo)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Model Training](#-model-training)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [API Endpoints](#-api-endpoints)
- [Configuration](#-configuration)
- [Acknowledgments](#-acknowledgments)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

- 🔍 **Real-time Disease Detection**: Upload leaf images and get instant disease identification using YOLOv11
- 🤖 **AI Treatment Assistant**: Chat with an intelligent bot for personalized medication and treatment advice
- 📊 **Visual Results**: View annotated images with bounding boxes highlighting affected areas
- 💬 **Conversational Interface**: Natural language interaction for treatment recommendations
- 🎨 **Modern UI**: Clean, responsive design with gradient backgrounds and smooth animations
- ⚡ **Fast Processing**: Optimized YOLOv11 model for quick detection and response times
- 🔄 **Context-Aware Chat**: Maintains conversation history for follow-up questions

## 🎥 Demo

| Step | Description |
|------|-------------|
| 1️⃣ | Upload a leaf image using the file selector |
| 2️⃣ | Click "Analyze Image" - loading spinner appears |
| 3️⃣ | View annotated image with detected disease badge |
| 4️⃣ | Chat with AI assistant for treatment advice |

## 🛠 Tech Stack

### Backend
| Technology | Purpose |
|------------|---------|
| **Flask** | Web framework & REST API |
| **Ultralytics YOLOv11** | Object detection model |
| **Google Gemini 2.0 Flash** | Conversational AI chatbot |
| **Pillow** | Image processing |
| **Flask-CORS** | Cross-origin resource sharing |
| **python-dotenv** | Environment variable management |

### Frontend
| Technology | Purpose |
|------------|---------|
| **HTML5** | Page structure |
| **CSS3** | Styling, gradients, animations |
| **Vanilla JavaScript** | Client-side logic & API calls |
| **Fetch API** | Asynchronous HTTP requests |

### AI/ML
| Technology | Purpose |
|------------|---------|
| **YOLOv11** | Real-time disease detection |
| **Gemini 2.0 Flash** | Natural language processing |
| **Roboflow** | Dataset management & augmentation |
| **Google Colab** | Model training environment |

## 📊 Dataset

### Original Dataset
This project uses the **PlantDoc Dataset** by Chainfly as the base dataset:

🔗 **Original Dataset**: [PlantDoc Dataset by Chainfly](YOUR_LINK_HERE)

### Custom Augmented Dataset
I created a custom augmented version of the dataset on Roboflow with various transformations to improve model performance:

🔗 **Augmented Dataset**: [YOUR_ROBOFLOW_DATASET_LINK_HERE]

### Augmentation Details

The following augmentations were applied to enhance the training data:

![Augmentation Details](assets/augmentations.png)

**Applied Augmentations:**
- Rotation (±15°)
- Horizontal & Vertical Flips
- Brightness & Contrast adjustments
- Blur effects
- Noise injection
- Mosaic augmentation
- Scale variations

These augmentations help the model generalize better to real-world conditions like different lighting, angles, and image quality.

## 🧠 Model Training

The YOLOv11 model was trained on Google Colab using the augmented dataset from Roboflow.

### Training Notebook
All training code is available in [`trainPlantDocToYOLO.ipynb`](trainPlantDocToYOLO.ipynb)

### How to Train Your Own Model

1. **Open the Colab Notebook**
   - Upload `trainPlantDocToYOLO.ipynb` to Google Colab
   - Or open directly from this repository

2. **Get Roboflow API Key**
   - Go to [my Roboflow dataset](YOUR_ROBOFLOW_DATASET_LINK_HERE)
   - Create an account or sign in
   - Navigate to your workspace settings
   - Copy your API key

3. **Configure the Notebook**
   ```python
   # Replace with your Roboflow API key
   rf = Roboflow(api_key="YOUR_API_KEY_HERE")
   ```

4. **Run All Cells**
   - The notebook will download the dataset
   - Train the YOLOv11 model
   - Export the best weights (`best.pt`)

5. **Download the Model**
   - After training, download `best.pt` from the output
   - Place it in `backend/assets/` folder

### Training Configuration
```python
model = YOLO('yolo11n.pt')  # YOLOv11 nano as base
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))
- Trained YOLO model (`best.pt`) or train your own

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/leaf-disease-detection.git
cd leaf-disease-detection
```

### Step 2: Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
Create a `.env` file in the **root directory** (not in backend):
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### Step 4: Add YOLO Model
Place your trained model file in the backend:
```
backend/assets/best.pt
```

### Step 5: Run the Application
```bash
cd backend
python app.py
```
Backend runs on `http://127.0.0.1:5000`

### Step 6: Open Frontend
Open `frontend/index.html` in browser or use Live Server extension in VS Code.

## 🚀 Usage

### Disease Detection Flow
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Upload Image   │ ──▶ │  YOLO Detection │ ──▶ │  Display Result │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Get Treatment  │ ◀── │  Gemini AI Chat │ ◀── │  Init Chatbot   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Steps
1. **Upload Image**: Select a leaf image (JPG, PNG)
2. **Analyze**: Click "Analyze Image" button
3. **View Results**: See annotated image + disease badge
4. **Chat**: Ask questions about treatment & prevention

## 📁 Project Structure

```
Leaf Disease Detection/
│
├── backend/
│   ├── app.py                    # Flask app & API routes
│   │   ├── /predict_json         # Disease detection endpoint
│   │   └── /chat                 # Chatbot endpoint
│   │
│   ├── chatbot.py                # Gemini AI integration
│   │   ├── initialize_chat()     # Start chat session with disease context
│   │   └── chat_with_gpt()       # Send messages & get responses
│   │
│   ├── requirements.txt          # Python dependencies
│   └── assets/
│       └── best.pt              # Trained YOLOv11 model weights
│
├── frontend/
│   ├── index.html               # Main HTML structure
│   │   ├── Header section
│   │   ├── Upload card
│   │   ├── Result card
│   │   └── Chatbot card
│   │
│   ├── style.css                # Styling
│   │   ├── Green gradient theme
│   │   ├── Card animations
│   │   ├── Loading spinner
│   │   └── Chat message bubbles
│   │
│   └── app.js                   # Client-side logic
│       ├── Form submission handler
│       ├── API calls (fetch)
│       └── Chat message handling
│
├── trainPlantDocToYOLO.ipynb    # Colab training notebook
├── .env                         # Environment variables (create this)
├── .gitignore                   # Git ignore rules
└── README.md                    # Documentation
```

## 🔌 API Endpoints

### Disease Detection
```http
POST /predict_json
Content-Type: multipart/form-data
```

**Request Body:**
| Field | Type | Description |
|-------|------|-------------|
| file | File | Leaf image (JPG/PNG) |

**Response:**
```json
{
  "diseases": ["Tomato Late Blight"],
  "image_b64": "base64_encoded_annotated_image..."
}
```

### Chat
```http
POST /chat
Content-Type: application/json
```

**Request Body:**
```json
{
  "message": "How do I treat this disease?"
}
```

**Response:**
```json
{
  "reply": "For Tomato Late Blight, apply copper-based fungicides early in the morning..."
}
```

### Error Responses
```json
{
  "error": "No file provided"
}
```

## ⚙️ Configuration

### Environment Variables
| Variable | Description | Location |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini API key | `.env` (root) |

### Model Path
Update in `backend/app.py` if needed:
```python
model = YOLO('assets/best.pt')
```

### Chatbot System Prompt
Customize AI behavior in `backend/chatbot.py`:
```python
SYSTEM_PROMPT = """You are a concise plant pathology assistant..."""
```

## 🙏 Acknowledgments

### Inspiration
This project was inspired by and built upon tutorials from **Augmented AI** YouTube channel. Their videos provided valuable guidance on implementing plant disease detection systems.

🔗 **Augmented AI YouTube Channel**: [https://www.youtube.com/@AugmentedAI](https://www.youtube.com/@AugmentedAI)

### Resources & Tools
| Resource | Purpose |
|----------|---------|
| [PlantDoc Dataset](YOUR_LINK_HERE) | Original dataset by Chainfly |
| [Roboflow](https://roboflow.com) | Dataset augmentation & management |
| [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics) | Object detection framework |
| [Google Gemini](https://deepmind.google/technologies/gemini/) | Conversational AI |
| [Google Colab](https://colab.research.google.com) | Free GPU for training |

## 🤝 Contributing

Contributions are welcome! Here's how:

1. **Fork** the repository
2. **Create** a feature branch
   ```bash
   git checkout -b feature/YourFeature
   ```
3. **Commit** your changes
   ```bash
   git commit -m "Add YourFeature"
   ```
4. **Push** to the branch
   ```bash
   git push origin feature/YourFeature
   ```
5. **Open** a Pull Request

### Ideas for Contribution
- Add more disease classes
- Improve UI/UX
- Add multi-language support
- Mobile responsive improvements
- Add disease prevention tips database

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

**Your Name** - [your.email@example.com](mailto:your.email@example.com)

🔗 **Project Link**: [https://github.com/yourusername/leaf-disease-detection](https://github.com/yourusername/leaf-disease-detection)

---

<p align="center">
  ⭐ If you found this project helpful, please give it a star!
</p>

<p align="center">
  Made with 💚 for plant health
</p>