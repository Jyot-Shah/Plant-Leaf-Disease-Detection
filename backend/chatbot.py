import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from parent directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("WARNING: GEMINI_API_KEY not found in .env file. Chat functionality will be limited.")
else:
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")

SYSTEM_PROMPT = """You are a concise plant pathology assistant.
When replying:
- Use short sentences and a friendly, professional tone.
- Do NOT use bullet points or numbered lists; use brief paragraphs.
- Keep answers under 120 words.
Given a detected leaf disease name and the user's message, provide:
- Likely treatment/medication (safe, region-agnostic).
- Cultural practices and precautions.
- When to consult a local agronomist.
If disease is empty, ask for one."""

model = genai.GenerativeModel('gemini-2.5-flash')
chat_session = None

def initialize_chat(disease: str):
    """Initialize chat session with detected disease context"""
    global chat_session
    try:
        chat_session = model.start_chat(history=[])
        initial_message = f"{SYSTEM_PROMPT}\n\nDisease detected: {disease}"
        chat_session.send_message(initial_message)
    except Exception as e:
        print(f"Error initializing chat: {e}")
        chat_session = None

def chat_with_gpt(user_message: str) -> str:
    """Send user message to chat session and return response"""
    global chat_session
    if chat_session is None:
        return "Chat service is currently unavailable. Please try uploading an image first."
    
    try:
        response = chat_session.send_message(user_message)
        return response.text.strip()
    except Exception as e:
        print(f"Chatbot error: {e}")
        return "Unable to process your message at this time. Please try again."