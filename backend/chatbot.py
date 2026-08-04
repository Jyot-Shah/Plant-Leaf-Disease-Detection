import os
import uuid
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
Given the detected leaf disease name in the context, provide:
- Likely treatment/medication (safe, region-agnostic).
- Cultural practices and precautions.
- When to consult a local agronomist."""

# default model (unused for chat, just kept if needed)
model = genai.GenerativeModel('gemini-2.5-flash')
ACTIVE_SESSIONS = {}

def initialize_chat(disease: str) -> str:
    """Initialize chat session with detected disease context, returns session_id"""
    try:
        session_id = str(uuid.uuid4())
        # Inject the parsed disease into the systemic instruction for this session
        instruction = f"{SYSTEM_PROMPT}\n\nIMPORTANT CONTEXT: The user's plant is diagnosed with '{disease}'."
        local_model = genai.GenerativeModel('gemini-2.5-flash', system_instruction=instruction)
        
        chat_session = local_model.start_chat(history=[])
        ACTIVE_SESSIONS[session_id] = chat_session
        return session_id
    except Exception as e:
        print(f"Error initializing chat: {e}")
        return None

def chat_with_gpt(session_id: str, user_message: str) -> str:
    """Send user message to chat session and return response"""
    if not session_id or session_id not in ACTIVE_SESSIONS:
        return "Chat service is currently unavailable or session expired. Please analyze a new image."
    
    chat_session = ACTIVE_SESSIONS[session_id]
    try:
        response = chat_session.send_message(user_message)
        return response.text.strip()
    except Exception as e:
        print(f"Chatbot error: {e}")
        return "Unable to process your message at this time. Please try again."