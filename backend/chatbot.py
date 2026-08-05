import os
import uuid
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
# Load environment variables from parent directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
client = None
if not api_key:
    print("WARNING: GEMINI_API_KEY not found in .env file. Chat functionality will be limited.")
else:
    try:
        client = genai.Client(api_key=api_key)
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

ACTIVE_SESSIONS = {}
SESSION_TTL_SECONDS = 1800  # 30 minutes

def cleanup_sessions():
    """Remove expired sessions from memory."""
    current_time = time.time()
    expired_keys = [k for k, v in ACTIVE_SESSIONS.items() if current_time - v['last_accessed'] > SESSION_TTL_SECONDS]
    for k in expired_keys:
        del ACTIVE_SESSIONS[k]

def initialize_chat(disease: str) -> str:
    """Initialize chat session with detected disease context, returns session_id"""
    cleanup_sessions()
    try:
        session_id = str(uuid.uuid4())
        if not client:
            return None
            
        instruction = f"{SYSTEM_PROMPT}\n\nIMPORTANT CONTEXT: The user's plant is diagnosed with '{disease}'."
        chat_session = client.chats.create(
            model='gemini-2.5-flash',
            config=types.GenerateContentConfig(
                system_instruction=instruction
            )
        )
        ACTIVE_SESSIONS[session_id] = {
            'session': chat_session,
            'last_accessed': time.time()
        }
        return session_id
    except Exception as e:
        print(f"Error initializing chat: {e}")
        return None

def chat_with_gpt(session_id: str, user_message: str) -> str:
    """Send user message to chat session and return response"""
    cleanup_sessions()
    
    if not session_id or session_id not in ACTIVE_SESSIONS:
        return "Chat service is currently unavailable or session expired. Please analyze a new image."
    
    session_data = ACTIVE_SESSIONS[session_id]
    chat_session = session_data['session']
    
    try:
        response = chat_session.send_message(user_message)
        session_data['last_accessed'] = time.time()  # Update last accessed
        return response.text.strip()
    except Exception as e:
        print(f"Chatbot error: {e}")
        return "Unable to process your message at this time. Please try again."