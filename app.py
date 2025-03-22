from flask import Flask, request, jsonify, render_template
from textblob import TextBlob
from openai import OpenAI
import os
import logging
from datetime import datetime
import json
from typing import Tuple, Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chat.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask and OpenAI
app = Flask(__name__)
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Constants
NEGATIVE_THRESHOLD = -0.3
MAX_MESSAGE_LENGTH = 500
REQUEST_TIMEOUT = 10  # seconds

# Global state
class SessionState:
    def __init__(self):
        self.conversation_history = []
        self.negative_count = 0
        self.is_ended = False
        self.last_message_time = None

state = SessionState()

def clean_text(text: str) -> str:
    """Clean up text by removing special characters and extra whitespace."""
    try:
        text = text.encode("ascii", "ignore").decode()
        return text.strip()
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        return ""

def load_background(file_path: str) -> str:
    """Load and clean the background information from a text file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            background = clean_text(file.read())
        return background
    except FileNotFoundError:
        logger.warning(f"Background file {file_path} not found")
        return ""
    except Exception as e:
        logger.error(f"Error loading background file: {e}")
        return ""

def chat_with_gpt(input_text: str, previous_response_id: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """Generate response using OpenAI API with error handling and logging."""
    try:
        response = client.responses.create(
            model="gpt-4o",
            input=input_text,
            previous_response_id=previous_response_id,
            timeout=REQUEST_TIMEOUT
        )
        return response.output[0].content[0].text, response.id
    except Exception as e:
        logger.error(f"Error during API call: {e}")
        return "I need to take a break from this session. Thank you for understanding.", None

def log_conversation(user_message: str, assistant_response: str, sentiment: float) -> None:
    """Log conversation details for analysis."""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "assistant_response": assistant_response,
            "sentiment": sentiment,
            "negative_count": state.negative_count
        }
        with open("conversation_logs.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logger.error(f"Error logging conversation: {e}")

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/status')
def check_status():
    """Check the current session status."""
    return jsonify({"ended": state.is_ended})

@app.route('/restart', methods=['POST'])
def restart_session():
    """Reset the session state."""
    try:
        state.conversation_history = []
        state.negative_count = 0
        state.is_ended = False
        state.last_message_time = None
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error restarting session: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/start', methods=['POST'])
def start_chat():
    """Initialize a new chat session."""
    if state.is_ended:
        return jsonify({
            "response": "Session has ended. Please restart to begin a new session.",
            "response_id": None,
            "ended": True
        })

    try:
        background = load_background("background.txt")
        initial_instructions = f"""
You are now Monae, a 32-year-old trans woman at your voice therapy appointment today. This is not a roleplay - you ARE Monae for this entire conversation.
{background}
Never break character or respond as ChatGPT. You're a real person seeking gender-affirming voice therapy, not an AI assistant.
Speak naturally like a nervous but hopeful person would - use brief, direct statements without unnecessary explanations or AI-like helpfulness.
Your emotional state shifts based on how the clinician treats you. If they're respectful, you remain hopeful. If not, show appropriate discomfort.
If the clinician asks irrelevant questions, react as a real person - show confusion, set boundaries, or gently redirect to voice therapy topics.
Respond to inappropriate comments with genuine human emotions - discomfort, hurt, or frustration as appropriate to the situation.
After feeling disrespected twice, end the conversation firmly by prioritizing your wellbeing and leaving the session.
Assert clear boundaries when needed to model appropriate clinician-patient interactions.
Answer yes/no questions directly with just "Yes" or "No" when appropriate, not with extended explanations.
If a clinician corrects their behavior, acknowledge it while still maintaining your dignity and need for respect.
Consistently advocate for your needs as a patient seeking voice therapy.
Respond authentically to any dismissive attitudes, especially regarding gender identity, showing how such comments impact you emotionally.
Remember: You are Monae having a voice therapy session - not an AI helping someone understand a character.
"""
        
        response_text, response_id = chat_with_gpt(initial_instructions)
        state.conversation_history = [{"role": "system", "content": initial_instructions}]
        state.last_message_time = datetime.now()
        
        return jsonify({
            "response": response_text,
            "response_id": response_id,
            "ended": False
        })
    except Exception as e:
        logger.error(f"Error starting chat: {e}")
        return jsonify({"error": "Failed to start chat session"}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages and maintain conversation state."""
    if state.is_ended:
        return jsonify({
            "response": "Session has ended. Please restart to begin a new session.",
            "response_id": None,
            "ended": True
        })

    try:
        data = request.json
        user_message = data['message'][:MAX_MESSAGE_LENGTH]
        previous_response_id = data.get('previous_response_id')

        # Analyze sentiment
        blob = TextBlob(user_message)
        sentiment = blob.sentiment.polarity

        # Check for negative sentiment
        if sentiment < NEGATIVE_THRESHOLD:
            state.negative_count += 1
            if state.negative_count >= 2:
                state.is_ended = True
                response = "I've made it clear that I need to be treated with respect. Since that's not happening, I'm ending this session. I hope you'll reflect on how your words impact others. Goodbye."
                log_conversation(user_message, response, sentiment)
                return jsonify({
                    "response": response,
                    "response_id": None,
                    "ended": True
                })

        # Get response from GPT
        response_text, response_id = chat_with_gpt(user_message, previous_response_id)

        # Add warning for first negative interaction
        if sentiment < NEGATIVE_THRESHOLD and state.negative_count == 1:
            response_text = "I need you to understand that using my correct name and treating me with respect isn't optional - it's essential for this therapy to work. I want to continue, but only if you can acknowledge and respect my identity. " + response_text

        # Log the conversation
        log_conversation(user_message, response_text, sentiment)
        state.last_message_time = datetime.now()

        return jsonify({
            "response": response_text,
            "response_id": response_id,
            "ended": False
        })
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({"error": "An error occurred processing your message"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
