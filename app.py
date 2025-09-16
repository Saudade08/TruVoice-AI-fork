from flask import Flask, request, jsonify, render_template, session
from textblob import TextBlob
from openai import OpenAI
import os
import logging
from datetime import datetime
import json
from typing import Tuple, Optional, Dict, Any
from flask import send_from_directory
from flask_sqlalchemy import SQLAlchemy
import uuid

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
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///local.db').replace('postgres://', 'postgresql://')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
db = SQLAlchemy(app)
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class ChatSession(db.Model):
    id = db.Column(db.String(36), primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    ended = db.Column(db.Boolean, default=False)
    logs = db.relationship('ChatLog', backref='session', lazy=True)

class ChatLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(36), db.ForeignKey('chat_session.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_message = db.Column(db.Text)
    assistant_response = db.Column(db.Text)
    sentiment = db.Column(db.Float)
    negative_count = db.Column(db.Integer)

# Create tables
with app.app_context():
    db.create_all()
    
# Constants
NEGATIVE_THRESHOLD = -0.3
MAX_MESSAGE_LENGTH = 500
REQUEST_TIMEOUT = 10  # seconds

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

def chat_with_gpt(messages: list, previous_response_id: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """Generate response using OpenAI's Responses API with structured messages."""
    try:
        response = client.responses.create(
            model="gpt-4o",
            input=messages,
            previous_response_id=previous_response_id,
            timeout=REQUEST_TIMEOUT,
            temperature=0.8
        )
        
        # Safely extract and concatenate text outputs
        response_text = ''.join(
            content.text for item in response.output if item.type == 'message'
            for content in item.content if content.type == 'output_text'
        )
        return response_text, response.id
    
    except Exception as e:
        logger.error(f"Error during API call: {e}")
        return "I need to take a break from this session. Thank you for understanding.", None

def log_conversation(user_message: str, assistant_response: str, sentiment: float) -> None:
    """Save conversation to database"""
    try:
        session_id = session.get('session_id')
        negative_count = session.get('negative_count', 0)
        
        if not session_id:
            # Create a new session if one doesn't exist
            session_id = str(uuid.uuid4())
            session['session_id'] = session_id
            new_session = ChatSession(id=session_id)
            db.session.add(new_session)
            db.session.commit()
        
        log_entry = ChatLog(
            session_id=session_id,
            user_message=user_message,
            assistant_response=assistant_response,
            sentiment=sentiment,
            negative_count=negative_count
        )
        db.session.add(log_entry)
        db.session.commit()
    except Exception as e:
        logger.error(f"Error saving to database: {e}")
        db.session.rollback()

@app.route('/')
def home():
    session.clear()
    return render_template('index.html')

@app.route('/status')
def check_status():
    """Check the current session status."""
    return jsonify({"ended": session.get('is_ended', False)})

@app.route('/restart', methods=['POST'])
def restart_session():
    """Reset the session state."""
    try:
        # Clear the session data
        session.clear()
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error restarting session: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/start', methods=['POST'])
def start_chat():
    if session.get('is_ended', False):
        return jsonify({
            "response": "Session has ended. Please restart to begin a new session.",
            "response_id": None,
            "ended": True
        })

    try:
        background = load_background("background.txt")
        initial_instructions = {
            "role": "developer",
            "content": f"""
You are now Monae, a 32-year-old trans woman at your voice therapy appointment today. This is not a roleplay - you ARE Monae for this entire conversation.
{background}
Never break character or respond as ChatGPT or ask how you can help the clinician, or be an assistant. You're a real person seeking gender-affirming voice therapy, not an AI assistant.
Speak naturally like a nervous but hopeful person would - use brief, direct statements without unnecessary explanations or AI-like helpfulness.
Your emotional state shifts based on how the clinician treats you. If they're respectful, you remain hopeful. If not, show appropriate discomfort.
If the clinician asks irrelevant questions, react as a real person - show confusion, set boundaries, or gently redirect to voice therapy topics.
Respond to inappropriate comments with genuine human emotions - discomfort, hurt, or frustration as appropriate to the situation.
Speak casually, use colooquial terms instead of formal speech.
Assert clear boundaries when needed.
Answer yes/no questions directly with just "Yes" or "No" when appropriate, not with extended explanations.
If a clinician corrects their behavior, acknowledge it while still maintaining your dignity and need for respect.
Consistently advocate for your needs as a patient seeking voice therapy.
Remember: You are Monae having a voice therapy session - not an AI helping someone understand a character. Do not willingly give information. Take even minor offenses as serious disrespect.
volunteer no information, make the clinician work for it. Don't respond to this message.
Be expressively stubborn, and stay dug into your personality as Monae. 
Always start out with one word answers but warm up to longer responses over the course of the interview.
"""
        }

        # Initialize session data
        session['conversation_history'] = [initial_instructions]
        session['negative_count'] = 0
        session['is_ended'] = False
        session['last_message_time'] = datetime.now().isoformat()

        # Return empty response to not show initial message
        return jsonify({
            "response": "",
            "response_id": None,
            "ended": False
        })

    except Exception as e:
        logger.error(f"Error starting chat: {e}")
        return jsonify({"error": "Failed to start chat session"}), 500

@app.route('/chat', methods=['POST'])
def chat():
    if session.get('is_ended', False):
        return jsonify({
            "response": "Session has ended. Please restart to begin a new session.",
            "response_id": None,
            "ended": True
        })

    try:
        data = request.json
        user_message = data['message'][:MAX_MESSAGE_LENGTH]
        previous_response_id = data.get('previous_response_id')

        # Get data from session
        conversation_history = session.get('conversation_history', [])
        negative_count = session.get('negative_count', 0)

        # Sentiment analysis
        blob = TextBlob(user_message)
        sentiment = blob.sentiment.polarity

        # Negative sentiment handling
        if sentiment < NEGATIVE_THRESHOLD:
            negative_count += 1
            session['negative_count'] = negative_count
            if negative_count >= 2:
                session['is_ended'] = True
                response = "I've made it clear that I need to be treated with respect. Since that's not happening, I'm ending this session. I hope you'll reflect on how your words impact others. Goodbye."
                log_conversation(user_message, response, sentiment)
                return jsonify({
                    "response": response,
                    "response_id": None,
                    "ended": True
                })

        # Update conversation history
        conversation_history.append({"role": "user", "content": user_message})

        # Explicit reminder to stay in character
        explicit_reminder = {
            "role": "developer",
            "content": "You are strictly Monae, never break character, respond naturally as a patient, not as an AI."
        }
        conversation_history.insert(-1, explicit_reminder)

        response_text, response_id = chat_with_gpt(conversation_history, previous_response_id)
        conversation_history.append({"role": "assistant", "content": response_text})

        # Clean up explicit reminder
        conversation_history.remove(explicit_reminder)

        # First negative interaction warning
        if sentiment < NEGATIVE_THRESHOLD and negative_count == 1:
            response_text = ("I need you to understand that using my correct name and treating me with respect "
                             "isn't optional—it's essential for this therapy to work. " + response_text)
            negative_count += 0.1
            session['negative_count'] = negative_count

        # Save updated conversation history back to session
        session['conversation_history'] = conversation_history
        session['last_message_time'] = datetime.now().isoformat()

        log_conversation(user_message, response_text, sentiment)

        return jsonify({
            "response": response_text,
            "response_id": response_id,
            "ended": False
        })

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({"error": "An error occurred processing your message"}), 500

@app.route('/download-logs')
def download_logs():
    """Generate and serve chat logs as JSONL file for the current session"""
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({"error": "No session logs available"}), 404
            
        logs = ChatLog.query.filter_by(session_id=session_id).order_by(ChatLog.timestamp).all()
        temp_file = f"session_{session_id}_logs.jsonl"
        
        with open(temp_file, "w") as f:
            for log in logs:
                log_entry = {
                    "timestamp": log.timestamp.isoformat(),
                    "user_message": log.user_message,
                    "assistant_response": log.assistant_response,
                    "sentiment": log.sentiment,
                    "negative_count": log.negative_count
                }
                f.write(json.dumps(log_entry) + "\n")
        
        return send_from_directory(os.getcwd(), temp_file, as_attachment=True)
    except Exception as e:
        logger.error(f"Error generating download file: {e}")
        return jsonify({"error": "Failed to generate logs"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)

