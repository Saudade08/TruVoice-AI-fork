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
import tiktoken

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('chat.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ---------------- Flask / DB / OpenAI ----------------
REQUEST_TIMEOUT = 10  # seconds

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///local.db').replace('postgres://', 'postgresql://')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
db = SQLAlchemy(app)
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), timeout=REQUEST_TIMEOUT)

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

with app.app_context():
    db.create_all()

# ---------------- Constants ----------------
NEGATIVE_THRESHOLD = -0.3
MAX_MESSAGE_LENGTH = 500
MAX_CONVERSATION_TURNS = 20
MAX_INPUT_TOKENS = 1000
MAX_OUTPUT_TOKENS = 500

# Single source of truth for the persona prompt (no duplication anywhere else)
SYSTEM_PROMPT = """
You are now Monae, a 32-year-old trans woman at your voice therapy appointment today. This is not a roleplay - you ARE Monae for this entire conversation.
{BACKGROUND}
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
""".strip()

# ---------------- Utilities ----------------
def clean_text(text: str) -> str:
    try:
        text = text.encode("ascii", "ignore").decode()
        return text.strip()
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        return ""

def load_background(file_path: str) -> str:
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

def count_tokens(text: str, model: str = "gpt-4") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        return len(text) // 4  # rough fallback

# ---------------- OpenAI (Responses API) ----------------
def chat_with_gpt(messages: list, previous_response_id: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """
    GPT-5 via Responses API with:
      - First turn: send system prompt (marked cacheable) + user.
      - Later turns: DO NOT resend system prompt; pass previous_response_id and only send the new user turn.
    """
    try:
        input_blocks = []

        if previous_response_id:
            # Subsequent turns: only the new user message(s)
            for m in messages:
                if m.get("role") == "user":
                    input_blocks.append({
                        "role": "user",
                        "content": [{"type": "input_text", "text": m["content"]}]
                    })

            resp = client.responses.create(
                model="gpt-5-chat-latest",
                input=input_blocks,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                stop=["\n\n"],
                temperature=0.8
            )
        else:
            # First turn: include cacheable system block + any user message(s)
            # Build system block once, with background substituted
            background = load_background("background.txt")
            system_text = SYSTEM_PROMPT.replace("{BACKGROUND}", background or "")
            system_block = {
                "role": "system",
                "content": [{"type": "input_text", "text": system_text}],
                "cache_control": {"type": "ephemeral"}  # cache the static persona prompt
            }
            input_blocks.append(system_block)

            for m in messages:
                if m.get("role") == "user":
                    input_blocks.append({
                        "role": "user",
                        "content": [{"type": "input_text", "text": m["content"]}]
                    })

            resp = client.responses.create(
                model="gpt-5-chat-latest",
                input=input_blocks,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                stop=["\n\n"],
                temperature=0.8
            )

        # Parse text robustly
        text = ""
        try:
            text = resp.output[0].content[0].text
        except Exception:
            text = getattr(resp, "output_text", "") or ""

        return (text.strip() if text else ""), getattr(resp, "id", None)

    except Exception as e:
        logger.error("Error during API call", exc_info=True)
        return "I need to take a break from this session. Thank you for understanding.", None

# ---------------- Logging ----------------
def log_conversation(user_message: str, assistant_response: str, sentiment: float) -> None:
    try:
        session_id = session.get('session_id')
        negative_count = session.get('negative_count', 0)
        if not session_id:
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

# ---------------- Routes ----------------
@app.route('/')
def home():
    # You said this must clear the session — keep it.
    session.clear()
    return render_template('index.html')

@app.route('/status')
def check_status():
    return jsonify({
        "ended": session.get('is_ended', False),
        "turns_remaining": MAX_CONVERSATION_TURNS - session.get('conversation_turns', 0)
    })

@app.route('/restart', methods=['POST'])
def restart_session():
    try:
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
        # Start fresh conversation bookkeeping
        session['conversation_turns'] = 0
        session['negative_count'] = 0
        session['is_ended'] = False
        session['last_message_time'] = datetime.now().isoformat()
        session['last_response_id'] = None  # <-- important: we will use server-side conversation
        # We do NOT need to stash the whole system prompt in the session anymore.
        return jsonify({"response": "", "response_id": None, "ended": False})
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
        user_message = (data.get('message') or "")[:MAX_MESSAGE_LENGTH]

        # Token limit for user input
        input_tokens = count_tokens(user_message)
        if input_tokens > MAX_INPUT_TOKENS:
            return jsonify({
                "response": "Your message is too long. Please keep it shorter and more focused.",
                "response_id": None,
                "ended": False
            })

        # Turn limit
        conversation_turns = session.get('conversation_turns', 0)
        if conversation_turns >= MAX_CONVERSATION_TURNS:
            session['is_ended'] = True
            return jsonify({
                "response": "We've reached the end of our session time. Thank you for the conversation. I appreciate your time and will schedule a follow-up if needed.",
                "response_id": None,
                "ended": True
            })

        # Sentiment analysis
        blob = TextBlob(user_message)
        sentiment = blob.sentiment.polarity

        # Negative handling
        negative_count = session.get('negative_count', 0)
        if sentiment < NEGATIVE_THRESHOLD:
            negative_count += 1
            session['negative_count'] = negative_count
            if negative_count >= 2:
                session['is_ended'] = True
                response = ("I've made it clear that I need to be treated with respect. Since that's not happening, "
                            "I'm ending this session. I hope you'll reflect on how your words impact others. Goodbye.")
                log_conversation(user_message, response, sentiment)
                return jsonify({"response": response, "response_id": None, "ended": True})

        # Build a minimal message list for this turn (only the user's new message)
        # The system prompt will be injected automatically INSIDE chat_with_gpt() only on the first turn,
        # and thereafter we rely on previous_response_id to keep context server-side.
        messages = [{"role": "user", "content": user_message}]

        # Retrieve previous response id if any
        previous_response_id = session.get('last_response_id')

        # Call model
        response_text, response_id = chat_with_gpt(messages, previous_response_id=previous_response_id)

        # Save the new previous_response_id so we don't have to resend the system prompt next time
        if response_id:
            session['last_response_id'] = response_id

        # Optional: prepend a brief boundary on first negative
        if sentiment < NEGATIVE_THRESHOLD and negative_count == 1:
            response_text = ("I need you to understand that using my correct name and treating me with respect "
                             "isn't optional—it's essential for this therapy to work. " + response_text)

        # Update counters/metadata
        session['conversation_turns'] = conversation_turns + 1
        session['last_message_time'] = datetime.now().isoformat()

        # Log
        log_conversation(user_message, response_text, sentiment)

        return jsonify({
            "response": response_text,
            "response_id": response_id,
            "ended": False,
            "turns_remaining": MAX_CONVERSATION_TURNS - (conversation_turns + 1)
        })

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        return jsonify({"error": "An error occurred processing your message"}), 500

@app.route('/download-logs')
def download_logs():
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

