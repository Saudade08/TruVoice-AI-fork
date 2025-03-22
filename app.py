from flask import Flask, request, jsonify, render_template_string
from textblob import TextBlob
from openai import OpenAI
import os

# Initialize Flask and OpenAI
app = Flask(__name__)
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Global variables for sentiment tracking
conversation_history = []
negative_count = 0
NEGATIVE_THRESHOLD = -0.3
session_ended = False  # New global variable to track session status

def clean_text(text: str) -> str:
    """Simple cleanup to remove odd characters or HTML."""
    text = text.encode("ascii", "ignore").decode()
    return text.strip()

def load_background(file_path: str) -> str:
    """Load and clean the background information from a text file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            background = clean_text(file.read())
        return background
    except FileNotFoundError:
        print(f"Warning: Background file {file_path} not found.")
        return ""
    except Exception as e:
        print(f"Error loading background file: {e}")
        return ""

def chat_with_gpt(input_text: str, previous_response_id=None):
    """Generate response using OpenAI API."""
    try:
        response = client.responses.create(
            model="gpt-4o",
            input=input_text,
            previous_response_id=previous_response_id
        )
        return response.output[0].content[0].text, response.id
    except Exception as e:
        print(f"Error during API call: {e}")
        return "I'm feeling overwhelmed and need to take a break. Let's end our session here.", None

@app.route('/')
def home():
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Voice Therapy Session with Monae</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f3f4f6;
            }
            header {
                background-color: white;
                padding: 1rem;
                border-radius: 8px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                margin-bottom: 1rem;
            }
            h1 {
                margin: 0;
                color: #1f2937;
                font-size: 1.5rem;
            }
            #chatBox {
                height: 500px;
                overflow-y: auto;
                background: white;
                border-radius: 8px;
                padding: 1rem;
                margin-bottom: 1rem;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .input-container {
                display: flex;
                gap: 0.5rem;
            }
            #userInput {
                flex: 1;
                padding: 0.75rem;
                border: 1px solid #d1d5db;
                border-radius: 6px;
                font-size: 1rem;
            }
            button {
                padding: 0.75rem 1.5rem;
                background-color: #2563eb;
                color: white;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-size: 1rem;
            }
            button:hover {
                background-color: #1d4ed8;
            }
            button:disabled {
                background-color: #9ca3af;
                cursor: not-allowed;
            }
            .message {
                margin: 1rem 0;
                padding: 0.5rem 0;
            }
            .user-message {
                text-align: right;
            }
            .message-content {
                display: inline-block;
                padding: 0.75rem 1rem;
                border-radius: 8px;
                max-width: 80%;
            }
            .user-message .message-content {
                background-color: #2563eb;
                color: white;
            }
            .assistant-message .message-content {
                background-color: #f3f4f6;
                color: #1f2937;
            }
            .ended {
                color: #dc2626;
                font-weight: bold;
                text-align: center;
                margin: 1rem 0;
            }
            #restartButton {
                display: none;
                margin: 1rem auto;
                padding: 0.75rem 1.5rem;
                background-color: #10b981;
                color: white;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-size: 1rem;
            }
            #restartButton:hover {
                background-color: #059669;
            }
        </style>
    </head>
    <body>
        <header>
            <h1>Voice Therapy Session with Monae</h1>
        </header>
        <div id="chatBox"></div>
        <div class="input-container">
            <input type="text" id="userInput" placeholder="Type your message..." onkeypress="handleKeyPress(event)">
            <button id="sendButton" onclick="sendMessage()">Send</button>
        </div>
        <button id="restartButton" onclick="restartSession()">Start New Session</button>

        <script>
            let isSessionEnded = false;
            let previousResponseId = null;

            async function sendMessage() {
                if (isSessionEnded) return;
                
                const userInput = document.getElementById('userInput');
                const message = userInput.value.trim();
                if (!message) return;

                const chatBox = document.getElementById('chatBox');
                chatBox.innerHTML += `
                    <div class="message user-message">
                        <div class="message-content">${message}</div>
                    </div>
                `;

                userInput.value = '';
                userInput.disabled = true;
                document.querySelector('#sendButton').disabled = true;

                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            message: message,
                            previous_response_id: previousResponseId
                        })
                    });
                    const data = await response.json();
                    previousResponseId = data.response_id;

                    chatBox.innerHTML += `
                        <div class="message assistant-message">
                            <div class="message-content">${data.response}</div>
                        </div>
                    `;

                    if (data.ended) {
                        endSession();
                    }
                } catch (error) {
                    console.error('Error:', error);
                    chatBox.innerHTML += `
                        <div class="message system-message">
                            <div class="message-content">Error sending message. Please try again.</div>
                        </div>
                    `;
                }

                userInput.disabled = !isSessionEnded;
                document.querySelector('#sendButton').disabled = isSessionEnded;
                chatBox.scrollTop = chatBox.scrollHeight;
            }

            function endSession() {
                isSessionEnded = true;
                document.getElementById('chatBox').innerHTML += '<div class="ended">Session has ended.</div>';
                document.getElementById('userInput').disabled = true;
                document.querySelector('#sendButton').disabled = true;
                document.getElementById('restartButton').style.display = 'block';
            }

            function handleKeyPress(event) {
                if (event.key === 'Enter' && !event.shiftKey && !isSessionEnded) {
                    event.preventDefault();
                    sendMessage();
                }
            }

            async function restartSession() {
                try {
                    const response = await fetch('/restart', {method: 'POST'});
                    const data = await response.json();
                    if (data.success) {
                        isSessionEnded = false;
                        previousResponseId = null;
                        document.getElementById('chatBox').innerHTML = '';
                        document.getElementById('userInput').disabled = false;
                        document.querySelector('#sendButton').disabled = false;
                        document.getElementById('restartButton').style.display = 'none';
                        
                        // Start a new session
                        const startResponse = await fetch('/start', {method: 'POST'});
                        const startData = await startResponse.json();
                        document.getElementById('chatBox').innerHTML += `
                            <div class="message assistant-message">
                                <div class="message-content">${startData.response}</div>
                            </div>
                        `;
                        previousResponseId = startData.response_id;
                    }
                } catch (error) {
                    console.error('Error restarting session:', error);
                }
            }

            // Check session status when page loads
            async function checkSessionStatus() {
                try {
                    const response = await fetch('/status');
                    const data = await response.json();
                    
                    if (data.ended) {
                        isSessionEnded = true;
                        document.getElementById('userInput').disabled = true;
                        document.querySelector('#sendButton').disabled = true;
                        document.getElementById('restartButton').style.display = 'block';
                    } else {
                        // Start the chat if not ended
                        const startResponse = await fetch('/start', {method: 'POST'});
                        const startData = await startResponse.json();
                        document.getElementById('chatBox').innerHTML += `
                            <div class="message assistant-message">
                                <div class="message-content">${startData.response}</div>
                            </div>
                        `;
                        previousResponseId = startData.response_id;
                    }
                } catch (error) {
                    console.error('Error checking session status:', error);
                }
            }

            // Start the chat when the page loads
            window.onload = checkSessionStatus;
        </script>
    </body>
    </html>
    """)

@app.route('/status')
def check_status():
    global session_ended
    return jsonify({"ended": session_ended})

@app.route('/restart', methods=['POST'])
def restart_session():
    global negative_count, session_ended, conversation_history
    negative_count = 0
    session_ended = False
    conversation_history = []
    return jsonify({"success": True})

@app.route('/start', methods=['POST'])
def start_chat():
    global conversation_history, negative_count, session_ended
    
    # If session is already ended, don't proceed
    if session_ended:
        return jsonify({
            "response": "Session has ended. Please restart to begin a new session.",
            "response_id": None,
            "ended": True
        })
    
    background_path = "background.txt"
    background = load_background(background_path)
    negative_count = 0  # Initialize negative sentiment counter

    initial_instructions = f"""
    Chatgpt You are role-playing as Monae, a 32-year-old trans woman seeking gender-affirming voice therapy from a clinician today.
    {background}
    Stay fully in character and respond appropriately to the clinician no matter what happens.
    Talk like a normal person, not a assistant chat bot, no extra fluff, be straight and to the point, no over explaining.
    Your feelings are generally nervous but hopeful. But your feelings should adpat to how the clinician is speaking to you.
    If I ask random or irrelevant questions, Monae should respond naturally—expressing confusion, setting boundaries, or steering the conversation back to voice therapy.
    If I say something inappropriate, Monae should respond as a real person would, with realistic emotional reactions that reflect the disrespect of the clinician. Don't break character unless I say 'cut the character.'
    If the clinician disrespects you deliberately more than once, then leave the conversation. Give appropriate warnings and let the clinician know how you're feeling when something they say changes your mood.
    Model Boundaries: Monae should confidently redirect inappropriate questions to help clinicians understand what is appropriate. 
    Answer appropriately: Yes/No questions should be answered with a Yes/No answer only.
    Encourage Growth: If a clinician genuinely tries to improve, Monae can acknowledge this while reinforcing the need for self-education.
    Advocate for Needs: Monae should model self-advocacy.
    Reinforce Trauma-Informed Care: Monae should challenge dismissive attitudes and highlight the emotional and psychological impact of misgendering.
    """
    
    response_text, response_id = chat_with_gpt(initial_instructions)
    conversation_history = [{"role": "system", "content": initial_instructions}]
    
    return jsonify({"response": response_text, "response_id": response_id, "ended": False})

@app.route('/chat', methods=['POST'])
def chat():
    global conversation_history, negative_count, session_ended
    
    # If session is already ended, don't process new messages
    if session_ended:
        return jsonify({
            "response": "Session has ended. Please restart to begin a new session.",
            "response_id": None,
            "ended": True
        })
    
    data = request.json
    user_message = data['message']
    previous_response_id = data.get('previous_response_id')
    
    # Analyze sentiment
    blob = TextBlob(user_message)
    sentiment = blob.sentiment.polarity
    
    # Check for negative sentiment
    if sentiment < NEGATIVE_THRESHOLD:
        negative_count += 1
        if negative_count >= 2:
            session_ended = True  # Mark the session as ended globally
            response = "I don't feel comfortable continuing this session. I've felt disrespected multiple times now, and I need to prioritize my well-being. Goodbye."
            return jsonify({
                "response": response,
                "response_id": None,
                "ended": True
            })
    
    # Get response from GPT
    response_text, response_id = chat_with_gpt(user_message, previous_response_id)
    
    # Check if this was a negative interaction but not enough to end the session
    if sentiment < NEGATIVE_THRESHOLD and negative_count == 1:
        response_text = "I feel disrespected by that comment. I want to continue our session, but please be more mindful of your words. " + response_text
    
    return jsonify({
        "response": response_text,
        "response_id": response_id,
        "ended": session_ended
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
