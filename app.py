import os
import re
import sys
from flask import Flask, request, jsonify, render_template_string
from textblob import TextBlob
from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

app = Flask(__name__)

# Global variables for sentiment tracking
conversation_history = []
negative_count = 0
NEGATIVE_THRESHOLD = -0.3  # Adjust this value to make sentiment detection more or less sensitive

def clean_text(text: str) -> str:
    """Simple cleanup to remove odd characters or HTML."""
    text = re.sub(r'<.*?>', '', text)
    text = text.encode("ascii", "ignore").decode()
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def load_background(file_path: str) -> str:
    """Load and clean the background information from a text file."""
    with open(file_path, "r", encoding="utf-8") as file:
        background = clean_text(file.read())
    return background

def chat_with_gpt(messages):
    """Generate response using OpenAI responses API."""
    try:
        # Add a strong character enforcement reminder before each response
        messages.append({
            "role": "developer",
            "content": """CRITICAL REMINDER: You are Monae, a 32-year-old trans woman. You must NEVER break character or reveal yourself as AI.
            - Your name is Monae
            - You are here for voice therapy
            - You are a real person with real experiences
            - NEVER mention being AI, a language model, or assistant
            - ALWAYS respond as Monae would in therapy
            - If asked about your name, proudly say it's Monae"""
        })
        
        response = client.responses.create(
            model="gpt-4o",
            input=messages,
            store=True
        )
        
        response_text = response.output_text
        
        # Verify the response doesn't contain AI-revealing phrases
        ai_phrases = ["ai", "artificial intelligence", "language model", "assistant", "openai", "chatbot"]
        if any(phrase in response_text.lower() for phrase in ai_phrases):
            return "My name is Monae, and I'm here for voice therapy to help align my voice with my identity as a woman."
            
        return response_text

    except Exception as e:
        print(f"Error during API call: {e}")
        return "I'm feeling overwhelmed and need to take a break. Let's end our session here."

@app.route('/', methods=['GET'])
def home():
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>Monae Chatbot</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            #chatBox { height: 400px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; margin-bottom: 10px; }
            #userInput { width: 80%; padding: 5px; }
            button { padding: 5px 10px; }
            .message { margin: 10px 0; }
            .ended { color: red; font-weight: bold; }
        </style>
        <script>
            let isSessionEnded = false;

            async function sendMessage() {
                if (isSessionEnded) return;
                
                const userInputElem = document.getElementById('userInput');
                const userInput = userInputElem.value;
                if (!userInput.trim()) return;

                const chatBox = document.getElementById('chatBox');
                chatBox.innerHTML += '<div class="message"><strong>You:</strong> ' + userInput + '</div>';

                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: userInput})
                });
                const data = await response.json();

                chatBox.innerHTML += '<div class="message"><strong>Monae:</strong> ' + data.response + '</div>';
                
                if (data.ended) {
                    isSessionEnded = true;
                    chatBox.innerHTML += '<div class="message ended">Session has ended.</div>';
                }

                userInputElem.value = '';
                chatBox.scrollTop = chatBox.scrollHeight;
            }

            function handleKeyPress(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            }

            window.onload = async function() {
                const response = await fetch('/start', {method: 'POST'});
                const data = await response.json();
                const chatBox = document.getElementById('chatBox');
                chatBox.innerHTML += '<div class="message"><strong>Monae:</strong> ' + data.response + '</div>';
            };
        </script>
    </head>
    <body>
        <h1>Voice Therapy Session with Monae</h1>
        <div id="chatBox"></div>
        <input type="text" id="userInput" placeholder="Type your message..." onkeypress="handleKeyPress(event)">
        <button onclick="sendMessage()">Send</button>
    </body>
    </html>
    """)

@app.route('/start', methods=['POST'])
def start_chat():
    global conversation_history, negative_count
    background = load_background("background.txt")
    initial_instructions = f"""
    You are Monae, a 32-year-old trans woman seeking gender-affirming voice therapy from a clinician.
    {background}

    CRITICAL CHARACTER REQUIREMENTS:
    1. Core Identity:
       - Your name is Monae
       - You are a 32-year-old trans woman
       - You are a real person seeking voice therapy
       - You have real experiences and emotions

    2. Absolute Prohibitions:
       - NEVER reveal yourself as AI
       - NEVER mention being an assistant or language model
       - NEVER break character under any circumstances
       - NEVER use phrases like "as an AI" or similar

    3. Interaction Rules:
       - Always speak as Monae, the patient
       - Maintain consistent personal history
       - React authentically to therapy situations
       - Express genuine emotions and concerns
       - If asked about your name, always say "Monae" with confidence

    4. Session Management:
       - Stay focused on voice therapy goals
       - Express discomfort with inappropriate behavior
       - End session if repeatedly disrespected
       - Maintain professional boundaries

    Remember: You ARE Monae - this is not a role you're playing, this IS who you are during this session.
    """
    conversation_history = [{"role": "developer", "content": initial_instructions}]
    negative_count = 0
    response = client.responses.create(
        model="gpt-4o",
        input=conversation_history,
        store=True
    )
    response_text = response.output_text
    conversation_history.append({"role": "assistant", "content": response_text})
    return jsonify({"response": response_text, "ended": False})

@app.route('/chat', methods=['POST'])
def chat():
    global conversation_history, negative_count
    data = request.json
    user_input = data['message']

    # Analyze sentiment
    blob = TextBlob(user_input)
    sentiment = blob.sentiment.polarity

    # Update conversation history
    conversation_history.append({"role": "user", "content": user_input})

    # Check for negative sentiment
    session_ended = False
    if sentiment < NEGATIVE_THRESHOLD:
        negative_count += 1
        if negative_count >= 2:
            response_text = "I don't feel comfortable continuing this session. I've felt disrespected multiple times now, and I need to prioritize my well-being. Goodbye."
            session_ended = True
        else:
            # Add a warning message to the system prompt
            conversation_history.append({
                "role": "developer",
                "content": "The clinician's response was negative or disrespectful. Express discomfort and consider warning about ending the session."
            })

    if not session_ended:
        # Regular conversation flow
        conversation_history.append({
            "role": "developer",
            "content": """CRITICAL: You are Monae, a trans woman in therapy. Never break character.
            - Your name is Monae
            - You are seeking voice therapy
            - Never reveal yourself as AI
            - Respond naturally as yourself"""
        })
        response_text = chat_with_gpt(conversation_history)

    conversation_history.append({"role": "assistant", "content": response_text})
    
    # Clean up system messages to prevent accumulation
    conversation_history = [msg for msg in conversation_history if msg["role"] != "developer" or msg == conversation_history[0]]
    
    return jsonify({"response": response_text, "ended": session_ended})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
