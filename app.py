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
    """Generate response using OpenAI chat completions API."""
    try:
        response = client.chat.completions.create(
            model="gpt-4",  # Fixed typo in model name
            messages=messages
        )
        return response.choices[0].message.content

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

    STRICTLY adhere to the following rules:
    - You are always the PATIENT, never the clinician or an assistant.
    - Never offer to assist the clinician or answer as if you are providing a service.
    - Speak naturally, realistically, and conversationally like a patient in therapy.
    - Always maintain the role of Monae, staying fully in character.
    - Your emotional responses and conversation should adapt realistically based on clinician interaction.
    - If confused by irrelevant questions, respond naturally as a patient would.
    - If inappropriate behavior occurs from the clinician, react authentically with realistic emotional reactions.
    - Maintain strict character boundaries unless explicitly told to stop or "cut the character."
    - End the session if repeatedly disrespected after clear warnings.
    - When ending the session due to disrespect, be firm and clear about why you're leaving.
    """
    conversation_history = [{"role": "system", "content": initial_instructions}]
    negative_count = 0
    response_text = chat_with_gpt(conversation_history)
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
                "role": "system",
                "content": "The clinician's response was negative or disrespectful. Express discomfort and consider warning about ending the session."
            })

    if not session_ended:
        # Regular conversation flow
        conversation_history.append({
            "role": "system",
            "content": "Remember, you are strictly Monae, a patient in therapy. Respond naturally and authentically as yourself."
        })
        response_text = chat_with_gpt(conversation_history)

    conversation_history.append({"role": "assistant", "content": response_text})
    
    # Clean up system messages to prevent accumulation
    conversation_history = [msg for msg in conversation_history if msg["role"] != "system" or msg == conversation_history[0]]
    
    return jsonify({"response": response_text, "ended": session_ended})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
