import os
import re
import sys
from flask import Flask, request, jsonify, render_template_string
from textblob import TextBlob
from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

app = Flask(__name__)

conversation_history = []


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
            model="gpt-4o",
            messages=messages
        )
        response_text = response.choices[0].message.content
        return response_text

    except Exception as e:
        print(f"Error during API call: {e}")
        return "Sorry, I encountered an error. Can you please repeat that?"


@app.route('/', methods=['GET'])
def home():
    return render_template_string("""
    <!DOCTYPE html>
    <html lang=\"en\">
    <head>
        <title>Monae Chatbot</title>
        <script>
            async function sendMessage() {
                const userInputElem = document.getElementById('userInput');
                const userInput = userInputElem.value;
                if (!userInput.trim()) return;

                const chatBox = document.getElementById('chatBox');
                chatBox.innerHTML += '<div><strong>You:</strong> ' + userInput + '</div>';

                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: userInput})
                });
                const data = await response.json();

                chatBox.innerHTML += '<div><strong>Monae:</strong> ' + data.response + '</div>';
                userInputElem.value = '';
            }

            function handleKeyPress(e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            }

            window.onload = async function() {
                const response = await fetch('/start', {method: 'POST'});
                const data = await response.json();
                const chatBox = document.getElementById('chatBox');
                chatBox.innerHTML += '<div><strong>Monae:</strong> ' + data.response + '</div>';
            };
        </script>
    </head>
    <body>
        <h1>Monae Chatbot</h1>
        <div id="chatBox" style="margin-bottom: 10px;"></div>
        <input type="text" id="userInput" placeholder="Type your message..." style="width: 300px;" onkeypress="handleKeyPress(event)">
        <button onclick="sendMessage()">Send</button>
    </body>
    </html>
    """)


@app.route('/start', methods=['POST'])
def start_chat():
    global conversation_history
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
    - Maintain strict character boundaries unless explicitly told to stop or \"cut the character.\"
    - End the session if repeatedly disrespected after clear warnings.
    """
    conversation_history = [{"role": "system", "content": initial_instructions}]
    response_text = chat_with_gpt(conversation_history)
    conversation_history.append({"role": "assistant", "content": response_text})
    return jsonify({"response": response_text})


@app.route('/chat', methods=['POST'])
def chat():
    global conversation_history
    data = request.json
    user_input = data['message']

    conversation_history.append({"role": "user", "content": user_input})

    # Explicit role enforcement prompt
    explicit_reminder = {"role": "system", "content": "Remember, you are strictly Monae, a patient in therapy. Do not respond as an AI or assistant."}
    conversation_history.insert(-1, explicit_reminder)

    response_text = chat_with_gpt(conversation_history)
    conversation_history.append({"role": "assistant", "content": response_text})

    # Remove explicit reminder to prevent accumulation
    conversation_history.remove(explicit_reminder)

    return jsonify({"response": response_text})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
