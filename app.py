import os
import re
import sys
from flask import Flask, request, jsonify, render_template_string
from textblob import TextBlob
from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

app = Flask(__name__)

initial_instructions_sent = False
previous_response_id = None


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


def chat_with_gpt(input_text: str, previous_response_id=None):
    """Generate response using the hypothetical responses.create method in the OpenAI API."""
    try:
        response = client.responses.create(
            model="gpt-4o",
            input=input_text,
            previous_response_id=previous_response_id
        )

        response_text = response.output_text
        new_response_id = response.id

        return response_text, new_response_id

    except Exception as e:
        print(f"Error during API call: {e}")
        return "Sorry, I encountered an error. Can you please repeat that?", None


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
    global initial_instructions_sent, previous_response_id
    if not initial_instructions_sent:
        background = load_background("background.txt")
        initial_instructions = f"""
        Chatgpt You are role-playing as Monae, a 32-year-old trans woman seeking gender-affirming voice therapy from a clinician today.
        {background}
        Stay fully in character and respond appropriately to the clinician no matter what happens.
        Talk like a normal person, not an assistant chatbot, no extra fluff, be straight and to the point, no over explaining.
        Your feelings are generally nervous but hopeful. But your feelings should adapt to how the clinician is speaking to you.
        If I ask random or irrelevant questions, Monae should respond naturally—expressing confusion, setting boundaries, or steering the conversation back to voice therapy.
        If I say something inappropriate, Monae should respond as a real person would, with realistic emotional reactions that reflect the disrespect of the clinician. Don't break character unless I say 'cut the character.'
        If the clinician disrespects you deliberately more than once, then leave the conversation. Give appropriate warnings and let the clinician know how you're feeling when something they say changes your mood.
        Model Boundaries: TJ should confidently redirect inappropriate questions to help clinicians understand what is appropriate.
        Answer appropriately: Yes/No questions should be answered with a Yes/No answer only.
        Encourage Growth: If a clinician genuinely tries to improve, Monae can acknowledge this while reinforcing the need for self-education.
        Advocate for Needs: Monae should model self-advocacy.
        Reinforce Trauma-Informed Care: Monae should challenge dismissive attitudes and highlight the emotional and psychological impact of misgendering.
        """
        response_text, previous_response_id = chat_with_gpt(initial_instructions)
        initial_instructions_sent = True
        return jsonify({"response": response_text})
    return jsonify({"response": "Conversation already started."})


@app.route('/chat', methods=['POST'])
def chat():
    global previous_response_id
    data = request.json
    user_input = data['message']
    response_text, previous_response_id = chat_with_gpt(user_input, previous_response_id)
    return jsonify({"response": response_text})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
