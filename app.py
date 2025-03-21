import os
import re
import sys
from flask import Flask, request, jsonify, render_template_string
from textblob import TextBlob
from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

app = Flask(__name__)


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
                const userInput = document.getElementById('userInput').value;
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: userInput})
                });
                const data = await response.json();
                document.getElementById('chatResponse').textContent = data.response;
            }
        </script>
    </head>
    <body>
        <h1>Monae Chatbot</h1>
        <input type=\"text\" id=\"userInput\" placeholder=\"Type your message...\" style=\"width: 300px;\">
        <button onclick=\"sendMessage()\">Send</button>
        <p><strong>Response:</strong></p>
        <div id=\"chatResponse\" style=\"white-space: pre-wrap; margin-top: 10px;\"></div>
    </body>
    </html>
    """)


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data['message']
    response_text, _ = chat_with_gpt(user_input)
    return jsonify({"response": response_text})


def main():
    background_path = "background.txt"
    background = load_background(background_path)
    negative_threshold = -0.3
    negative_count = 0

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

    print("Starting chat with Monae. Type 'quit' or 'exit' to end the conversation.")
    response_text, previous_id = chat_with_gpt(initial_instructions)
    print("\nMonae:", response_text)

    while True:
        user_input = input("\nClinician: ")
        if user_input.lower() in ["quit", "exit"]:
            print("Ending chat. Goodbye.")
            break

        blob = TextBlob(user_input)
        sentiment = blob.sentiment.polarity

        if sentiment < negative_threshold:
            negative_count += 1

        if negative_count >= 2:
            print("\nMonae: This conversation has become too negative too often. I'm ending this session now. Goodbye.")
            sys.exit(0)

        response_text, previous_id = chat_with_gpt(user_input, previous_id)
        print("\nMonae:", response_text)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'cli':
        main()
    else:
        app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
