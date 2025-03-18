import os
import google.generativeai as genai

# Configure your API key
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))


def clean_text(text: str) -> str:
    """Simple cleanup to remove odd characters or HTML."""
    import re
    text = re.sub(r'<.*?>', '', text)
    text = text.encode("ascii", "ignore").decode()
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    return text.strip()


def load_background(file_path: str) -> str:
    """Load and clean the background information from a text file."""
    with open(file_path, "r", encoding="utf-8") as file:
        background = clean_text(file.read())
    return background


def chat_with_gemini(user_message: str, background: str, conversation_history):
    """Generate response using Gemini with the given background information and conversation history."""

    model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp')

    # Initialize the chat if this is the first message
    if not conversation_history:
        system_message = f"""
        You are roleplaying as Monae, a 32-year-old trans woman seeking gender-affirming voice therapy.

        {background}

        IMPORTANT INSTRUCTIONS:
        1. You are ONLY Monae, the patient. You are NOT the clinician.
        2. Only respond as Monae would respond to the clinician's questions.
        3. Never switch roles or pretend to be the clinician.
        4. Do not provide meta-commentary or analysis about the conversation.
        5. Stay in character at all times unless explicitly told "cut the character."
        6. If the clinician says something inappropriate, respond as Monae would naturally react.
        7. If disrespected repeatedly, Monae would express discomfort and may leave the conversation.

        The following is a conversation between Monae (you) and a voice therapist (the clinician).
        """
        # Store the system message at the beginning of the conversation
        conversation_history.append({"role": "system", "content": system_message})

    # Build the complete chat history to maintain context
    chat = model.start_chat(history=[])

    # First add the system message
    system_content = next((msg["content"] for msg in conversation_history if msg["role"] == "system"), "")
    if system_content:
        chat.send_message(system_content)

    # Then add the conversation history as alternating user/model messages
    for i in range(1, len(conversation_history)):
        msg = conversation_history[i]
        if msg["role"] == "user":
            chat.send_message(f"Clinician: {msg['content']}")
        elif msg["role"] == "model" or msg["role"] == "assistant":
            # Don't send these to the model, as they're already part of the chat history
            pass

    # Add the new user message
    response = chat.send_message(f"Clinician: {user_message}\n\nRemember, you are Monae. Respond as Monae would:")

    if response.text:
        # Clean up any potential prefixing the model might add
        response_text = response.text
        response_text = response_text.replace("Monae:", "").strip()

        # Sometimes models try to include both roles - let's make sure we only get Monae's response
        if "Clinician:" in response_text:
            response_text = response_text.split("Clinician:")[0].strip()

        # Add the user's message and the response to the conversation history
        conversation_history.append({"role": "user", "content": user_message})
        conversation_history.append({"role": "assistant", "content": response_text})

        return response_text
    else:
        error_message = "No response generated."
        print(error_message)
        return error_message


def main():
    background_path = "background.txt"
    character_background = load_background(background_path)

    # Initialize conversation history
    conversation_history = []

    print("Starting chat with Monae. Type 'quit' or 'exit' to end the conversation.")

    while True:
        user_input = input("\nClinician: ")
        if user_input.lower() in ["quit", "exit"]:
            print("Ending chat. Goodbye.")
            break

        response = chat_with_gemini(user_input, character_background, conversation_history)
        print("\nMonae:", response)


if __name__ == "__main__":
    main()