# TruVoice-AI
# ChatGPT Role-Playing Bot - Monae

## Overview
This project implements a Python-based chatbot that simulates conversations with a character named Monae, a 32-year-old trans woman seeking gender-affirming voice therapy. The bot uses OpenAI's GPT-4 model to generate responses that are sensitive, context-aware, and adhere to the set scenario. The conversation dynamic adapts based on the clinician's approach, aiming to provide a realistic and educational experience for users simulating the role of a clinician.

## Features
- **Contextual Role-Playing**: Monae stays in character throughout interactions, providing responses based on a scripted background and dynamic conversation flow.
- **Sentiment Analysis**: Utilizes TextBlob to monitor the sentiment of the conversation, providing cues to Monae to adapt the conversation flow or terminate it based on negative sentiment.
- **Dynamic Response ID Tracking**: Maintains the context of conversation by tracking response IDs for continuous dialogue progression.

## Installation

To run this project, you'll need Python 3.8 or higher. Clone this repository and install the required Python libraries:

```bash
git clone https://TruVoice-AI.git
cd TruVoiceAI
pip install -r requirements.txt

Usage
To start the application, run the following command in the terminal:

python app.py

Follow the on-screen prompts to interact with Monae. Type 'quit' or 'exit' to end the conversation.

export OPENAI_API_KEY='your-api-key-here'
