class ChatManager {
  constructor() {
    this.isSessionEnded = false;
    this.previousResponseId = null;
    this.setupElements();
    this.setupEventListeners();
    // Don't check session status immediately - wait for patient selection
  }

  setupElements() {
    // Welcome screen elements
    this.welcomeScreen = document.getElementById('welcomeScreen');
    this.patientCards = document.querySelectorAll('.patient-card');
    
    // Chat interface elements
    this.chatInterface = document.getElementById('chatInterface');
    this.backButton = document.getElementById('backButton');
    this.currentPatientName = document.getElementById('currentPatientName');
    this.chatBox = document.getElementById('chatBox');
    this.userInput = document.getElementById('userInput');
    this.sendButton = document.getElementById('sendButton');
    this.restartButton = document.getElementById('restartButton');
    this.downloadButton = document.getElementById('downloadButton');
    this.typingIndicator = document.getElementById('typingIndicator');
    this.typingName = document.getElementById('typingName');
  }

  setupEventListeners() {
    // Welcome screen events
    this.patientCards.forEach(card => {
      card.addEventListener('click', () => this.selectPatient(card.dataset.patient));
    });
    
    // Chat interface events
    this.backButton.addEventListener('click', () => this.returnToWelcomeScreen());
    this.sendButton.addEventListener('click', () => this.sendMessage());
    this.userInput.addEventListener('keypress', (e) => this.handleKeyPress(e));
    this.restartButton.addEventListener('click', () => this.restartSession());
    this.downloadButton.addEventListener('click', () => this.downloadConversation());
    
    // Auto-resize textarea
    this.userInput.addEventListener('input', () => {
      this.userInput.style.height = 'auto';
      this.userInput.style.height = this.userInput.scrollHeight + 'px';
    });
  }

  selectPatient(patientId) {
    // Store selected patient
    this.currentPatient = patientId;
    
    // Update UI for selected patient
    if (patientId === 'monae') {
      this.currentPatientName.textContent = 'Voice Interview Session with Monae';
      this.typingName.textContent = 'Monae';
    }
    // Add more patient options here in the future
    
    // Switch from welcome screen to chat interface
    this.welcomeScreen.classList.add('hidden');
    this.chatInterface.classList.remove('hidden');
    
    // Start session
    this.checkSessionStatus();
  }

  returnToWelcomeScreen() {
    // If session is active, confirm before leaving
    if (!this.isSessionEnded) {
      if (!confirm("Are you sure you want to end this session?")) {
        return;
      }
    }
    
    // Reset chat interface
    this.chatBox.innerHTML = '';
    this.isSessionEnded = false;
    this.previousResponseId = null;
    
    // Switch from chat interface to welcome screen
    this.chatInterface.classList.add('hidden');
    this.welcomeScreen.classList.remove('hidden');
    
    // Enable inputs
    this.userInput.disabled = false;
    this.sendButton.disabled = false;
    this.restartButton.classList.add('hidden');
  }

  async checkSessionStatus() {
    try {
      const response = await this.fetchWithTimeout('/status');
      const data = await response.json();
      
      if (data.ended) {
        this.endSession();
      } else {
        await this.startNewSession();
      }
    } catch (error) {
      this.handleError('Error checking session status');
    }
  }

  async sendMessage() {
    if (this.isSessionEnded || !this.userInput.value.trim()) return;

    const message = this.userInput.value.trim();
    this.userInput.value = '';
    this.userInput.style.height = 'auto';

    this.addMessage(message, 'user');
    this.setLoading(true);

    try {
      const response = await this.fetchWithTimeout('/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          message,
          previous_response_id: this.previousResponseId
        })
      });

      const data = await response.json();
      this.previousResponseId = data.response_id;
      
      this.addMessage(data.response, 'assistant');
      
      if (data.ended) {
        this.endSession();
      }
    } catch (error) {
      this.handleError('Failed to send message');
    } finally {
      this.setLoading(false);
    }
  }

  async startNewSession() {
    try {
      const response = await this.fetchWithTimeout('/start', {
        method: 'POST'
      });
      const data = await response.json();
      this.addMessage(data.response, 'assistant');
      this.previousResponseId = data.response_id;
    } catch (error) {
      this.handleError('Failed to start new session');
    }
  }

  async restartSession() {
    try {
      const response = await this.fetchWithTimeout('/restart', {
        method: 'POST'
      });
      const data = await response.json();
      
      if (data.success) {
        this.isSessionEnded = false;
        this.previousResponseId = null;
        this.chatBox.innerHTML = '';
        this.userInput.disabled = false;
        this.sendButton.disabled = false;
        this.restartButton.classList.add('hidden');
        this.downloadButton.classList.add('hidden');
        await this.startNewSession();
      }
    } catch (error) {
      this.handleError('Failed to restart session');
    }
  }

  addMessage(content, role) {
    const timestamp = new Date().toLocaleTimeString();
    const messageHtml = `
      <div class="message message-${role}">
        <div class="message-content">
          <p>${this.escapeHtml(content)}</p>
          <div class="message-timestamp">${timestamp}</div>
        </div>
      </div>
    `;
    this.chatBox.insertAdjacentHTML('beforeend', messageHtml);
    this.scrollToBottom();
  }

  endSession() {
    this.isSessionEnded = true;
    this.chatBox.insertAdjacentHTML('beforeend', '<div class="session-ended">Session has ended.</div>');
    this.userInput.disabled = true;
    this.sendButton.disabled = true;
    this.restartButton.classList.remove('hidden');
    this.downloadButton.classList.remove('hidden');
  }

  setLoading(isLoading) {
    this.userInput.disabled = isLoading;
    this.sendButton.disabled = isLoading;
    this.typingIndicator.classList.toggle('hidden', !isLoading);
  }

  handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey && !this.isSessionEnded) {
      event.preventDefault();
      this.sendMessage();
    }
  }

  handleError(message) {
    console.error(message);
    this.chatBox.insertAdjacentHTML('beforeend', `
      <div class="message message-error">
        <div class="message-content">
          <p>${message}. Please try again.</p>
        </div>
      </div>
    `);
  }

  scrollToBottom() {
    this.chatBox.scrollTop = this.chatBox.scrollHeight;
  }

  escapeHtml(unsafe) {
    return unsafe
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  async fetchWithTimeout(url, options = {}) {
    const timeout = 10000; // 10 seconds
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeout);

    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal
      });
      clearTimeout(id);
      return response;
    } catch (error) {
      clearTimeout(id);
      throw error;
    }
  }
  
  downloadConversation() {
    // Get all messages from the chat box
    const messages = this.chatBox.querySelectorAll('.message');
    
    // Determine if we should create HTML or plain text
    const useHTML = true; // Set to true for Word-like formatting
    
    if (useHTML) {
      this.downloadAsHTML(messages);
    } else {
      this.downloadAsText(messages);
    }
  }
  
  downloadAsHTML(messages) {
    const currentDate = new Date();
    const formattedDate = currentDate.toLocaleDateString();
    const patientName = this.currentPatient === 'monae' ? 'Monae' : this.currentPatient;
    
    // Create HTML content
    let htmlContent = `
      <!DOCTYPE html>
      <html>
      <head>
        <meta charset="UTF-8">
        <title>TruVision AI - Session Transcript</title>
        <style>
          body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.5;
          }
          .header {
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid #ccc;
          }
          .header h1 {
            margin-bottom: 5px;
            color: #2563eb;
          }
          .header p {
            margin: 5px 0;
            color: #666;
          }
          .message {
            margin-bottom: 20px;
            page-break-inside: avoid;
          }
          .speaker {
            font-weight: bold;
            margin-bottom: 5px;
          }
          .clinician {
            color: #2563eb;
          }
          .patient {
            color: #10b981;
          }
          .content {
            background-color: #f9f9f9;
            padding: 10px 15px;
            border-radius: 8px;
            border-left: 4px solid #ddd;
          }
          .clinician-content {
            border-left-color: #2563eb;
          }
          .patient-content {
            border-left-color: #10b981;
          }
          .timestamp {
            font-size: 0.8em;
            color: #888;
            text-align: right;
          }
          .session-end {
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ccc;
            color: #888;
            font-style: italic;
          }
        </style>
      </head>
      <body>
        <div class="header">
          <h1>TruVision AI - Session Transcript</h1>
          <p>Date: ${formattedDate}</p>
          <p>Patient: ${patientName}</p>
        </div>
    `;
    
    messages.forEach((message) => {
      // Get role (user or assistant)
      const isUser = message.classList.contains('message-user');
      const role = isUser ? 'You (Clinician)' : patientName;
      const roleClass = isUser ? 'clinician' : 'patient';
      const contentClass = isUser ? 'clinician-content' : 'patient-content';
      
      // Get message content and timestamp
      const content = message.querySelector('p')?.textContent || '';
      const timestamp = message.querySelector('.message-timestamp')?.textContent || '';
      
      htmlContent += `
        <div class="message">
          <div class="speaker ${roleClass}">${role}</div>
          <div class="content ${contentClass}">${content}</div>
          <div class="timestamp">${timestamp}</div>
        </div>
      `;
    });
    
    // End the HTML document
    htmlContent += `
        <div class="session-end">End of Session</div>
      </body>
      </html>
    `;
    
    // Create download link
    const element = document.createElement('a');
    const blob = new Blob([htmlContent], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    
    element.setAttribute('href', url);
    element.setAttribute('download', `TruVision-Session-${currentDate.toISOString().slice(0,10)}.html`);
    
    // Hide link, add to body, click it, then remove it
    element.style.display = 'none';
    document.body.appendChild(element);
    element.click();
    
    // Clean up
    setTimeout(() => {
      document.body.removeChild(element);
      URL.revokeObjectURL(url);
    }, 100);
  }
  
  downloadAsText(messages) {
    // Create text content for download
    let conversationText = `TruVision AI - Session Transcript\n`;
    conversationText += `=================================\n\n`;
    conversationText += `Date: ${new Date().toLocaleDateString()}\n`;
    conversationText += `Patient: ${this.currentPatient === 'monae' ? 'Monae' : this.currentPatient}\n\n`;
    conversationText += `=================================\n\n`;
    
    messages.forEach((message) => {
      // Get role (user or assistant)
      const isUser = message.classList.contains('message-user');
      const role = isUser ? 'You (Clinician)' : this.currentPatient === 'monae' ? 'Monae' : 'Patient';
      
      // Get message content and timestamp
      const content = message.querySelector('p')?.textContent || '';
      const timestamp = message.querySelector('.message-timestamp')?.textContent || '';
      
      conversationText += `${role} [${timestamp}]\n`;
      conversationText += `${content}\n\n`;
      conversationText += `----------------------------------\n\n`;
    });
    
    conversationText += `=================================\n`;
    conversationText += `End of Session\n`;
    
    // Create download link
    const element = document.createElement('a');
    element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(conversationText));
    element.setAttribute('download', `TruVision-Session-${new Date().toISOString().slice(0,10)}.txt`);
    
    // Hide link, add to body, click it, then remove it
    element.style.display = 'none';
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  }
}

// Initialize chat when the page loads
document.addEventListener('DOMContentLoaded', () => {
  window.chatManager = new ChatManager();
});
