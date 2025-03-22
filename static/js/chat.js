class ChatManager {
  constructor() {
    this.isSessionEnded = false;
    this.previousResponseId = null;
    this.setupElements();
    this.setupEventListeners();
    this.checkSessionStatus();
  }

  setupElements() {
    this.chatBox = document.getElementById('chatBox');
    this.userInput = document.getElementById('userInput');
    this.sendButton = document.getElementById('sendButton');
    this.restartButton = document.getElementById('restartButton');
    this.typingIndicator = document.getElementById('typingIndicator');
  }

  setupEventListeners() {
    this.sendButton.addEventListener('click', () => this.sendMessage());
    this.userInput.addEventListener('keypress', (e) => this.handleKeyPress(e));
    this.restartButton.addEventListener('click', () => this.restartSession());
    
    // Auto-resize textarea
    this.userInput.addEventListener('input', () => {
      this.userInput.style.height = 'auto';
      this.userInput.style.height = this.userInput.scrollHeight + 'px';
    });
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
}

// Initialize chat when the page loads
document.addEventListener('DOMContentLoaded', () => {
  window.chatManager = new ChatManager();
});
