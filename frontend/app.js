const API_BASE_URL = window.location.origin.startsWith('http')
    ? window.location.origin
    : 'http://127.0.0.1:5000';

// DOM elements
const form = document.getElementById('uploadForm');
const fileInput = document.getElementById('fileInput');
const resultDiv = document.getElementById('result');
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const chatWindow = document.getElementById('chat-window');
const chatStatus = document.getElementById('chat-status');

let currentDisease = "";

// Check backend health on page load
window.addEventListener('DOMContentLoaded', async () => {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        if (!data.model_loaded) {
            chatStatus.textContent = 'Warning: Model not loaded on server';
            chatStatus.style.color = '#d32f2f';
        }
    } catch (error) {
        chatStatus.textContent = 'Cannot connect to backend server';
        chatStatus.style.color = '#d32f2f';
    }
});

// Handle image upload and disease detection
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!fileInput.files[0]) {
        alert('Please select an image file');
        return;
    }

    // Validate file type
    const file = fileInput.files[0];
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp', 'image/tiff'];
    if (!allowedTypes.includes(file.type)) {
        alert('Invalid file type. Please upload PNG, JPG, JPEG, BMP, or TIFF images.');
        return;
    }

    // Validate file size (10MB max)
    if (file.size > 10 * 1024 * 1024) {
        alert('File too large. Maximum size is 10MB.');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    // Show loading state
    resultDiv.classList.add('active');
    resultDiv.innerHTML = `
        <div class="loading-container">
            <div class="loading-spinner"></div>
            <div class="loading-text">Analyzing Image...</div>
        </div>
    `;
    chatStatus.textContent = 'Analyzing image...';
    chatStatus.style.color = '';
    chatInput.disabled = true;
    chatForm.querySelector('button').disabled = true;

    try {
        const response = await fetch(`${API_BASE_URL}/predict_json`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        // Handle different error status codes
        if (!response.ok) {
            let errorMessage = data.error || 'Detection failed';
            if (response.status === 503) {
                errorMessage = 'Server model not available. Please contact administrator.';
            } else if (response.status === 400) {
                errorMessage = data.error || 'Invalid request';
            }
            throw new Error(errorMessage);
        }

        currentDisease = (data.diseases && data.diseases[0]) || "";
        
        // Build disease list if multiple detected
        const allDiseases = data.diseases && data.diseases.length > 1
            ? data.diseases.map(d => `<li>${d}</li>`).join('')
            : '';

        // Display results
        resultDiv.innerHTML = `
            <div class="result-container">
                <div class="result-image-wrapper">
                    <img src="data:image/jpeg;base64,${data.image_b64}" alt="Detected Result" />
                </div>
                <div class="result-info">
                    <div class="disease-title">Detected Disease:</div>
                    <div class="disease-badge">${currentDisease || 'No disease detected'}</div>
                    ${allDiseases ? `
                        <div class="disease-title" style="margin-top: 20px;">All Detected:</div>
                        <div class="disease-list"><ul>${allDiseases}</ul></div>
                    ` : ''}
                </div>
            </div>
        `;

        // Enable chat if disease detected
        if (currentDisease) {
            chatStatus.textContent = `Ready to help with ${currentDisease}`;
            chatStatus.style.color = '#138A36';
            chatInput.disabled = false;
            chatForm.querySelector('button').disabled = false;
            chatWindow.innerHTML = '';
            addChatMessage('bot', `I've detected ${currentDisease} in your leaf. How can I help you treat this disease?`);
        } else {
            chatStatus.textContent = 'No disease detected - Leaf appears healthy!';
            chatStatus.style.color = '#138A36';
        }
    } catch (error) {
        console.error('Detection error:', error);
        resultDiv.innerHTML = `
            <div style="text-align: center; padding: 20px; color: #d32f2f;">
                <h3>❌ Error</h3>
                <p>${error.message}</p>
                <p style="font-size: 0.9em; margin-top: 10px;">Please try again or check your connection.</p>
            </div>
        `;
        chatStatus.textContent = 'Detection failed';
        chatStatus.style.color = '#d32f2f';
    }
});

// Add message to chat window
function addChatMessage(sender, text) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-msg ${sender}`;
    messageDiv.textContent = text;
    chatWindow.appendChild(messageDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

// Handle chat messages
chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const message = chatInput.value.trim();
    
    if (!message) return;
    
    if (!currentDisease) {
        alert('Please upload and analyze an image first');
        return;
    }

    if (message.length > 500) {
        alert('Message too long. Please keep it under 500 characters.');
        return;
    }

    addChatMessage('user', message);
    chatInput.value = '';
    chatForm.querySelector('button').disabled = true;

    try {
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.reply || 'Failed to get response');
        }
        
        addChatMessage('bot', data.reply || 'No response received');
    } catch (error) {
        console.error('Chat error:', error);
        addChatMessage('bot', '❌ Unable to connect to chatbot. Please check your connection and try again.');
    } finally {
        chatForm.querySelector('button').disabled = false;
        chatInput.focus();
    }
});