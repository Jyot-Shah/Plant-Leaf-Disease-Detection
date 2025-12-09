const form = document.getElementById('uploadForm');
const fileInput = document.getElementById('fileInput');
const resultDiv = document.getElementById('result');

const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const chatWindow = document.getElementById('chat-window');
const chatStatus = document.getElementById('chat-status');

let currentDisease = "";

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!fileInput.files[0]) return;

    const fd = new FormData();
    fd.append('file', fileInput.files[0]);

    resultDiv.classList.add('active');
    resultDiv.innerHTML = `
        <div class="loading-container">
            <div class="loading-spinner"></div>
            <div class="loading-text">Analyzing Image...</div>
        </div>
    `;
    chatStatus.textContent = 'Analyzing image...';
    chatInput.disabled = true;
    chatForm.querySelector('button').disabled = true;

    try {
        const res = await fetch('http://127.0.0.1:5000/predict_json', {
            method: 'POST',
            body: fd
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Detection failed');

        currentDisease = (data.diseases && data.diseases[0]) || "";
        const allDiseases = data.diseases && data.diseases.length > 1
            ? data.diseases.map(d => `<li>${d}</li>`).join('')
            : '';

        resultDiv.innerHTML = `
            <div class="result-container">
                <div class="result-image-wrapper">
                    <img src="data:image/jpeg;base64,${data.image_b64}" alt="Result" />
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

        if (currentDisease) {
            chatStatus.textContent = `Ready to help with ${currentDisease}`;
            chatInput.disabled = false;
            chatForm.querySelector('button').disabled = false;
            chatWindow.innerHTML = '';
            addChatMessage('bot', `I've detected ${currentDisease} in your leaf. How can I help you treat this disease?`);
        } else {
            chatStatus.textContent = 'No disease detected - Leaf appears healthy!';
        }
    } catch (err) {
        resultDiv.innerHTML = `<h3>Error: ${err.message}</h3>`;
        chatStatus.textContent = 'Detection failed';
    }
});

function addChatMessage(author, text) {
    const div = document.createElement('div');
    div.className = `chat-msg ${author}`;
    div.textContent = text;
    chatWindow.appendChild(div);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const msg = chatInput.value.trim();
    if (!msg || !currentDisease) return;

    addChatMessage('user', msg);
    chatInput.value = '';
    chatForm.querySelector('button').disabled = true;

    try {
        const res = await fetch('http://127.0.0.1:5000/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: msg })
        });
        const data = await res.json();
        addChatMessage('bot', data.reply || 'No reply');
    } catch (err) {
        addChatMessage('bot', 'Error contacting chatbot.');
    } finally {
        chatForm.querySelector('button').disabled = false;
    }
});