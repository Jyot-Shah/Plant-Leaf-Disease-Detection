// Target local Python Flask Instance
const API_BASE_URL = "http://127.0.0.1:5000";

// DOM elements
const form = document.getElementById("uploadForm");
const fileInput = document.getElementById("fileInput");
const dropZone = document.getElementById("dropZone");
const dropZonePrompt = document.getElementById("dropZonePrompt");
const resultDiv = document.getElementById("result");

const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const chatWindow = document.getElementById("chat-window");
const chatStatus = document.getElementById("chat-status");

let currentDisease = sessionStorage.getItem("currentDisease") || "";
let currentSessionId = sessionStorage.getItem("currentSessionId") || null;

// Drag and drop logic for index page
if (dropZone && fileInput) {
  fileInput.addEventListener("change", (e) => {
    if (fileInput.files.length) {
      dropZonePrompt.textContent = fileInput.files[0].name;
      dropZone.classList.add("dragover");
    }
  });

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  ["dragleave", "dragend"].forEach((type) => {
    dropZone.addEventListener(type, (e) => {
      dropZone.classList.remove("dragover");
    });
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    if (e.dataTransfer.files.length) {
      fileInput.files = e.dataTransfer.files;
      dropZonePrompt.textContent = e.dataTransfer.files[0].name;
      dropZone.classList.add("dragover");
    }
  });
}

// Check backend health on page load and initialize chat if on assistant page
window.addEventListener("DOMContentLoaded", async () => {
  if (chatStatus) {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      const data = await response.json();
      if (!data.model_loaded) {
        chatStatus.textContent = "Warning: Model not loaded on server";
        chatStatus.style.color = "#d32f2f";
      }
    } catch (error) {
      chatStatus.textContent = "Cannot connect to backend server";
      chatStatus.style.color = "#d32f2f";
    }
  }

  // Init chat on assistant page
  if (chatForm && currentDisease) {
    chatStatus.textContent = `Ready to help with ${currentDisease}`;
    chatStatus.style.color = "#138A36";
    chatInput.disabled = false;
    chatForm.querySelector("button").disabled = false;
    chatWindow.innerHTML = "";
    addChatMessage(
      "bot",
      `I've detected ${currentDisease} in your leaf. How can I help you treat this disease?`,
    );
  } else if (chatForm) {
    chatStatus.textContent = "Access Restricted";
    chatStatus.style.color = "#d32f2f";
    chatWindow.innerHTML = `
      <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; text-align: center; gap: 15px; padding: 20px;">
        <i class="fa-solid fa-lock" style="font-size: 3em; color: #d32f2f;"></i>
        <h3 style="color: #d32f2f; margin: 0;">No Analysis Found</h3>
        <p style="color: #0D1B13; margin: 0;">Please upload a leaf image on the Detection page first to start an analysis.</p>
        <a href="index.html" class="redirect-button" style="width: auto;">Go to Detection <i class="fa-solid fa-arrow-right"></i></a>
      </div>
    `;
    chatForm.style.display = "none";
  }
});

// Handle image upload and disease detection
if (form) {
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!fileInput.files[0]) {
      alert("Please select an image file");
      return;
    }

    // Validate file type
    const file = fileInput.files[0];
    const allowedTypes = [
      "image/png",
      "image/jpeg",
      "image/jpg",
      "image/bmp",
      "image/tiff",
    ];
    if (!allowedTypes.includes(file.type)) {
      alert(
        "Invalid file type. Please upload PNG, JPG, JPEG, BMP, or TIFF images.",
      );
      return;
    }

    // Validate file size (10MB max)
    if (file.size > 10 * 1024 * 1024) {
      alert("File too large. Maximum size is 10MB.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    // Show loading state and scroll slightly down
    resultDiv.classList.add("active");
    resultDiv.innerHTML = `
          <div class="loading-container">
              <div class="loading-spinner"></div>
              <div class="loading-text">Analyzing Image...</div>
          </div>
      `;

    // Disable submit button during request
    const submitBtn = form.querySelector('button[type="submit"]');
    if (submitBtn) {
      submitBtn.disabled = true;
      submitBtn.textContent = "Analyzing...";
    }

    setTimeout(() => {
      resultDiv.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }, 100);

    try {
      const response = await fetch(`${API_BASE_URL}/predict_json`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      // Handle different error status codes
      if (!response.ok) {
        let errorMessage = data.error || "Detection failed";
        if (response.status === 503) {
          errorMessage =
            "Server model not available. Please contact administrator.";
        } else if (response.status === 400) {
          errorMessage = data.error || "Invalid request";
        }
        throw new Error(errorMessage);
      }

      currentDisease = (data.diseases && data.diseases[0]) || "";
      currentSessionId = data.session_id || null;

      if (currentDisease) {
        sessionStorage.setItem("currentDisease", currentDisease);
      }
      if (currentSessionId) {
        sessionStorage.setItem("currentSessionId", currentSessionId);
      }

      // Display results
      resultDiv.innerHTML = `
              <div class="result-container">
                  <div class="result-image-wrapper">
                      <img src="data:image/jpeg;base64,${data.image_b64}" alt="Detected Result" />
                  </div>
                  <div class="result-info">
                      <div class="disease-title">Detected Disease:</div>
                      <div class="disease-badge">
                        ${currentDisease || "No disease detected"}
                      </div>

                      ${
                        currentDisease
                          ? `
                          <a href="assistant.html" class="redirect-button">
                              Get Treatment Advice <i class="fa-solid fa-arrow-right"></i>
                          </a>
                        `
                          : ""
                      }
                  </div>
              </div>
          `;

      // Smooth scroll to the result
      setTimeout(() => {
        resultDiv.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }, 100);
    } catch (error) {
      console.error("Detection error:", error);
      resultDiv.innerHTML = `
              <div style="text-align: center; padding: 20px; color: #d32f2f;">
                  <h3>❌ Error</h3>
                  <p>${error.message}</p>
                  <p style="font-size: 0.9em; margin-top: 10px;">Please try again or check your connection.</p>
              </div>
          `;
    } finally {
      if (submitBtn) {
        submitBtn.disabled = false;
        submitBtn.textContent = "Analyze Image";
      }
    }
  });
}

// Add message to chat window
function addChatMessage(sender, text) {
  if (!chatWindow) return;
  const messageDiv = document.createElement("div");
  messageDiv.className = `chat-msg ${sender}`;
  messageDiv.textContent = text;
  chatWindow.appendChild(messageDiv);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

// Handle chat messages from assistant page
if (chatForm) {
  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const message = chatInput.value.trim();

    if (!message) return;

    if (!currentDisease) {
      alert("Please go to the Detection page and analyze an image first.");
      return;
    }

    if (message.length > 500) {
      alert("Message too long. Please keep it under 500 characters.");
      return;
    }

    addChatMessage("user", message);
    chatInput.value = "";
    chatForm.querySelector("button").disabled = true;

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, session_id: currentSessionId }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.reply || "Failed to get response");
      }

      addChatMessage("bot", data.reply || "No response received");
    } catch (error) {
      console.error("Chat error:", error);
      addChatMessage(
        "bot",
        "❌ Unable to connect to chatbot. Please check your connection and try again.",
      );
    } finally {
      chatForm.querySelector("button").disabled = false;
      chatInput.focus();
    }
  });
}
