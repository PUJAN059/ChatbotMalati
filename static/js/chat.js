/**
 * Malati Chatbot — Frontend Logic
 * Handles messaging, UI state, theme, clear chat, and error handling.
 */

"use strict";

/* ── DOM references ─────────────────────────────────────────── */
const messagesArea  = document.getElementById("messagesArea");
const userInput     = document.getElementById("userInput");
const sendBtn       = document.getElementById("sendBtn");
const typingWrapper = document.getElementById("typingWrapper");
const themeToggle   = document.getElementById("themeToggle");
const clearBtn      = document.getElementById("clearBtn");
const modalOverlay  = document.getElementById("modalOverlay");
const modalCancel   = document.getElementById("modalCancel");
const modalConfirm  = document.getElementById("modalConfirm");
const headerStatus  = document.getElementById("headerStatus");
const toast         = document.getElementById("toast");
const quickActions  = document.getElementById("quickActions");
const bgCanvas      = document.getElementById("bgCanvas");

/* ── State ──────────────────────────────────────────────────── */
let isBotTyping  = false;
let toastTimeout = null;

/* ── Generate starfield background ──────────────────────────── */
(function initStars() {
  if (!bgCanvas) return;
  const count = 80;
  for (let i = 0; i < count; i++) {
    const star = document.createElement("div");
    star.className = "star";
    star.style.left = Math.random() * 100 + "%";
    star.style.top = Math.random() * 100 + "%";
    star.style.setProperty("--dur", (1.5 + Math.random() * 3) + "s");
    star.style.animationDelay = Math.random() * 3 + "s";
    const size = 1 + Math.random() * 2;
    star.style.width = size + "px";
    star.style.height = size + "px";
    bgCanvas.appendChild(star);
  }
})();

/* ── Theme management ───────────────────────────────────────── */
(function initTheme() {
  const saved = localStorage.getItem("malati-theme") || "dark";
  document.documentElement.setAttribute("data-theme", saved);
})();

themeToggle.addEventListener("click", () => {
  const current = document.documentElement.getAttribute("data-theme");
  const next    = current === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  localStorage.setItem("malati-theme", next);
  showToast(next === "light" ? "☀️ Light mode" : "🌙 Dark mode");
});

/* ── Utility: format timestamp ──────────────────────────────── */
function formatTime(date = new Date()) {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

/* ── Utility: escape HTML to prevent XSS ───────────────────── */
function escapeHTML(str) {
  return str
    .replace(/&/g,  "&amp;")
    .replace(/</g,  "&lt;")
    .replace(/>/g,  "&gt;");
}

/* ── Utility: auto-resize textarea ─────────────────────────── */
function autoResize() {
  userInput.style.height = "auto";
  userInput.style.height = Math.min(userInput.scrollHeight, 120) + "px";
}

/* ── Utility: scroll to bottom ─────────────────────────────── */
function scrollToBottom(smooth = true) {
  messagesArea.scrollTo({
    top: messagesArea.scrollHeight,
    behavior: smooth ? "smooth" : "instant",
  });
}

/* ── Utility: toast notification ───────────────────────────── */
function showToast(message, duration = 2200) {
  clearTimeout(toastTimeout);
  toast.textContent = message;
  toast.classList.add("show");
  toastTimeout = setTimeout(() => toast.classList.remove("show"), duration);
}

/* ── Utility: update send button state ─────────────────────── */
function updateSendBtn() {
  sendBtn.disabled = userInput.value.trim() === "" || isBotTyping;
}

/* ── Utility: detect and handle URLs ──────────────────────── */
function extractURLs(text) {
  const urlRegex = /(?:https?:\/\/)?(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:\/[^\s,]*)?/gi;
  const matches = text.match(urlRegex) || [];
  return matches.filter(m => m.includes("."));
}

function formatMessageText(text) {
  let safe = escapeHTML(text);
  safe = safe.replace(
    /(https?:\/\/[^\s&]+|www\.[^\s&]+|([a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:\/[^\s&]*))?)/gi,
    (match) => {
      let url = match;
      if (!url.startsWith("http")) url = "https://" + url;
      return '<a href="' + url + '" target="_blank" rel="noopener" style="color: var(--primary); text-decoration: underline; cursor: pointer; font-weight: 500;">' + match + '</a>';
    }
  );
  return safe;
}

function isOpenCommand(text) {
  const lower = text.toLowerCase().trim();
  return /^(open|go to|visit|navigate to|browse to|can you open|open this|open that)\b/.test(lower);
}

/* ── Build a message row element ────────────────────────────── */
function createMessageRow(text, sender, isError = false) {
  const isUser = sender === "user";

  const row = document.createElement("div");
  row.className = `message-row ${isUser ? "user-row" : "bot-row"}`;

  // Small avatar circle
  const avatar = document.createElement("div");
  avatar.className = "row-avatar";
  avatar.setAttribute("aria-hidden", "true");
  if (isUser) {
    avatar.textContent = "🧑";
  } else {
    avatar.innerHTML = `<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:100%;">
      <ellipse cx="60" cy="95" rx="28" ry="18" fill="#6366f1" opacity="0.9"/>
      <circle cx="60" cy="45" r="30" fill="#fbbf24"/>
      <ellipse cx="42" cy="52" rx="5" ry="3.5" fill="#f9a8d4" opacity="0.5"/>
      <ellipse cx="78" cy="52" rx="5" ry="3.5" fill="#f9a8d4" opacity="0.5"/>
      <g class="malati-eyes">
        <ellipse cx="48" cy="42" rx="4" ry="4.5" fill="#1e293b"/>
        <ellipse cx="72" cy="42" rx="4" ry="4.5" fill="#1e293b"/>
        <circle cx="49.5" cy="40.5" r="1.5" fill="#fff"/>
        <circle cx="73.5" cy="40.5" r="1.5" fill="#fff"/>
      </g>
      <path d="M50 54 Q60 63 70 54" fill="none" stroke="#1e293b" stroke-width="2.2" stroke-linecap="round"/>
      <g class="malati-wave-arm">
        <path d="M32 80 Q18 60 14 42" fill="none" stroke="#fbbf24" stroke-width="6" stroke-linecap="round"/>
        <circle cx="14" cy="40" r="6" fill="#fbbf24"/>
        <g class="malati-wave-fingers">
          <line x1="11" y1="35" x2="9" y2="28" stroke="#fbbf24" stroke-width="2.5" stroke-linecap="round"/>
          <line x1="14" y1="34" x2="14" y2="26" stroke="#fbbf24" stroke-width="2.5" stroke-linecap="round"/>
          <line x1="17" y1="35" x2="19" y2="28" stroke="#fbbf24" stroke-width="2.5" stroke-linecap="round"/>
        </g>
      </g>
      <path d="M88 80 Q98 72 102 65" fill="none" stroke="#fbbf24" stroke-width="6" stroke-linecap="round"/>
      <circle cx="102" cy="64" r="5" fill="#fbbf24"/>
    </svg>`;
  }

  // Bubble + timestamp wrapper
  const group = document.createElement("div");
  group.className = "bubble-group";

  const bubble = document.createElement("div");
  bubble.className = "bubble" + (isError ? " error" : "");
  if (isUser || isError) {
    bubble.textContent = text;
  } else {
    bubble.innerHTML = formatMessageText(text);
  }

  const ts = document.createElement("span");
  ts.className = "timestamp";
  ts.setAttribute("aria-label", `Sent at ${formatTime()}`);
  ts.textContent = formatTime();

  group.appendChild(bubble);
  group.appendChild(ts);

  if (isUser) {
    row.appendChild(group);
    row.appendChild(avatar);
  } else {
    row.appendChild(avatar);
    row.appendChild(group);
  }

  return row;
}

/* ── Append a system / date separator ──────────────────────── */
function appendSystemMessage(text) {
  const div = document.createElement("div");
  div.className = "system-message";
  div.innerHTML = `<span>${escapeHTML(text)}</span>`;
  messagesArea.appendChild(div);
}

/* ── Typing indicator visibility ────────────────────────────── */
function showTyping() {
  typingWrapper.classList.add("visible");
  typingWrapper.setAttribute("aria-hidden", "false");
  headerStatus.textContent = "Typing…";
  headerStatus.style.color = "var(--text-muted)";
  scrollToBottom();
}

function hideTyping() {
  typingWrapper.classList.remove("visible");
  typingWrapper.setAttribute("aria-hidden", "true");
  headerStatus.textContent = "Hey! Ready to chat 😊";
  headerStatus.style.color = "";
}


/* ── Core: send message ─────────────────────────────────────── */
async function sendMessage() {
  const text = userInput.value.trim();
  if (!text || isBotTyping) return;

  // Clear input immediately
  userInput.value = "";
  autoResize();
  updateSendBtn();

  // Append user bubble
  const userRow = createMessageRow(text, "user");
  messagesArea.appendChild(userRow);
  scrollToBottom();

  // Auto-open URLs found anywhere in the message
  const urls = extractURLs(text);
  if (urls.length > 0) {
    urls.forEach((u) => {
      let url = u;
      if (!url.startsWith("http")) url = "https://" + url;
      window.open(url, "_blank");
    });
    showToast("🔗 Opening link...");
    // Don't send to bot — just open the URL
    return;
  }

  // If it starts with "open" + a domain, extract and open
  if (isOpenCommand(text)) {
    const urls2 = extractURLs(text);
    if (urls2.length > 0) {
      urls2.forEach((u) => {
        let url = u;
        if (!url.startsWith("http")) url = "https://" + url;
        window.open(url, "_blank");
      });
      showToast("🔗 Opening link...");
      return;
    }
  }

  // Show typing indicator
  isBotTyping = true;
  updateSendBtn();
  showTyping();

  // Simulate a short, natural delay (400–900 ms) before replying
  const thinkMs = 400 + Math.random() * 500;

  try {
    const [response] = await Promise.all([
      fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      }),
      new Promise((r) => setTimeout(r, thinkMs)),
    ]);

    hideTyping();
    isBotTyping = false;
    updateSendBtn();

    if (!response.ok) {
      // Server returned an error status
      let errMsg = "Oops! Something went wrong on my end. Try again! 🔧";
      try {
        const errData = await response.json();
        if (errData.response) errMsg = errData.response;
      } catch (_) { /* ignore */ }

      const errRow = createMessageRow(errMsg, "bot", true);
      messagesArea.appendChild(errRow);
    } else {
      const data = await response.json();
      const botRow = createMessageRow(data.response, "bot");
      messagesArea.appendChild(botRow);
    }
  } catch (networkErr) {
    // Network failure (offline, server down, etc.)
    hideTyping();
    isBotTyping = false;
    updateSendBtn();

    const errRow = createMessageRow(
      "I can't reach the server right now. Make sure the app is running and try again! 🔌",
      "bot",
      true
    );
    messagesArea.appendChild(errRow);
  }

  scrollToBottom();
}

/* ── Clear chat flow ────────────────────────────────────────── */
clearBtn.addEventListener("click", () => {
  // Don't open modal if there's nothing to clear
  if (messagesArea.children.length === 0) {
    showToast("Nothing to clear! 💬");
    return;
  }
  modalOverlay.removeAttribute("hidden");
  modalConfirm.focus();
});

modalCancel.addEventListener("click", () => {
  modalOverlay.setAttribute("hidden", "");
});

modalOverlay.addEventListener("click", (e) => {
  if (e.target === modalOverlay) modalOverlay.setAttribute("hidden", "");
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !modalOverlay.hasAttribute("hidden")) {
    modalOverlay.setAttribute("hidden", "");
  }
});

modalConfirm.addEventListener("click", async () => {
  modalOverlay.setAttribute("hidden", "");

  try {
    await fetch("/api/clear", { method: "POST" });
  } catch (_) { /* best-effort — clear UI regardless */ }

  // Wipe messages with a fade-out
  messagesArea.style.opacity = "0";
  messagesArea.style.transition = "opacity 0.25s ease";
  setTimeout(() => {
    messagesArea.innerHTML = "";
    messagesArea.style.opacity = "1";
    appendWelcomeMessage();
    showToast("🗑️ Chat cleared");
  }, 260);
});

/* ── Welcome message ────────────────────────────────────────── */
function appendWelcomeMessage() {
  appendSystemMessage("Today · Start of conversation");

  const welcomeRow = createMessageRow(
    "Hey there! 👋 I'm Malati, built by Pujan Subedi.\nI'm not that advanced, but I can try to make your day! Ask me anything — jokes, fun facts, or just say hi! 😊",
    "bot"
  );
  messagesArea.appendChild(welcomeRow);
  scrollToBottom(false);
}

/* ── Quick action buttons ───────────────────────────────────── */
quickActions.addEventListener("click", (e) => {
  const btn = e.target.closest(".quick-btn");
  if (!btn) return;
  const msg = btn.getAttribute("data-msg");
  if (msg) {
    userInput.value = msg;
    autoResize();
    updateSendBtn();
    sendMessage();
  }
});

/* ── Input event listeners ──────────────────────────────────── */
userInput.addEventListener("input", () => {
  autoResize();
  updateSendBtn();
});

userInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    if (!sendBtn.disabled) sendMessage();
  }
});

sendBtn.addEventListener("click", sendMessage);

/* ── Bootstrap ──────────────────────────────────────────────── */
appendWelcomeMessage();
updateSendBtn();
userInput.focus();
