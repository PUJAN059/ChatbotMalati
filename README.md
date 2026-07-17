# Aria — AI Chatbot

A modern conversational chatbot built for casual, everyday interactions. Powered by Python and NLP, with a polished messaging-style frontend.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1-lightgrey?logo=flask)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-orange)
![ML](https://img.shields.io/badge/ML-scikit--learn-f7931e?logo=scikit-learn)

---

## Features

- Clean, modern chat interface inspired by popular messaging apps
- User and bot message bubbles with smooth entry animations
- Real-time conversation with a natural typing delay
- Typing indicator while the bot is composing a reply
- Auto-scroll to the latest message
- Responsive design — works on desktop, tablet, and mobile
- Send messages with **Enter** · new line with **Shift+Enter**
- Timestamps shown on hover for every message
- Friendly welcome message on first load
- Clear chat button with confirmation modal
- Toast notifications for UI actions
- **Light and dark mode** with system preference memory
- Graceful error handling for network failures and server errors

## Chatbot Capabilities

Aria can:

- Greet users naturally and engage in small talk
- Tell jokes 😂
- Share fun facts 🌍
- Respond to compliments and handle rude messages gracefully
- Answer simple questions about itself
- Remember the last few messages during a session
- Fall back gracefully on unknown inputs

## Tech Stack

| Layer     | Technology                          |
|-----------|-------------------------------------|
| Backend   | Python 3.10+, Flask                 |
| NLP / ML  | TF-IDF vectorisation, cosine similarity (scikit-learn) |
| Frontend  | Vanilla HTML, CSS, JavaScript (no frameworks) |
| Data      | JSON intents file                   |

## Project Structure

```
Chatbot/
├── app.py                  # Flask API server
├── requirements.txt
├── chatbot/
│   ├── __init__.py
│   └── engine.py           # NLP chatbot engine
├── data/
│   └── intents.json        # Training intents, patterns & responses
├── templates/
│   └── index.html          # Chat UI
└── static/
    ├── css/style.css
    └── js/chat.js
```

## Getting Started

### 1. Clone / download the project

```bash
git clone <repo-url>
cd Chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the server

```bash
python app.py
```

### 5. Open in your browser

```
http://localhost:5000
```

## API Reference

| Method | Endpoint       | Description                        |
|--------|----------------|------------------------------------|
| GET    | `/`            | Serves the chat frontend           |
| POST   | `/api/chat`    | Send a message, receive a response |
| POST   | `/api/clear`   | Clear conversation history         |
| GET    | `/api/history` | Retrieve current chat history      |
| GET    | `/api/health`  | Health check                       |

### POST `/api/chat`

**Request body:**
```json
{ "message": "Tell me a joke!" }
```

**Response:**
```json
{
  "response": "Why do programmers prefer dark mode? Because light attracts bugs! 🐛💡😄",
  "intent": "joke",
  "confidence": 0.872
}
```

## Colour Palette

| Token      | Dark mode | Light mode |
|------------|-----------|------------|
| Background | `#0F172A` | `#F0F4FF`  |
| Surface    | `#1E293B` | `#FFFFFF`  |
| Primary    | `#3B82F6` | `#3B82F6`  |
| Accent     | `#22C55E` | `#16A34A`  |
| Text       | `#F8FAFC` | `#0F172A`  |

## License

MIT — free to use, modify, and share.
