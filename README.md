# SheeBot – AI Wellness Assistant

An AI-powered wellness assistant that combines habit tracking, conversational support, and personalized insights to help users improve their daily well-being.

---

## Features

* Conversational AI chatbot with natural, human-like responses
* Water intake tracking
* Sleep monitoring
* Mood tracking
* Interactive dashboards & visual analytics
* Weekly personalised reports
* AI-generated personalized insights
* Habit tracking with streak system

---

## Why this project?

Most wellness apps only track data — they don’t actually *talk* to you.

This Chatbot bridges that gap by combining:

* Data tracking (water, sleep, mood)
* Conversational AI support
* Personalized insights

This makes wellness tracking more engaging, interactive, and human-like.

---

## Screenshots

### Overview

![Overview](overview.png)

### 📊 Dashboard

![Dashboard](dashboard.png)

### 📈 Wellness Charts

![Charts](charts.png)

### Weekly Narrative (AI Insights)

![Weekly Narrative](weekly-narrative.png)

---

## Tech Stack

* **Backend:** Python, Flask
* **AI Model:** LLaMA3 (via Ollama)
* **Frontend:** HTML, CSS, JavaScript
* **Data Storage:** JSON (local logs & memory)
* **Visualization:** Chart-based dashboards

---

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Sheema-S/SheeBot-Wellness-Chatbot.git
cd SheeBot-Wellness-Chatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Ollama (required for AI)

```bash
ollama serve
ollama run llama3
```

### 4. Run the app

```bash
python app.py
```

### 5. Open in browser

```
http://127.0.0.1:5000
```

---

## Example Prompts

* “I drank 2L water today”
* “I slept 6 hours last night”
* “I feel stressed”
* “Show my weekly report”
* “How can I improve my sleep?”

---

## Future Improvements

* User authentication (multi-user support)
* Cloud database integration
* Mobile responsiveness
* Smarter AI personalization
* Voice interaction support

---

## Author

**Sheema S**

---

## ⭐ If you like this project

Give it a star ⭐ on GitHub!
