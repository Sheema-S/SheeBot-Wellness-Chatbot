#!/usr/bin/env python3
import os
import re
import json
import math
import hashlib
import threading
import traceback
from functools import wraps
from datetime import datetime, timedelta

from flask import Flask, render_template, request, jsonify

# optional imports that may not be present in every environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    from flask_cors import CORS
except Exception:
    CORS = None

import requests
import dateparser

# ========= Optional deps for better memory (graceful fallback) =========
try:
    from sentence_transformers import SentenceTransformer
    _SBERT_AVAILABLE = True
except Exception:
    SentenceTransformer = None
    _SBERT_AVAILABLE = False

# ========= Flask app (explicit folders for robustness) =========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)
if CORS:
    CORS(app)

# ========= Files & Folders =========
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "user_logs.jsonl")     # append-only
DATA_FILE = os.path.join(LOG_DIR, "user_data.json")
MEMORY_FILE = os.path.join(LOG_DIR, "memory.jsonl")
STATE_FILE = os.path.join(LOG_DIR, "state.json")        # small ephemeral state
PROFILE_FILE = os.path.join(LOG_DIR, "profile.json")
os.makedirs(LOG_DIR, exist_ok=True)

# Single process-wide file lock for safe writes
_FILE_LOCK = threading.Lock()

# ========= Small utils =========
def now_iso():
    return datetime.now().isoformat()

def read_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows

def append_jsonl(path, obj):
    with _FILE_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def read_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def write_json(path, obj):
    with _FILE_LOCK:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

def normalize_quotes(s: str) -> str:
    return (s or "").replace("’", "'").replace("“", '"').replace("”", '"')

# ========= Embeddings (fast fallback if no sentence-transformers) =========
_EMBED_DIM = 384
_sbert_model = None

def get_sbert():
    global _SBERT_AVAILABLE, _sbert_model
    if _SBERT_AVAILABLE and _sbert_model is None:
        try:
            _sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except Exception:
            _SBERT_AVAILABLE = False
            _sbert_model = None
    return _sbert_model

def cheap_hash_embed(text, dim=_EMBED_DIM):
    tokens = re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
    vec = [0.0] * dim
    for t in tokens:
        h = int.from_bytes(hashlib.blake2b(t.encode(), digest_size=8).digest(), "big")
        idx = h % dim
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]

def embed_text(text):
    m = get_sbert()
    if m is not None:
        v = m.encode([text or ""])[0].tolist()
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]
    return cheap_hash_embed(text or "")

def cos_sim(a, b):
    return sum(x * y for x, y in zip(a, b))

# ========= Chat Logs (short-term memory) =========
def save_log(user_input, bot_response):
    append_jsonl(LOG_FILE, {
        "timestamp": now_iso(),
        "user_input": user_input,
        "bot_response": bot_response
    })

def recent_dialogue(n_pairs=6):
    """
    Returns recent natural dialogue only.
    Filters out UI-ish or system-style bot messages that pollute context.
    """
    rows = read_jsonl(LOG_FILE)
    tail = rows[-(n_pairs * 3):]  # fetch a bit more, then filter
    lines = []
    for r in tail:
        ui = (r.get("user_input", "") or "").strip()
        br = (r.get("bot_response", "") or "").strip()

        if ui:
            lines.append(f"User: {ui}")

        if br:
            # Skip summary/UI noise
            if br.startswith("📅"):
                continue
            if br.startswith("🧠"):
                continue
            if br.startswith("🧼"):
                continue
            if br.startswith("✅ Profile saved"):
                continue
            lines.append(f"Bot: {br}")

    return "\n".join(lines[-2 * n_pairs:])

# ========= Long-term memory store & retrieve =========
KEYWORDS_IMPORTANT = {
    "headache", "migraine", "stress", "anxious", "anxiety", "depressed", "depression",
    "mood", "happy", "sad", "goal", "gym", "workout", "walk", "meditate", "meditation",
    "journal", "sleep", "slept", "water", "liters", "diet", "eat", "meal", "doctor",
    "appointment", "injury", "pain", "trigger", "allergy", "allergic", "caffeine"
}
YESLIKE = {"yes", "yeah", "yep", "sure", "ok", "okay", "sounds good", "please", "do it", "go ahead"}

def importance_score(text):
    score = 0.3
    t = (text or "").lower()
    hits = sum(1 for k in KEYWORDS_IMPORTANT if k in t)
    score += min(0.05 * hits, 0.4)
    if re.search(r"\b\d+(\.\d+)?\b", t):
        score += 0.15
    if any(w in t for w in ["i feel", "i am", "i'm", "today", "my goal", "i want"]):
        score += 0.15
    return min(score, 1.0)

def write_memory(text, tags=None, force=False):
    imp = importance_score(text)
    if not force and imp < 0.5:
        return None
    mem = {
        "timestamp": now_iso(),
        "text": text,
        "tags": tags or [],
        "importance": imp
    }
    append_jsonl(MEMORY_FILE, mem)
    return mem

def search_memory(query, top_k=5, min_sim=0.25):
    qv = embed_text(query)
    items = read_jsonl(MEMORY_FILE)
    scored = []
    for m in items:
        ev = embed_text(m.get("text", ""))
        sim = cos_sim(qv, ev)
        if sim >= min_sim:
            scored.append((sim, m))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:top_k]]

# ========= Health data storage (with memory facts) =========
def _safe_iso_to_datetime(date_str: str) -> datetime:
    try:
        if "T" in date_str:
            return datetime.fromisoformat(date_str)
        return datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        return datetime.now()

def update_user_data(user_input, manual_date=None):
    user_input = normalize_quotes(user_input or "")
    data = read_json(DATA_FILE, {"water": [], "sleep": [], "mood": []})

    base_datetime = datetime.strptime(manual_date, "%Y-%m-%d") if manual_date else datetime.now()
    parsed_full_date = dateparser.parse(user_input, settings={"RELATIVE_BASE": base_datetime}) or base_datetime

    # DO NOT split on '.' to preserve decimals
    parts = re.split(r"\band\b|,", user_input, flags=re.IGNORECASE)

    mood_keywords = {
        "happy": "😀", "joyful": "😁", "excited": "🤩",
        "sad": "😔", "depressed": "😞", "anxious": "😟",
        "angry": "😠", "tired": "😴", "calm": "😌",
        "lonely": "😢", "stressed": "😣", "nervous": "😰",
        "good": "🙂", "fine": "🙂"
    }

    def overwrite_entry(entries, key, value, date_str):
        entries[:] = [e for e in entries if _safe_iso_to_datetime(e["timestamp"]).date().isoformat() != date_str]
        ts = _safe_iso_to_datetime(date_str).isoformat()
        entries.append({key: value, "timestamp": ts})

    facts = []
    lowered = user_input.lower()

    goal_match = re.search(r"\b(my goal is|i (want|plan) to)\b(.+)", lowered)
    if goal_match:
        text_goal = goal_match.group(3).strip()
        if text_goal:
            write_memory(f"User goal: {text_goal}", tags=["goal", "preference"], force=True)

    pref_match = re.search(r"\bi (don't) (like|eat|drink)\b(.+)", lowered)
    if pref_match:
        pref = pref_match.group(3).strip()
        write_memory(f"User preference: avoids {pref}", tags=["preference"], force=True)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        ts = dateparser.parse(part, settings={"RELATIVE_BASE": parsed_full_date}) or parsed_full_date
        date_str = ts.date().isoformat()

        # 💧 water
        water_match = re.search(
            r"(\d+(?:\.\d+)?)\s*((?:liters?|litres?|ltr?s?|l|ml|milliliters?|millilitres?|cups?))\b",
            part,
            re.IGNORECASE,
        )
        if water_match:
            amount = float(water_match.group(1))
            unit = water_match.group(2).lower()
            if unit.startswith("ml") or unit.startswith("millil"):
                amount /= 1000.0
            elif unit.startswith("cup"):
                amount = amount * 240.0 / 1000.0
            overwrite_entry(data["water"], "amount", amount, date_str)
            facts.append(f"User drank {amount:.2f}L water on {date_str}.")

        # 🌙 sleep
        sleep_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)\b", part, re.IGNORECASE)
        if re.search(r"\bsleep|slept\b", part.lower()) and sleep_match:
            hours = float(sleep_match.group(1))
            overwrite_entry(data["sleep"], "hours", hours, date_str)
            facts.append(f"User slept {hours:.2f} hours on {date_str}.")

        # 😊 mood
        for word, emoji in mood_keywords.items():
            if re.search(rf"\b{re.escape(word)}\b", part, re.IGNORECASE):
                data["mood"].append({"text": word, "emoji": emoji, "timestamp": ts.isoformat()})
                facts.append(f"User felt {word} {emoji} on {date_str}.")
                break

        # symptoms (store as memory)
        if any(s in part.lower() for s in ["headache", "migraine", "nausea", "fatigue", "tired", "anxious", "anxiety", "stressed", "stress", "pain"]):
            facts.append(f"Symptom noted on {date_str}: {part.strip()}")

    write_json(DATA_FILE, data)

    for f in facts:
        write_memory(f, tags=["health", "metric"])

    if any(w in lowered for w in ["headache", "migraine"]):
        set_state({"last_topic": "symptom_support"})
    elif "sleep" in lowered and re.search(r"\d", lowered):
        set_state({"last_topic": "sleep_log"})
    elif "water" in lowered and re.search(r"\d", lowered):
        set_state({"last_topic": "water_log"})

# ========= Tiny state (for yes/ok follow-ups) =========
def get_state():
    return read_json(STATE_FILE, {})

def set_state(patch: dict):
    s = get_state()
    s.update(patch or {})
    write_json(STATE_FILE, s)

# ========= Streak & recent totals =========
def get_streak():
    data = read_json(DATA_FILE, {"water": [], "sleep": [], "mood": []})
    dates_logged = set()
    for cat in ("water", "sleep", "mood"):
        for e in data.get(cat, []):
            ts = e.get("timestamp")
            if not ts:
                continue
            try:
                dt = datetime.fromisoformat(ts)
                dates_logged.add(dt.date())
            except Exception:
                pass
    today = datetime.now().date()
    streak = 0
    while (today - timedelta(days=streak)) in dates_logged:
        streak += 1
    return streak

def recent_totals_for_prompt(days=3):
    data = read_json(DATA_FILE, {"water": [], "sleep": [], "mood": []})
    cutoff = datetime.now() - timedelta(days=days)

    def sum_since(entries, key):
        total = 0.0
        for e in entries:
            ts = e.get("timestamp")
            val = e.get(key)
            if ts is None or val is None:
                continue
            try:
                dt = datetime.fromisoformat(ts)
            except Exception:
                continue
            if dt >= cutoff:
                try:
                    total += float(val)
                except Exception:
                    pass
        return round(total, 2)

    return {
        "water_last3d": sum_since(data.get("water", []), "amount"),
        "sleep_last3d": sum_since(data.get("sleep", []), "hours"),
    }

# ========= Personalised targets & profile =========
def read_profile():
    return read_json(PROFILE_FILE, {
        "age": None,
        "gender": None,
        "height_cm": None,
        "weight_kg": None,
        "activity": "moderate"
    })

def write_profile(p):
    write_json(PROFILE_FILE, p)

def recommended_sleep_hours(age):
    if age is None:
        return (7.0, 9.0, 8.0)
    try:
        age = int(age)
    except Exception:
        return (7.0, 9.0, 8.0)
    if age <= 5:
        return (10.0, 13.0, 11.5)
    if age <= 13:
        return (9.0, 11.0, 10.0)
    if age <= 17:
        return (8.0, 10.0, 9.0)
    if age <= 64:
        return (7.0, 9.0, 8.0)
    return (7.0, 8.0, 7.5)

def compute_targets(profile):
    weight = profile.get("weight_kg")
    height_cm = profile.get("height_cm")
    activity = (profile.get("activity") or "moderate").lower()

    if isinstance(weight, (int, float)) and weight and weight > 0:
        base_l = weight * 0.035
    else:
        base_l = 1.8

    bonus = 0.0
    if activity == "moderate":
        bonus = 0.3
    elif activity == "high":
        bonus = 0.6
    water_target = round(base_l + bonus, 2)

    _, _, tgt_s = recommended_sleep_hours(profile.get("age"))
    sleep_target = float(tgt_s)

    bmi = None
    if isinstance(weight, (int, float)) and isinstance(height_cm, (int, float)) and weight and height_cm:
        h_m = height_cm / 100.0
        if h_m > 0:
            bmi = round(weight / (h_m * h_m), 1)

    return water_target, sleep_target, bmi

# ========= Ollama helpers (robust; tuned for local server) =========
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")

def query_ollama(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.55,
            "top_p": 0.9,
            "repeat_penalty": 1.12,
            "num_ctx": 8192
        }
    }
    timeouts = [(5, 120), (5, 300)]
    last_err = None

    for connect_t, read_t in timeouts:
        try:
            print(f"=== query_ollama: attempt connect={connect_t}s read={read_t}s model={OLLAMA_MODEL}")
            r = requests.post(OLLAMA_URL, json=payload, timeout=(connect_t, read_t))
            try:
                data = r.json()
            except Exception as je:
                last_err = f"non-JSON from Ollama ({je})"
                print(">>>", last_err)
                continue

            if r.status_code != 200:
                last_err = f"HTTP {r.status_code}: {data}"
                print(">>>", last_err)
                continue

            if data.get("error"):
                last_err = f"ollama error: {data['error']}"
                print(">>>", last_err)
                continue

            return (data.get("response") or "").strip() or "Bot: (no text generated)."

        except requests.exceptions.ReadTimeout:
            last_err = f"read timeout after {read_t}s"
            print(">>>", last_err)
            continue
        except requests.exceptions.ConnectionError as ce:
            last_err = f"connection error: {ce}"
            print(">>>", last_err)
            break
        except Exception as e:
            last_err = f"unexpected: {e}"
            print(">>>", last_err)
            break

    return (
        "Bot: I couldn’t reach the local model engine just now. "
        "Please ensure `ollama serve` is running and the model name matches your install "
        f"(model={OLLAMA_MODEL}). Detail: {last_err}"
    )

# ========= Simple auth for destructive routes (optional) =========
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN")

def maybe_require_admin(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if ADMIN_TOKEN:
            if request.headers.get("X-Admin-Token") != ADMIN_TOKEN:
                return jsonify({"error": "unauthorized"}), 401
        return fn(*args, **kwargs)
    return wrapper

# ========= Helpers =========
_GREETING_RE = re.compile(r"^\s*(hi|hello|hey|yo|hiya|hola|sup|heyy+|hai)\b[!.]*\s*$", re.I)

def is_greeting(txt: str) -> bool:
    if not txt:
        return False
    t = txt.strip()
    if t in {"👋", "🙂", "😊", "😄"}:
        return True
    return bool(_GREETING_RE.match(t))

def is_short_neutral(txt: str) -> bool:
    t = (txt or "").strip().lower()
    if not t or t in YESLIKE or is_greeting(t):
        return False
    intentish = (
        "suggest", "suggestion", "tip", "plan", "exercise", "breath", "breathing",
        "work", "faster", "study", "focus", "task", "todo", "overwhelm",
        "anx", "anxiety", "stress", "sleep", "hour", "water", "liter",
        "migraine", "headache", "hydrate"
    )
    if any(k in t for k in intentish):
        return False
    tokens = re.findall(r"[a-zA-Z0-9]+", t)
    return len(tokens) <= 2

def detect_intent(text: str) -> str:
    t = (text or "").strip().lower()

    if any(x in t for x in [
        "weekly report", "show report", "summary", "show summary",
        "how am i doing", "my habits", "weekly narrative"
    ]):
        return "report"

    if any(x in t for x in [
        "i drank", "drank", "water today", "litre", "liter", "ml", "cups",
        "i slept", "slept", "hours", "hrs", "i feel", "i felt", "my mood"
    ]):
        return "log"

    if any(x in t for x in [
        "headache", "migraine", "stomach ache", "stomachache", "pain", "nausea"
    ]):
        return "symptom_support"

    if any(x in t for x in [
        "sleep", "bedtime", "insomnia", "tired", "stress", "anxiety", "anxious",
        "hydration", "hydrate", "water", "habit", "mood", "overwhelmed"
    ]):
        return "wellness"

    if is_greeting(t) or is_short_neutral(t):
        return "chat"

    return "chat"

def crisis_detected(text: str) -> bool:
    t = (text or "").lower()
    patterns = [
        "kill myself", "end my life", "suicide", "self harm",
        "hurt myself", "want to die", "harm others", "kill someone"
    ]
    return any(p in t for p in patterns)

def natural_log_reply(user_input: str, data: dict, targets: dict) -> str:
    messages = []

    if data.get("water_logged") is not None:
        water = data["water_logged"]
        target = targets["water_target"]
        messages.append(
            f"I’ve logged {water:.2f}L of water for today."
        )
        if target:
            pct = int((water / target) * 100) if target > 0 else 0
            messages.append(f"That’s about {pct}% of your daily water goal.")

    if data.get("sleep_logged") is not None:
        sleep = data["sleep_logged"]
        target = targets["sleep_target"]
        messages.append(f"I’ve logged {sleep:.1f} hours of sleep.")
        if target:
            if sleep >= target:
                messages.append("Nice — that’s around or above your target.")
            else:
                messages.append(f"That’s a bit below your target of {target:.1f} hours.")

    if data.get("mood_logged"):
        messages.append(f"I’ve logged your mood as {data['mood_logged']}.")

    if not messages:
        return "Bot: Got it — I’ve noted that. Tell me if you’d like a quick tip, a summary, or help with your habits."

    return "Bot: " + " ".join(messages)

def extract_logged_values(user_input: str):
    t = normalize_quotes(user_input or "")
    result = {
        "water_logged": None,
        "sleep_logged": None,
        "mood_logged": None,
    }

    water_match = re.search(
        r"(\d+(?:\.\d+)?)\s*((?:liters?|litres?|ltr?s?|l|ml|milliliters?|millilitres?|cups?))\b",
        t,
        re.IGNORECASE,
    )
    if water_match:
        amount = float(water_match.group(1))
        unit = water_match.group(2).lower()
        if unit.startswith("ml") or unit.startswith("millil"):
            amount /= 1000.0
        elif unit.startswith("cup"):
            amount = amount * 240.0 / 1000.0
        result["water_logged"] = amount

    sleep_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)\b", t, re.IGNORECASE)
    if sleep_match and re.search(r"\bsleep|slept\b", t.lower()):
        result["sleep_logged"] = float(sleep_match.group(1))

    mood_keywords = [
        "happy", "joyful", "excited", "sad", "depressed", "anxious", "angry",
        "tired", "calm", "lonely", "stressed", "nervous", "good", "fine"
    ]
    lower_t = t.lower()
    for mood in mood_keywords:
        if re.search(rf"\b{re.escape(mood)}\b", lower_t):
            result["mood_logged"] = mood
            break

    return result

def build_chat_prompt(intent, user_input, selected_date, streak, totals, short_context, mem_snippets):
    context_block = ""
    if short_context:
        context_block += f"\n\nRecent dialogue:\n{short_context}"
    if mem_snippets:
        context_block += f"\n\nRelevant memories:\n{mem_snippets}"

    system = """
You are SheeBot, a friendly, natural, conversational AI wellness assistant.

Your default style is warm, clear, human-like, and helpful.
You should sound like a normal chatbot in everyday conversation.

Behavior by intent:
- chat: respond naturally and conversationally like a normal helpful chatbot.
- wellness: give practical, natural advice about sleep, hydration, mood, stress, or habits.
- log: acknowledge the logged information naturally and briefly.
- report: summarise patterns clearly and concisely.
- symptom_support: do NOT diagnose and do NOT suggest remedies or medication. Give gentle self-care suggestions and suggest professional help if symptoms are severe or persistent.

Rules:
- Do not force every response into coaching language.
- Do not always give “tiny next steps”.
- Do not turn simple questions into jokes or unrelated content.
- Do not overuse emotional language.
- Keep the answer relevant to the user’s actual question.
- Be concise, but natural.
- Avoid medical claims.
"""

    if intent == "chat":
        instruction = "Respond naturally like a friendly chatbot."
    elif intent == "wellness":
        instruction = "Respond with practical, conversational wellness advice. Keep it useful and natural."
    elif intent == "log":
        instruction = "Acknowledge the logged information naturally and briefly."
    elif intent == "report":
        instruction = "Respond with a clear and structured summary."
    elif intent == "symptom_support":
        instruction = (
            "Respond safely. Do not diagnose. Suggest simple self-care like rest, hydration, "
            "and seeking professional help if needed."
        )
    else:
        instruction = "Respond naturally and helpfully."

    return f"""{system}

Detected intent: {intent}
Streak: {streak} days
Selected date (if any): {selected_date or 'today'}
Recent totals: water_last3d={totals['water_last3d']}L, sleep_last3d={totals['sleep_last3d']}h

User: {user_input}{context_block}

{instruction}
Respond as 'Bot:'.
Bot:"""

# ========= Flask Routes =========
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health():
    try:
        ping = query_ollama("Say 'pong' briefly.")
        ok = bool(ping)
        return jsonify({"ok": ok, "engine_reply": ping[:120]})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# ---- Profile API ----
@app.route("/profile", methods=["GET", "POST"])
def profile_route():
    try:
        if request.method == "GET":
            return jsonify(read_profile())

        payload = request.get_json(force=True) or {}
        p = read_profile()

        def to_float(x):
            try:
                if x is None or x == "":
                    return None
                return float(x)
            except Exception:
                return None

        def to_int(x):
            try:
                if x is None or x == "":
                    return None
                return int(x)
            except Exception:
                return None

        age = to_int(payload.get("age"))
        gender = payload.get("gender") or None
        height_cm = to_float(payload.get("height_cm"))
        weight_kg = to_float(payload.get("weight_kg"))
        activity = (payload.get("activity") or "moderate").lower()
        if activity not in ("low", "moderate", "high"):
            activity = "moderate"

        p.update({
            "age": age,
            "gender": gender,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "activity": activity
        })
        write_profile(p)
        return jsonify({"status": "ok", "profile": p})

    except Exception as e:
        print("❌ /profile error:", e)
        traceback.print_exc()
        return jsonify({"error": "Failed to save profile"}), 500

# ---- Lightweight targets endpoint (charts use this) ----
@app.route("/targets", methods=["GET"])
def targets():
    try:
        profile = read_profile()
        water_target, sleep_target, bmi = compute_targets(profile)
        result = {
            "water_target_L": water_target,
            "sleep_target_h": sleep_target,
            "bmi": bmi,
            "profile": profile
        }
        return jsonify(result)
    except Exception as e:
        print("❌ /targets error:", e)
        traceback.print_exc()
        return jsonify({"error": "Internal error in /targets", "detail": str(e)}), 500

# ---- Chat API ----
@app.route("/chat", methods=["POST"])
def chat():
    try:
        payload = request.get_json(force=True) or {}
        user_input = (payload.get("message", "") or "").strip()
        selected_date = payload.get("selected_date")

        print("=== /chat RECEIVED payload ===")
        print(payload)
        print("=== /chat user_input ===", repr(user_input), "selected_date:", repr(selected_date))

        if not user_input:
            return jsonify({"error": "message is required"}), 400

        if crisis_detected(user_input):
            reply = (
                "Bot: I’m really sorry you’re feeling this way. You deserve immediate support. "
                "Please contact emergency services or a crisis helpline in your area right now, "
                "and reach out to someone you trust. Are you safe right now?"
            )
            save_log(user_input, reply)
            return jsonify({"response": reply})

        intent = detect_intent(user_input)

        # Handle greetings / tiny casual messages locally
        if intent == "chat" and (is_greeting(user_input) or is_short_neutral(user_input)):
            reply = (
                "Bot: 👋 Hey! I’m here. You can chat with me normally, log things like sleep or water, "
                "or ask for wellness advice. What would you like help with?"
            )
            save_log(user_input, reply)
            return jsonify({"response": reply})

        # update structured logs (best-effort)
        try:
            update_user_data(user_input, manual_date=selected_date)
        except Exception as log_err:
            append_jsonl(LOG_FILE, {
                "timestamp": now_iso(),
                "user_input": user_input,
                "bot_response": "",
                "note": f"update_user_data failed: {repr(log_err)}"
            })

        lowered = user_input.lower()
        state = get_state()

        if lowered in YESLIKE and state.get("last_topic") == "symptom_support":
            user_input = "Please continue helping me with gentle symptom support."
            intent = "symptom_support"
        elif lowered in YESLIKE and state.get("last_topic"):
            user_input = f"Please continue helping me with {state.get('last_topic')}."
            intent = "wellness"

        profile = read_profile()
        water_target, sleep_target, _ = compute_targets(profile)

        # Natural direct reply for log messages
        if intent == "log":
            extracted = extract_logged_values(user_input)
            reply = natural_log_reply(
                user_input,
                extracted,
                {"water_target": water_target, "sleep_target": sleep_target}
            )
            save_log(user_input, reply)

            ui_low = user_input.lower()
            if any(x in ui_low for x in ["sleep", "slept"]):
                set_state({"last_topic": "sleep"})
            elif "water" in ui_low or "litre" in ui_low or "liter" in ui_low:
                set_state({"last_topic": "hydration"})
            elif "feel" in ui_low or "mood" in ui_low:
                set_state({"last_topic": "mood"})

            return jsonify({"response": reply})

        # Report shortcut
        if intent == "report":
            report = build_weekly_report(days=7)
            reply = "Bot: " + report["summary"] + "\n\nSuggestions:\n- " + "\n- ".join(report["suggestions"][:3])
            save_log(user_input, reply)
            set_state({"last_topic": "report"})
            return jsonify({"response": reply})

        # Build LLM prompt
        short_context = recent_dialogue(n_pairs=6)
        retrieved = search_memory(user_input, top_k=5, min_sim=0.25)
        mem_snippets = "\n".join(f"- {m['text']} ({m.get('timestamp', '')})" for m in retrieved)
        totals = recent_totals_for_prompt()
        streak = get_streak()

        full_prompt = build_chat_prompt(
            intent=intent,
            user_input=user_input,
            selected_date=selected_date,
            streak=streak,
            totals=totals,
            short_context=short_context,
            mem_snippets=mem_snippets
        )

        bot_response = query_ollama(full_prompt)

        print("=== /chat bot_response (trimmed) ===")
        print(repr(bot_response)[:1000])

        # Ensure Bot: prefix
        if not bot_response.strip().startswith("Bot:"):
            bot_response = "Bot: " + bot_response.strip()

        save_log(user_input, bot_response)

        ui_low = user_input.lower()
        if any(x in ui_low for x in ["headache", "migraine", "stomach ache", "pain", "nausea"]):
            set_state({"last_topic": "symptom_support"})
        elif any(x in ui_low for x in ["sleep", "slept", "bedtime", "insomnia"]):
            set_state({"last_topic": "sleep"})
        elif any(x in ui_low for x in ["water", "hydrate", "hydration", "litre", "liter"]):
            set_state({"last_topic": "hydration"})
        elif any(x in ui_low for x in ["stress", "anxiety", "mood", "happy", "sad"]):
            set_state({"last_topic": "mood"})
        else:
            set_state({"last_topic": "chat"})

        return jsonify({"response": bot_response})

    except Exception as e:
        print("❌ /chat error:", e)
        traceback.print_exc()
        return jsonify({
            "response": "Bot: I hit an unexpected error, but I’m here. Try asking again in a simpler way, and I’ll help."
        }), 200

# ---- Graph data for charts ----
@app.route("/graph-data", methods=["GET"])
def graph_data():
    try:
        data = read_json(DATA_FILE, {"water": [], "sleep": [], "mood": []})
        days_param = (request.args.get("days", "7") or "7").lower()

        if days_param != "all":
            try:
                days_int = max(1, int(days_param))
            except ValueError:
                return jsonify({"error": "Invalid 'days' parameter"}), 400
            cutoff = datetime.now() - timedelta(days=days_int)
        else:
            cutoff = datetime.min

        def group_daily_sum(entries, key):
            grouped = {}
            for e in entries:
                ts = e.get("timestamp")
                val = e.get(key)
                if ts is None or val is None:
                    continue
                try:
                    dt = datetime.fromisoformat(ts)
                    if dt >= cutoff:
                        day = dt.strftime("%Y-%m-%d")
                        grouped[day] = grouped.get(day, 0) + float(val)
                except Exception as err:
                    print(f"⚠️ Skipping bad ts: {ts} -> {err}")
            return [[v, d] for d, v in sorted(grouped.items())]

        mood_by_day = {}
        for entry in data.get("mood", []):
            ts = entry.get("timestamp")
            emoji = entry.get("emoji")
            if not ts or not emoji:
                continue
            try:
                dt = datetime.fromisoformat(ts)
                dstr = dt.strftime("%Y-%m-%d")
                if dt >= cutoff:
                    if dstr not in mood_by_day:
                        mood_by_day[dstr] = {}
                    mood_by_day[dstr][emoji] = mood_by_day[dstr].get(emoji, 0) + 1
            except Exception as err:
                print(f"⚠️ Skipping mood entry {entry}: {err}")

        return jsonify({
            "water": group_daily_sum(data.get("water", []), "amount"),
            "sleep": group_daily_sum(data.get("sleep", []), "hours"),
            "mood": mood_by_day
        })
    except Exception as e:
        print("❌ Error in /graph-data:", e)
        return jsonify({"error": "Internal error"}), 500

# ========= Weekly summary helpers & route =========
def _last_n_days_dates(n=7):
    end = datetime.now().date()
    return [(end - timedelta(days=i)).isoformat() for i in range(n)][::-1]

def _zero_day_map(n=7):
    return {d: 0.0 for d in _last_n_days_dates(n)}

def _classify_mood(emoji_or_text: str) -> str:
    positive = {"😀", "😁", "🤩", "😌", "🙂", "happy", "joyful", "excited", "calm", "good", "fine"}
    negative = {"😔", "😞", "😟", "😰", "😣", "😢", "😠", "sad", "depressed", "anxious", "stressed", "nervous", "angry"}
    if emoji_or_text in positive:
        return "positive"
    if emoji_or_text in negative:
        return "negative"
    return "neutral"

def build_weekly_report(days=7):
    profile = read_profile()
    water_target, sleep_target, bmi = compute_targets(profile)

    data = read_json(DATA_FILE, {"water": [], "sleep": [], "mood": []})
    day_keys = _last_n_days_dates(days)
    water = _zero_day_map(days)
    sleep = _zero_day_map(days)
    mood_counts_by_day = {d: {} for d in day_keys}

    for e in data.get("water", []):
        ts = e.get("timestamp")
        amt = e.get("amount")
        if not ts or amt is None:
            continue
        try:
            d = datetime.fromisoformat(ts).date().isoformat()
            if d in water:
                water[d] += float(amt)
        except Exception:
            pass

    for e in data.get("sleep", []):
        ts = e.get("timestamp")
        hrs = e.get("hours")
        if not ts or hrs is None:
            continue
        try:
            d = datetime.fromisoformat(ts).date().isoformat()
            if d in sleep:
                sleep[d] += float(hrs)
        except Exception:
            pass

    for e in data.get("mood", []):
        ts = e.get("timestamp")
        emo = e.get("emoji") or e.get("text")
        if not ts or not emo:
            continue
        try:
            d = datetime.fromisoformat(ts).date().isoformat()
            if d in mood_counts_by_day:
                bucket = _classify_mood(emo)
                mood_counts_by_day[d][bucket] = mood_counts_by_day[d].get(bucket, 0) + 1
        except Exception:
            pass

    water_values = [water[d] for d in day_keys]
    sleep_values = [sleep[d] for d in day_keys]

    water_total = round(sum(water_values), 2)
    water_avg = round(water_total / days, 2)
    sleep_avg = round(sum(sleep_values) / days, 2)

    days_meet_water = sum(1 for d in day_keys if water[d] >= water_target)
    days_meet_sleep = sum(1 for d in day_keys if sleep[d] >= sleep_target)

    mood_totals = {"positive": 0, "neutral": 0, "negative": 0}
    for d in day_keys:
        for k, v in mood_counts_by_day[d].items():
            mood_totals[k] = mood_totals.get(k, 0) + v
    mood_total_all = sum(mood_totals.values()) or 1
    mood_pos_pct = round(100 * mood_totals.get("positive", 0) / mood_total_all)

    days_with_any_log = sum(1 for d in day_keys if (water[d] > 0 or sleep[d] > 0 or mood_counts_by_day[d]))
    consistency = f"{days_with_any_log}/{days}"

    suggestions = []
    if water_avg < water_target:
        gap = max(0.0, round(water_target - water_avg, 2))
        glasses = max(1, round((gap * 1000) / 250))
        suggestions.append(
            f"Hydration: average {water_avg}L/day, below your {water_target}L goal. "
            f"Try adding about {gap}L/day (~{glasses} glasses of 250ml)."
        )
    else:
        suggestions.append(
            f"Hydration: on target — {water_avg}L/day versus your {water_target}L goal."
        )

    if sleep_avg < sleep_target:
        suggestions.append(
            f"Sleep: average {sleep_avg}h versus your {sleep_target}h target. "
            f"Try a 15–30 minute earlier wind-down and less screen time before bed."
        )
    else:
        suggestions.append(
            f"Sleep: good consistency — {sleep_avg}h average, which is around your target."
        )

    if mood_pos_pct < 50:
        suggestions.append("Mood: fewer positive check-ins this week. A short walk, journaling, or reaching out to someone may help.")
    else:
        suggestions.append("Mood: trending positive — keep the small habits that are helping.")

    if days_with_any_log < 5:
        suggestions.append(f"Consistency: you logged data on {consistency} days. More regular logs will improve your insights.")

    if bmi is not None:
        if bmi >= 25:
            suggestions.append(f"BMI is around {bmi}. Gentle daily movement may help energy and sleep.")
        elif bmi < 18.5:
            suggestions.append(f"BMI is around {bmi}. Regular meals and snacks can support energy, mood, and sleep.")

    lines = []
    lines.append("📅 Weekly Report (last 7 days, personalised)")
    lines.append(f"💧 Water: {water_total}L total • {water_avg}L/day average • Goal {water_target}L/day • {days_meet_water}/7 days met")
    lines.append(f"🌙 Sleep: {sleep_avg}h/day average • Goal {sleep_target}h/day • {days_meet_sleep}/7 days met")
    lines.append(f"😊 Mood: {mood_pos_pct}% positive entries • Consistency: {consistency}")

    return {
        "summary": "\n".join(lines),
        "stats": {
            "days": day_keys,
            "water_by_day": water,
            "sleep_by_day": sleep,
            "mood_by_day": mood_counts_by_day,
            "water_avg": water_avg,
            "sleep_avg": sleep_avg,
            "mood_positive_pct": mood_pos_pct,
            "consistency_days": days_with_any_log,
            "targets": {
                "water_target_L": water_target,
                "sleep_target_h": sleep_target,
                "bmi": bmi
            },
            "profile": profile,
        },
        "suggestions": suggestions
    }

@app.route("/weekly-report", methods=["GET"])
def weekly_report():
    try:
        report = build_weekly_report(days=7)
        return jsonify(report)
    except Exception as e:
        print("❌ Error in /weekly-report:", e)
        traceback.print_exc()
        return jsonify({"error": "Internal error"}), 500

@app.route("/weekly-narrative", methods=["GET"])
def weekly_narrative():
    try:
        report = build_weekly_report(days=7)
        prof = report["stats"]["profile"]
        t = report["stats"]["targets"]

        prompt = f"""
You are SheeBot. Write a warm, natural weekly wellness note in 6–9 sentences.
Use the data below. Sound conversational, supportive, and specific.
Do not sound robotic. Avoid medical claims.
End with 2 short bullet-point next steps.

PROFILE:
{json.dumps(prof, ensure_ascii=False)}

TARGETS:
{json.dumps(t, ensure_ascii=False)}

SUMMARY:
{report['summary']}

SUGGESTIONS:
- """ + "\n- ".join(report["suggestions"]) + """

Write the note now:
"""
        text = query_ollama(prompt)
        if not text.strip().startswith("Bot:"):
            text = "Bot: " + text.strip()
        return jsonify({"narrative": text})
    except Exception as e:
        print("❌ Error in /weekly-narrative:", e)
        traceback.print_exc()
        return jsonify({"error": "Internal error"}), 500

@app.route("/memories", methods=["GET"])
def list_memories():
    items = read_jsonl(MEMORY_FILE)
    items = items[-50:]
    text = "\n".join(f"- {m.get('text', '')} ({m.get('timestamp', '')})" for m in items) or "(none)"
    return jsonify({"response": "🧠 Memories (latest 50):\n" + text})

@app.route("/forget-all", methods=["POST"])
@maybe_require_admin
def forget_all():
    open(MEMORY_FILE, "w").close()
    return jsonify({"response": "🧼 Wiped all long-term memories."})

@app.route("/reset", methods=["POST"])
@maybe_require_admin
def reset():
    write_json(DATA_FILE, {"water": [], "sleep": [], "mood": []})
    open(LOG_FILE, "w").close()
    write_json(STATE_FILE, {})
    return jsonify({"status": "reset"})

if __name__ == "__main__":
    host = os.environ.get("APP_HOST", "127.0.0.1")
    port = int(os.environ.get("APP_PORT", "5000"))
    app.run(host=host, port=port, debug=True)
