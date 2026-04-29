import json
import os
from flask import Flask, request, jsonify, render_template
from groq import Groq

app = Flask(__name__)

# The key is now safely pulled from the server environment
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

AVAILABLE_MODELS = [
    {"id": "llama-3.3-70b-versatile", "label": "Llama 3.3 70B", "badge": "BEST"},
    {"id": "deepseek-r1-distill-llama-70b", "label": "DeepSeek R1 70B", "badge": "THINK"},
    {"id": "llama-3.1-8b-instant", "label": "Llama 3.1 8B Instant", "badge": "FAST"},
]

@app.route('/')
def index():
    return render_template('index.html', models=AVAILABLE_MODELS)

@app.route('/chat', methods=['POST'])
def chat():
    if not GROQ_API_KEY:
        return jsonify({"error": "API Key not configured on server"}), 500
        
    body = request.get_json()
    history = body.get('history', [])
    model = body.get('model', 'llama-3.3-70b-versatile')
    
    client = Groq(api_key=GROQ_API_KEY)
    msgs = []
    for m in history:
        role = "assistant" if m["role"] in ("model", "assistant") else "user"
        msgs.append({"role": role, "content": m.get("content", "")})
    
    try:
        completion = client.chat.completions.create(model=model, messages=msgs)
        return jsonify({"reply": completion.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
