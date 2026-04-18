from flask import Flask, request, jsonify
import joblib
import numpy as np
import tldextract
import re
import pickle
import os
import requests

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# 🔥 HuggingFace LINKS (replace if needed)
FILES = {
    "phishing_model.joblib": "https://huggingface.co/sourav0777/phishing-model/resolve/main/phishing_model.joblib",
    "scaler.joblib": "https://huggingface.co/sourav0777/phishing-model/resolve/main/scaler.joblib",
    "tfidf.pkl": "https://huggingface.co/sourav0777/phishing-model/resolve/main/tfidf.pkl"
}

IP_REGEX = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")

# 🔥 GLOBAL (lazy load)
model = None
scaler = None
tfidf = None

# ---------------- DOWNLOAD ----------------
def download_models():
    for name, url in FILES.items():
        path = os.path.join(MODEL_DIR, name)

        if not os.path.exists(path):
            print(f"⬇ Downloading {name}...")
            r = requests.get(url, timeout=30)
            r.raise_for_status()

            with open(path, "wb") as f:
                f.write(r.content)

            print(f"✅ {name} downloaded")

# ---------------- LOAD ----------------
def load_models():
    global model, scaler, tfidf

    if model is None:
        print("🔄 Loading models...")

        download_models()

        model = joblib.load(os.path.join(MODEL_DIR, "phishing_model.joblib"))
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
        tfidf = pickle.load(open(os.path.join(MODEL_DIR, "tfidf.pkl"), "rb"))

        print("✅ Models loaded")

# ---------------- FEATURE EXTRACTION ----------------
def extract_features(url):
    ext = tldextract.extract(url)

    lexical = [
        len(url),
        url.count("."),
        url.count("/"),
        url.count("-"),
        1 if IP_REGEX.search(url) else 0,
        len(ext.domain),
        len(ext.suffix),
        ext.subdomain.count(".") + 1 if ext.subdomain else 0,
        1 if "@" in url else 0,
        1 if "https" in ext.domain else 0,
    ]

    tfidf_features = tfidf.transform([url]).toarray()[0]

    final = np.concatenate([lexical, tfidf_features])

    return final.reshape(1, -1)

# ---------------- API ----------------
@app.route("/scan_url", methods=["POST"])
def scan_url():
    load_models()  # 🔥 IMPORTANT

    data = request.get_json()

    if not data or "url" not in data:
        return jsonify({"error": "URL required"}), 400

    url = data["url"].strip()

    if not url.startswith("http"):
        url = "http://" + url

    try:
        features = extract_features(url)
        features = scaler.transform(features)

        pred = model.predict(features)[0]
        confidence = float(np.max(model.predict_proba(features)) * 100)

        return jsonify({
            "prediction": int(pred),
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- ROOT (health check) ----------------
@app.route("/")
def home():
    return "API is running 🚀"

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
