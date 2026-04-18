from flask import Flask, request, jsonify
import joblib
import numpy as np
import tldextract
import re
import pickle
import os

app = Flask(__name__)

# 🔥 FIXED PATH (IMPORTANT)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "model", "phishing_model.joblib"))
scaler = joblib.load(os.path.join(BASE_DIR, "model", "scaler.joblib"))
tfidf = pickle.load(open(os.path.join(BASE_DIR, "model", "tfidf.pkl"), "rb"))

IP_REGEX = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")

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

# ---------------- API ROUTE ----------------
@app.route("/scan_url", methods=["POST"])
def scan_url():
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

        # optional confidence
        confidence = float(np.max(model.predict_proba(features)) * 100)

        return jsonify({
            "prediction": int(pred),
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)