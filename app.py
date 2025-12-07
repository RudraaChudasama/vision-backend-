from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import json
import os
import requests
from datetime import datetime

# ==============================================
# 1️⃣ GOOGLE DRIVE DIRECT DOWNLOAD URL
# ==============================================
MODEL_URL = "https://drive.google.com/uc?export=download&id=1n5gVMkh9471E_Y_4SmWzbGZOu26OTvy8"
MODEL_PATH = "best.pt"

app = Flask(__name__)
CORS(app)

# ==============================================
# 2️⃣ DOWNLOAD MODEL IF MISSING
# ==============================================
def download_model():
    if os.path.exists(MODEL_PATH):
        print("✅ best.pt already exists.")
        return
    
    print("📥 Downloading model from Google Drive…")
    response = requests.get(MODEL_URL)

    if response.status_code == 200:
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("✅ Model downloaded successfully!")
    else:
        print("❌ Failed to download model:", response.text)


# Download the model before loading YOLO
download_model()

# ==============================================
# 3️⃣ LOAD YOLO MODEL
# ==============================================
print("⚡ Loading YOLO model…")
model = YOLO(MODEL_PATH)
print("✅ YOLO Model Loaded Successfully!")

# ==============================================
# DATABASE FILE
# ==============================================
DATABASE_FILE = 'database.json'

# Disease mapping
DISEASE_CLASSES = {
    0: 'Normal',
    1: 'Cataract',
    2: 'Conjunctivitis',
    3: 'Eyelid',
    4: 'Uveitis'
}

# ==============================================
# DATABASE UTILITIES
# ==============================================
def init_database():
    if not os.path.exists(DATABASE_FILE):
        with open(DATABASE_FILE, 'w') as f:
            json.dump([], f)

def load_records():
    try:
        with open(DATABASE_FILE, 'r') as f:
            return json.load(f)
    except:
        return []

def save_records(records):
    with open(DATABASE_FILE, 'w') as f:
        json.dump(records, f, indent=2)


# ==============================================
# IMAGE DECODE FUNCTION
# ==============================================
def base64_to_image(base64_string):
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


# ==============================================
# API ROUTES
# ==============================================

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'message': 'Backend running'}), 200


@app.route('/detect', methods=['POST'])
def detect_disease():
    try:
        data = request.get_json()

        if 'image' not in data:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        img = base64_to_image(data['image'])

        results = model(img)

        if len(results[0].boxes) > 0:
            boxes = results[0].boxes
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()

            best_idx = np.argmax(confs)
            best_class = int(classes[best_idx])
            best_conf = float(confs[best_idx] * 100)

            disease = DISEASE_CLASSES.get(best_class, "Unknown")

            return jsonify({
                "success": True,
                "disease": disease,
                "confidence": round(best_conf, 2)
            })

        else:
            return jsonify({
                "success": True,
                "disease": "Normal",
                "confidence": 95.0
            })

    except Exception as e:
        print("Detection error:", str(e))
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/save', methods=['POST'])
def save_record():
    try:
        record = request.get_json()

        required = ['id', 'image_name', 'disease_detected', 'confidence_score', 'timestamp']
        for field in required:
            if field not in record:
                return jsonify({'success': False, 'error': f'Missing field: {field}'}), 400

        records = load_records()
        records.append(record)
        save_records(records)

        return jsonify({"success": True, "message": "Record saved"})

    except Exception as e:
        print("Save error:", str(e))
        return jsonify({"success": False, "error": str(e)})


@app.route('/records', methods=['GET'])
def get_all_records():
    try:
        return jsonify({"success": True, "records": load_records()})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/delete/<record_id>', methods=['DELETE'])
def delete_record(record_id):
    try:
        records = load_records()
        new_records = [r for r in records if r['id'] != record_id]

        if len(new_records) == len(records):
            return jsonify({"success": False, "error": "Record not found"}), 404

        save_records(new_records)
        return jsonify({"success": True, "message": "Record deleted"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# ==============================================
# RUN SERVER
# ==============================================
if __name__ == '__main__':
    print("🚀 Backend starting on http://localhost:5000 …")
    init_database()
    app.run(host="0.0.0.0", port=5000)
