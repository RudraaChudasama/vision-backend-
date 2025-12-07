from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import json
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Allow frontend to connect

# Load your YOLO model
MODEL_PATH = 'best.pt'  # Make sure your model file is here
model = YOLO(MODEL_PATH)

# Database file
DATABASE_FILE = 'database.json'

# Disease name mapping (adjust based on your model)
DISEASE_CLASSES = {
    0: 'Normal',
    1: 'Cataract',
    2: 'Conjunctivitis',
    3: 'Eyelid',
    4: 'Uveitis'
}


# Initialize database file if it doesn't exist
def init_database():
    if not os.path.exists(DATABASE_FILE):
        with open(DATABASE_FILE, 'w') as f:
            json.dump([], f)


# Load records from database
def load_records():
    try:
        with open(DATABASE_FILE, 'r') as f:
            return json.load(f)
    except:
        return []


# Save records to database
def save_records(records):
    with open(DATABASE_FILE, 'w') as f:
        json.dump(records, f, indent=2)


# Convert base64 image to OpenCV format
def base64_to_image(base64_string):
    # Remove header if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    # Decode base64
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


# ====== API ENDPOINTS ======

@app.route('/health', methods=['GET'])
def health_check():
    """Check if server is running"""
    return jsonify({'status': 'ok', 'message': 'Backend is running'}), 200


@app.route('/detect', methods=['POST'])
def detect_disease():
    """Detect eye disease from image"""
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        # Convert base64 image to OpenCV format
        img = base64_to_image(data['image'])
        
        # Run YOLO detection
        results = model(img)
        
        # Get the best detection
        if len(results[0].boxes) > 0:
            # Get class with highest confidence
            boxes = results[0].boxes
            confidences = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()
            
            best_idx = np.argmax(confidences)
            best_class = int(classes[best_idx])
            best_confidence = float(confidences[best_idx]) * 100
            
            disease_name = DISEASE_CLASSES.get(best_class, 'Unknown')
            
            return jsonify({
                'success': True,
                'disease': disease_name,
                'confidence': round(best_confidence, 2)
            }), 200
        else:
            # No detection - assume normal
            return jsonify({
                'success': True,
                'disease': 'Normal',
                'confidence': 95.0
            }), 200
            
    except Exception as e:
        print(f"Detection error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/save', methods=['POST'])
def save_record():
    """Save detection record to database"""
    try:
        record = request.get_json()
        
        # Validate required fields
        required_fields = ['id', 'image_name', 'disease_detected', 'confidence_score', 'timestamp']
        for field in required_fields:
            if field not in record:
                return jsonify({'success': False, 'error': f'Missing field: {field}'}), 400
        
        # Load existing records
        records = load_records()
        
        # Add new record
        records.append(record)
        
        # Save to database
        save_records(records)
        
        return jsonify({'success': True, 'message': 'Record saved'}), 200
        
    except Exception as e:
        print(f"Save error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/records', methods=['GET'])
def get_all_records():
    """Get all detection records"""
    try:
        records = load_records()
        return jsonify({'success': True, 'records': records}), 200
    except Exception as e:
        print(f"Get records error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/delete/<record_id>', methods=['DELETE'])
def delete_record(record_id):
    """Delete a specific record"""
    try:
        records = load_records()
        
        # Filter out the record to delete
        updated_records = [r for r in records if r['id'] != record_id]
        
        if len(updated_records) == len(records):
            return jsonify({'success': False, 'error': 'Record not found'}), 404
        
        # Save updated records
        save_records(updated_records)
        
        return jsonify({'success': True, 'message': 'Record deleted'}), 200
        
    except Exception as e:
        print(f"Delete error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print("🚀 Starting Eye Disease Detection Backend...")
    print("📡 Server will run on: http://localhost:5000")
    print("🔧 Make sure your YOLO model (best.pt) is in this folder")
    print("")
    
    # Initialize database
    init_database()
    
 