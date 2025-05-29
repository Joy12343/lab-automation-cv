from flask import Flask, render_template, jsonify
import cv2
import base64
import numpy as np
import traceback
from heinsight_app import HeinSight

app = Flask(__name__)

# Use your local lab camera stream URL
video_source = "http://10.63.6.98:5000/video_feed2"
heinsight = None  # Will be initialized in main

@app.route('/')
def index():
    return render_template('index_5.html')  # New frontend

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        cap = cv2.VideoCapture(video_source)
        success, frame = cap.read()
        cap.release()

        if not success:
            return jsonify({'error': 'Failed to capture frame'}), 500

        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run detection
        result_img = heinsight.run(frame_rgb)

        # Convert back to BGR for encoding
        result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', result_bgr)
        result_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({'processed_image': f'data:image/jpeg;base64,{result_base64}'}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    heinsight = HeinSight(
        vial_model_path="models/best_vessel.pt",
        contents_model_path="models/best_content.pt"
    )
    app.run(host="0.0.0.0", port=5000, debug=True)
