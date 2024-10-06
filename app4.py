from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tempfile
import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from mongoengine import connect, Document, StringField

app = Flask(__name__)
CORS(app)

# Connect to MongoDB
DB_URL = "mongodb://127.0.0.1:27017/react-app"
connect(host=DB_URL)

class RapportModel(Document):
    meta = {'collection': 'rapports', 'strict': False}
    description = StringField(required=True)
    PDF = StringField(required=True)
    video = StringField(required=False)

emotion_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
emotion_model = load_model('modelFinalPfa.h5')
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

output_video_dir = os.path.join(app.root_path, 'traitement')
os.makedirs(output_video_dir, exist_ok=True)

if face_detector.empty():
    raise IOError("Failed to load face detector")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_emotions', methods=['POST'])
def detect_emotions():
    try:
        video_file = request.files['video']
        
        # Save the uploaded video file to the 'traitement' directory
        video_filename = os.path.join(output_video_dir, video_file.filename)
        video_file.save(video_filename)
        
        emotion_percentages = process_video(video_filename)
        if emotion_percentages:
            report_content = generate_emotion_report(emotion_percentages)

            rapport_model = RapportModel(
                description="Emotion report for the video.",
                PDF="",  # Add logic to generate PDF if necessary
                video=video_filename
            )
            rapport_model.save()
            return jsonify({"report": report_content})
        else:
            return jsonify({"error": "Could not process video"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def process_video(input_video_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        return None

    emotion_counts = {emotion: 0 for emotion in emotion_dict.values()}
    total_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=10)

        for (x, y, w, h) in faces:
            roi_gray_frame = gray_frame[y:y+h, x:x+w]
            gray_input = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            gray_input = gray_input / 255.0

            try:
                emotion_prediction = emotion_model.predict(gray_input)
                maxindex = int(np.argmax(emotion_prediction))
                emotion_counts[emotion_dict[maxindex]] += 1
            except Exception as e:
                print(f"Error during prediction: {e}")

    cap.release()
    if total_frames == 0:
        return None
    emotion_percentages = {emotion: (count / total_frames) * 100 for emotion, count in emotion_counts.items()}
    return emotion_percentages

def generate_emotion_report(emotion_percentages):
    report_lines = [f"{emotion}: {percentage:.2f}%" for emotion, percentage in emotion_percentages.items()]
    return "\n".join(report_lines)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)
