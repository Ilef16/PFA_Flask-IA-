from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import os
import tempfile
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sqlalchemy import create_engine, Column, Integer, String, BLOB
from sqlalchemy.orm import Session
from sqlalchemy.ext.declarative import declarative_base
from bson.objectid import ObjectId

app = Flask(__name__)
CORS(app)

# Base de données SQLAlchemy
Base = declarative_base()
engine = create_engine('sqlite:///emotions.db')

class EmotionVideo(Base):
    __tablename__ = 'emotion_videos'
    id = Column(Integer, primary_key=True)
    poste_id = Column(String)
    candidat_id = Column(String)
    rapport = Column(String)
    video = Column(BLOB)

Base.metadata.create_all(engine)

emotion_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
emotion_model = load_model('modelFinalPfa.h5')
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if face_detector.empty():
    raise IOError("Failed to load face detector")

output_video_dir = os.path.join(app.root_path, 'traitement')
os.makedirs(output_video_dir, exist_ok=True)


@app.route('/detect_emotions', methods=['POST'])
def detect_emotions():
    video_file = request.files['video']
    
    try:
        poste_id = str(ObjectId(request.form['poste']))
    except (InvalidId, KeyError) as e:
        error_message = f"Erreur: le paramètre 'poste' n'est pas un ID valide - {e}"
        print(error_message)
        return jsonify({"error": error_message}), 400
    
    try:
        candidat_id = str(ObjectId(request.form['candidat']))
    except (InvalidId, KeyError) as e:
        error_message = f"Erreur: le paramètre 'candidat' n'est pas un ID valide - {e}"
        print(error_message)
        return jsonify({"error": error_message}), 400

    rapport = request.form['rapport']

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        video_file.save(tmp_file.name)
        output_video_path = process_video(tmp_file.name)

        if output_video_path:
            with open(output_video_path, 'rb') as f:
                video_data = f.read()

            emotion_video = EmotionVideo(
                poste_id=poste_id,
                candidat_id=candidat_id,
                rapport=rapport,
                video=video_data
            )

            with Session(engine) as session:
                session.add(emotion_video)
                session.commit()

            if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
                output_video_name = os.path.basename(output_video_path)
                destination_path = os.path.join(output_video_dir, output_video_name)
                os.rename(output_video_path, destination_path)
                return jsonify({"url": f"http://localhost:4000/traitement/{output_video_name}"})
            else:
                return jsonify({"error": "Output video file does not exist or is empty."}), 500
        else:
            return jsonify({"error": "Could not process video"}), 500

def process_video(input_video_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        return None

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        out = cv2.VideoWriter(tmp_file.name, fourcc, fps, (frame_width, frame_height))
        if not out.isOpened():
            cap.release()
            return None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=10)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
                roi_gray_frame = gray_frame[y:y+h, x:x+w]
                roi_color_frame = frame[y:y+h, x:x+w]

                gray_input = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
                color_input = np.expand_dims(cv2.resize(roi_color_frame, (48, 48)), 0)

                gray_input = gray_input / 255.0
                color_input = color_input / 255.0

                try:
                    emotion_prediction = emotion_model.predict([gray_input, color_input])
                    maxindex = int(np.argmax(emotion_prediction))
                    cv2.putText(frame, emotion_dict[maxindex], (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                except Exception as e:
                    print(f"Error during prediction: {e}")

            out.write(frame)

        cap.release()
        out.release()
        return tmp_file.name

@app.route('/traitement/<filename>')
def send_video(filename):
    return send_file(os.path.join(output_video_dir, filename))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)
