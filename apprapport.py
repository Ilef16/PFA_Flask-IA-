from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import os
import tempfile
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from mongoengine import connect, Document, StringField, ObjectIdField, ReferenceField
from bson.objectid import ObjectId, InvalidId
from moviepy.editor import VideoFileClip, AudioFileClip

app = Flask(__name__)
CORS(app)

# Connect to MongoDB
DB_URL = "mongodb://127.0.0.1:27017/react-app"
connect(host=DB_URL)

class RapportModel(Document):
    meta = {'collection': 'rapports', 'strict': False}
    description = StringField(required=False)
    PDF = StringField(required=False)
    video = ObjectIdField(required=True)

class VideoModel(Document):
    meta = {'collection': 'videos', 'strict': False}
    video = StringField(required=True)
    poste = ObjectIdField(required=True)
    candidat = ObjectIdField(required=True)
    rapport = ObjectIdField(required=False)


emotion_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
emotion_model = load_model('modelFinalPfa.h5')
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if face_detector.empty():
    raise IOError("Failed to load face detector")

# Modifier le chemin de sortie pour enregistrer la vidéo traitée dans le dossier "uploads" du projet React
output_video_dir = r'C:\Users\DELL\Desktop\folder\foler\backend\uploads'
os.makedirs(output_video_dir, exist_ok=True)

@app.route('/detect_emotions', methods=['POST'])
def detect_emotions():
    try:
        video_file = request.files['video']
        poste_id = ObjectId(request.form['poste'])
        candidat_id = ObjectId(request.form['candidat'])

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            video_file.save(tmp_file.name)
            output_video_path, emotion_percentages = process_video(tmp_file.name)

            if output_video_path:
                output_video_name = os.path.basename(output_video_path)
                permanent_path = os.path.join(output_video_dir, output_video_name)
                os.rename(output_video_path, permanent_path)

                # Save VideoModel
                video_model = VideoModel(
                    video=output_video_name,
                    poste=poste_id,
                    candidat=candidat_id
                )
                video_model.save()
                print(f"VideoModel saved: {video_model.to_json()}")

                # Create and save RapportModel
                emotion_description = ', '.join([f"{emotion}: {percentage:.2f}%" for emotion, percentage in emotion_percentages.items()])
                rapport_model = RapportModel(
                    description=emotion_description,
                    PDF="",
                    video=video_model.id
                )
                rapport_model.save()
                print(f"RapportModel saved: {rapport_model.to_json()}")

                # Update VideoModel with rapport reference
                video_model.rapport = rapport_model.id
                video_model.save()

                return jsonify({"url": f"http://localhost:4000/uploads/{output_video_name}"})
            else:
                print("Error: Could not process video")
                return jsonify({"error": "Could not process video"}), 500
    except KeyError as e:
        print(f"Missing required field: {e}")
        return jsonify({"error": f"Missing required field: {e}"}), 400
    except InvalidId as e:
        print(f"Invalid ObjectId format: {e}")
        return jsonify({"error": f"Invalid ObjectId format: {e}"}), 400
    except Exception as e:
        print(f"General error: {e}")
        return jsonify({"error": str(e)}), 500


def process_video(input_video_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Video capture could not be opened")
        return None, None

    try:
        # Extract audio
        video_clip = VideoFileClip(input_video_path)
        audio_path = input_video_path.replace('.mp4', '.mp3')
        video_clip.audio.write_audiofile(audio_path)
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None, None

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    emotion_counts = {emotion: 0 for emotion in emotion_dict.values()}
    frame_count = 0

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            out = cv2.VideoWriter(tmp_file.name, fourcc, fps, (frame_width, frame_height))
            if not out.isOpened():
                print("Error: Video writer could not be opened")
                cap.release()
                return None, None

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
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
                        emotion = emotion_dict[maxindex]
                        emotion_counts[emotion] += 1
                        cv2.putText(frame, emotion, (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    except Exception as e:
                        print(f"Error during prediction: {e}")

                out.write(frame)

            cap.release()
            out.release()

            # Combine audio and video
            processed_clip = VideoFileClip(tmp_file.name)
            audio_clip = AudioFileClip(audio_path)
            final_clip = processed_clip.set_audio(audio_clip)
            output_path = tmp_file.name.replace('.mp4', '_with_audio.mp4')
            final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

            # Calculate emotion percentages
            emotion_percentages = {emotion: (count / frame_count) * 100 for emotion, count in emotion_counts.items()}

            return output_path, emotion_percentages

    except Exception as e:
        print(f"Error processing video: {e}")
        return None, None




@app.route('/get_videos', methods=['GET'])
def get_videos():
    try:
        videos = VideoModel.objects()
        video_list = [json_util.loads(video.to_json()) for video in videos]

        for video in video_list:
            del video['_id']
            video['poste'] = str(video['poste'])
            video['candidat'] = str(video['candidat'])
            if 'rapport' in video:
                video['rapport'] = str(video['rapport'])

        return jsonify(video_list)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/uploads/<filename>', methods=['GET'])
def send_video(filename):
    if filename and filename != 'undefined':
        video_path = os.path.join(output_video_dir, filename)
        if os.path.exists(video_path):
            return send_file(video_path)
        else:
            return jsonify({"error": "File not found"}), 404
    else:
        return jsonify({"error": "Invalid filename"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)