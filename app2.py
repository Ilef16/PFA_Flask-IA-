import cv2
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, request, send_file, render_template, jsonify
import os
import tempfile
import datetime
from werkzeug.utils import secure_filename


app = Flask(__name__, template_folder='templates')

# Dictionnaire des émotions
emotion_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Charger le modèle
emotion_model = load_model('modelFinalPfa.h5')

# Charger le classificateur de visages
face_detector_path = 'haarcascade_frontalface_default.xml'
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Vérifier si le classificateur a été chargé correctement
if face_detector.empty():
    print("Error: Failed to load face detector")
    raise IOError("Failed to load face detector")

# Chemin de destination pour les vidéos générées
output_video_dir = os.path.join(app.root_path, 'traitement')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_emotions', methods=['POST'])
def detect_emotions():
    # Récupérer la vidéo de la requête
    video_file = request.files['video']
    
    # Obtenir le nom de la vidéo
    video_name = secure_filename(video_file.filename)

    # Enregistrer la vidéo temporairement
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        video_file.save(tmp_file.name)

        # Traiter la vidéo
        output_video_path = process_video(tmp_file.name)

        if output_video_path:
            # Renvoyer la vidéo traitée
            if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
                # Enregistrer le chemin complet du fichier vidéo généré dans le dossier de traitement
                output_video_name = os.path.basename(output_video_path)
                destination_path = os.path.join(output_video_dir, output_video_name)
                os.rename(output_video_path, destination_path)
                
                # Calculer les pourcentages d'émotions
                emotion_percentages = calculate_emotion_percentages(destination_path)

                # Enregistrer les pourcentages d'émotions dans un fichier texte avec le nom de la vidéo
                save_emotion_percentages(emotion_percentages, video_name)

                # Renvoyer le chemin du fichier vidéo généré
                return send_file(destination_path, mimetype='video/mp4')
            else:
                print("Error: Output video file does not exist or is empty.")
                return jsonify({"error": "Output video file does not exist or is empty."}), 500
        else:
            return jsonify({"error": "Could not process video"}), 500

def process_video(input_video_path):
    # Lire la vidéo
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        return None

    # Paramètres pour la sauvegarde de la vidéo
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Créer un fichier temporaire pour la vidéo de sortie
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        out = cv2.VideoWriter(tmp_file.name, fourcc, fps, (frame_width, frame_height))

        if not out.isOpened():
            print("Error: Could not open VideoWriter.")
            cap.release()
            return None

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=10)

            for (x, y, w, h) in num_faces:
                cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
                roi_gray_frame = gray_frame[y:y+h, x:x+w]
                roi_color_frame = frame[y:y+h, x:x+w]

                # Préparer les entrées pour le modèle
                gray_input = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
                color_input = np.expand_dims(cv2.resize(roi_color_frame, (48, 48)), 0)

                # Normaliser les entrées
                gray_input = gray_input / 255.0
                color_input = color_input / 255.0

                # Prédire l'émotion
                try:
                    emotion_prediction = emotion_model.predict([gray_input, color_input])
                    maxindex = int(np.argmax(emotion_prediction))
                    cv2.putText(frame, emotion_dict[maxindex], (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                except Exception as e:
                    print(f"Error during prediction: {e}")

            out.write(frame)
            frame_count += 1

        cap.release()
        out.release()

        print(f"Processed {frame_count} frames.")
        print(f"Output video path: {tmp_file.name}")

        return tmp_file.name

def calculate_emotion_percentages(video_path):
    # Initialiser un dictionnaire pour compter le nombre de fois que chaque émotion est détectée
    emotion_counts = {emotion: 0 for emotion in emotion_dict.values()}
    
    # Ouvrir la vidéo pour la lecture
    cap = cv2.VideoCapture(video_path)
    
    # Boucler sur chaque frame de la vidéo
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convertir la frame en niveaux de gris pour la détection des visages
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Détecter les visages dans la frame
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=10)
        
        # Boucler sur chaque visage détecté dans la frame
        for (x, y, w, h) in num_faces:
            roi_gray_frame = gray_frame[y:y+h, x:x+w]
            roi_color_frame = frame[y:y+h, x:x+w]

            # Préparer les entrées pour le modèle
            gray_input = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            color_input = np.expand_dims(cv2.resize(roi_color_frame, (48, 48)), 0)

            # Normaliser les entrées
            gray_input = gray_input / 255.0
            color_input = color_input / 255.0

            # Prédire l'émotion
            emotion_prediction = emotion_model.predict([gray_input, color_input])
            maxindex = int(np.argmax(emotion_prediction))
            
            # Mettre à jour le compteur d'émotions détectées
            emotion_counts[emotion_dict[maxindex]] += 1

    # Fermer la vidéo
    cap.release()

    # Calculer les pourcentages d'émotions
    total_frames = sum(emotion_counts.values())
    emotion_percentages = {emotion: (count / total_frames) * 100 for emotion, count in emotion_counts.items()}
    
    return emotion_percentages


def save_emotion_percentages(emotion_percentages, video_name):
    # Vérifier si le dossier de destination existe, sinon le créer
    if not os.path.exists(output_video_dir):
        os.makedirs(output_video_dir)
    
    # Générer le nom du fichier texte basé sur le nom de la vidéo
    text_file_name = f"{os.path.splitext(video_name)[0]}_emotion_percentages.txt"
    text_file_path = os.path.join(output_video_dir, text_file_name)
    
    # Ouvrir le fichier texte en mode écriture
    with open(text_file_path, 'w') as f:
        # Écrire les pourcentages d'émotions dans le fichier texte
        for emotion, percentage in emotion_percentages.items():
            f.write(f"{emotion}: {percentage:.2f}%\n")


# def save_emotion_percentages(emotion_percentages):
#     # Créer un nom de fichier unique en ajoutant la date et l'heure actuelles
#     current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     filename = f"emotion_percentages_{current_datetime}.txt"
    
#     # Chemin complet du fichier texte pour enregistrer les pourcentages d'émotions
#     text_file_path = os.path.join(output_video_dir, filename)
    
#     # Ouvrir le fichier texte en mode écriture
#     with open(text_file_path, 'w') as f:
#         # Écrire les pourcentages d'émotions dans le fichier texte
#         for emotion, percentage in emotion_percentages.items():
#             f.write(f"{emotion}: {percentage:.2f}%\n")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)
