from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np

# from PIL import Image
import io
import tensorflow as tf
from tensorflow import keras

# import numpy as np

app = Flask(__name__)
# print(tf.__version__)

# print(keras.__version__)
# emotion_model = load_model('best_model.h5')
model = load_model('modelFinalPfa.h5')


@app.route("/", methods=["GET"])
def hello():
    return jsonify({"hello": "kevin"})

# @app.before_first_request
# def load():
#     model_path = "modelFinalPfa.h5"
#     model = load_model(model_path, compile=False)
#     return model

# Chargement du model
# model = load()

# def preprocess(img):
#     img = img.resize((224, 224))
#     img = np.asarray(img)
#     img = np.expand_dims(img, axis=0)
#     return img


# @app.route("/predict", methods=['POST'])
# def predict():
#     # recuperer l'image
#     file = request.files['file']
#     image = file.read()

#     # Ouvrir l'image
#     img = Image.open(io.BytesIO(image))

#     #traitement de l'image
#     img_processed = preprocess(img)

#     # predictions
#     pred = model.predict(img_processed)

#     rec = pred[0][0].tolist()

#     return jsonify({"predictions" : rec})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)