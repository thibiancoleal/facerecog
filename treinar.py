from PIL import Image
import cv2
from cv2 import face
import os
import numpy as np

# Caminho para o seu diretório de fotos
path = './fotos'

# Inicializar o reconhecedor de faces
recognizer = face.LBPHFaceRecognizer_create()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Função para obter imagens e labels
def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.startswith('.')]
    face_samples = []
    ids = []
    for imagePath in image_paths:
        PIL_img = Image.open(imagePath).convert('L')  # Convertendo para grayscale
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[0])
        faces = face_cascade.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            face_samples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)
    return face_samples, ids

faces, ids = get_images_and_labels(path)
recognizer.train(faces, np.array(ids))

# Salvar o modelo treinado
recognizer.save('trainer.yml')
