import cv2
import numpy as np
import Jetson.GPIO as GPIO
import time
from cv2 import face

LED_PIN = 11
GPIO.setmode(GPIO.BOARD)  
GPIO.setup(LED_PIN, GPIO.OUT)

# Inicializa o reconhecedor facial
recognizer = face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')  # Certifique-se de ter o arquivo 'trainer.yml' no diretório

# Inicializa o classificador de face Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializa a câmera
cap = cv2.VideoCapture(0)  # Ajuste o índice se necessário

while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar imagem da câmera")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        # Nenhum rosto detectado, desliga o LED (ou relé)
        GPIO.output(LED_PIN, GPIO.LOW)
        print("Nenhum rosto detectado.")
    else:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            # Se o rosto é reconhecido como seu (ID 1) com confiança suficiente
            if id == 1 and confidence < 100:
                GPIO.output(LED_PIN, GPIO.HIGH)  # Acende o LED
                print("Rosto reconhecido!")
                # time.sleep(1)  # Pausa para evitar piscar rápido demais
                break  # Sai do loop for, pois já encontrou um rosto conhecido
            else:
                GPIO.output(LED_PIN, GPIO.LOW)  # Apaga o LED
                print("Rosto desconhecido.")
                # time.sleep(1)  # Pausa para evitar piscar rápido demais

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Pressione 'q' para sair
        break

# Limpeza
cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()
