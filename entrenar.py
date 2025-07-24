import os
import cv2
import numpy as np
import mediapipe as mp

# === Palabra a entrenar ===
nueva_palabra = "otro"

# Número de videos y frames por video
secuencia_video = 30
secuencia_frame = 30

# Guardado
DATA_PATH = os.path.join("MP_Data", nueva_palabra)

# Crear las carpetas para guardar los datos
for sec_video in range(secuencia_video):
    try:
        os.makedirs(os.path.join(DATA_PATH, str(sec_video)))
    except:
        pass

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


def mediapipe_deteccion(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    resultados = model.process(image)
    image.flags.writeable = True
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), resultados


def dibujar_landmarks(image, resultados):
    mp_drawing.draw_landmarks(
        image, resultados.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
    mp_drawing.draw_landmarks(
        image, resultados.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(
        image, resultados.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(
        image, resultados.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


def extraer_puntos(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten(
    ) if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten(
    ) if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten(
    ) if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten(
    ) if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


# Captura y guardado
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for sec_video in range(secuencia_video):
        for frame_number in range(secuencia_frame):
            ret, frame = cap.read()
            image, resultados = mediapipe_deteccion(frame, holistic)
            dibujar_landmarks(image, resultados)

            if frame_number == 0:
                cv2.putText(image, f'EMPEZANDO {nueva_palabra.upper()}...', (
                    20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
                cv2.imshow('Recolectando', image)
                cv2.waitKey(2000)
            else:
                cv2.putText(image, f'{nueva_palabra} | Video: {sec_video} | Frame: {frame_number}', (
                    10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('Recolectando', image)

            keypoints = extraer_puntos(resultados)
            np.save(os.path.join(DATA_PATH, str(sec_video),
                    str(frame_number)), keypoints)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
