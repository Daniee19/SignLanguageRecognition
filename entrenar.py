import os
import cv2
import numpy as np
import mediapipe as mp
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # para esconder warnings de TensorFlow

# === Palabra a entrenar ===

nueva_palabra = "comeer"

# Número de videos y frames por video
secuencia_video = 30
secuencia_frame = 30

# Guardado
DATA_PATH = os.path.join("MP_Data")

# Crear las carpetas para guardar los datos
for sec_video in range(secuencia_video):
    try:
        os.makedirs(os.path.join(DATA_PATH, nueva_palabra,str(sec_video)))
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
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("❌ No se pudo abrir la cámara.")
    
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    accion = nueva_palabra
    
    for sec_video in range(secuencia_video):
            for frame_number in range(secuencia_frame):
                #Leer por el video
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("❌ No se pudo capturar el frame")
                    continue  # o break, si quieres salir del bucle
                
                #Realizar detecciones
                image, resultados = mediapipe_deteccion(frame, holistic)
                dibujar_landmarks(image, resultados)

                if frame_number == 0:
                    cv2.putText(image, "EMPEZANDO LA RECOLECCION", (120,200),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 4, cv2.LINE_AA)
                    cv2.putText(image, "Recolectando {} frame Nro: {} Numero de video: {}".format(accion, frame_number, sec_video), (15,12),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV feed', image)
                    #Se hace esto para tener tiempo de ubicarnos bien
                    cv2.waitKey(2000)
                else:
                    cv2.putText(image, "Recolectando {} frame Nro: {} Numero de video: {}".format(accion, frame_number, sec_video), (15,12),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV feed', image)


                #Exportar puntos claves de cada frame
                keypoints = extraer_puntos(resultados)
                npy_path = os.path.join(DATA_PATH ,accion, str(sec_video), str(frame_number))
                np.save(npy_path, keypoints)

                #Si se presiona q se sale del bucle
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
cap.release()
cv2.destroyAllWindows()
