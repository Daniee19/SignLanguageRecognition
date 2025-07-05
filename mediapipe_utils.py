import cv2, numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face = mp.solutions.face_mesh

def detect_and_draw(image, holistic):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_rgb.flags.writeable = False
    results = holistic.process(img_rgb)
    img_rgb.flags.writeable = True
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    # dibujar landmarks
    #resultados.pose_landmarks -> Trae los puntos obtenidos de la imagen
    #mp_holistic.POSE_CONNECTIONS -> Establece como se deben de conectar los puntos
    mp_drawing.draw_landmarks(img, results.face_landmarks, mp_face.FACEMESH_TESSELATION,
                              landmark_drawing_spec=mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), #puntos
                              connection_drawing_spec =mp_drawing.DrawingSpec(color=(80,110,10), thickness=1) #líneas
                              ) #modifica la imagen pero no retorna nada
    
    mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), #puntos
                              connection_drawing_spec =mp_drawing.DrawingSpec(color=(80,110,10), thickness=1) #líneas
                              )
    mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), #puntos
                              connection_drawing_spec =mp_drawing.DrawingSpec(color=(80,110,10), thickness=1) #líneas
                              )
    mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), #puntos
                              connection_drawing_spec =mp_drawing.DrawingSpec(color=(80,110,10), thickness=1) #líneas
                              )
    return img, results

def extract_keypoints(results):
    #Cuando no se encuentra ningun punto tanto x, y, z
    #El array debe estar en 1D para manejarlo como red neuronal (El flatten se encarga de hacerlo en 1D)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])
