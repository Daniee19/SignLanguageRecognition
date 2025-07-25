from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect
# Librería para definir el tipo de respuesta
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, FileResponse
import cv2
import base64
from mediapipe_utils import detect_and_draw, extract_keypoints
from model import SignModel
import mediapipe as mp
import numpy as np
import json
import os

# ✅ Entrenamiento automático al finalizar la recolección

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

app = FastAPI()



# Establecer la ruta de los archivos estáticos y las plantillas


static_path = os.path.join(os.path.dirname(__file__), "static")

# Servir archivos estáticos
# Se pueden servir archivos estáticos como imágenes, CSS, JS, etc.
app.mount("/static", StaticFiles(directory=static_path), name="static")

#Se llama esto para que se pueda predecir (desde otra clase)
model = SignModel()


mp_holistic = mp.solutions.holistic

@app.get("/")
async def index():
    file_path = os.path.join(static_path, "html", "home.html")
    return FileResponse(file_path)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            data = await websocket.receive_text()

            header, b64 = data.split(",", 1)
            img_data = base64.b64decode(b64)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            img, results = detect_and_draw(frame, holistic)
            keypoints = extract_keypoints(results)
            
            action, prob = model.predict(keypoints)
            # re-encode frame
            _, buf = cv2.imencode('.jpg', img)
            jpg_b64 = base64.b64encode(buf).decode('utf-8')
            
            await websocket.send_json({
                "frame": f"data:image/jpeg;base64,{jpg_b64}",
                "action": str(action),

                # o None si prefieres
                "probability": float(prob) if prob is not None else 0.0
            })

SEQUENCES = 30
FRAMES = 30

# 👇 Este método se ejecuta solo una vez cuando ya tengas todas las secuencias
def entrenar():
    print("📌 Entrando al bloque de entrenamiento...")

    try:
        DATA_PATH = "MP_Data"
        if not os.path.exists("MP_Data"):
            raise FileNotFoundError("❌ La carpeta 'MP_Data' no existe.")
        
        labels = os.listdir(DATA_PATH)
        print("🗂 Contenido de MP_Data:", os.listdir(DATA_PATH))

        label_map = {label: idx for idx, label in enumerate(labels)}

        sequences, labels_encoded = [], []

        # for label in labels:
        #     for seq in range(SEQUENCES):
        #         window = []
        #         for frame in range(FRAMES):
        #             path = os.path.join(DATA_PATH, label, str(seq), f"{frame}.npy")
        #             if not os.path.exists(path):
        #                 raise FileNotFoundError(f"Falta el archivo: {path}")
        #             window.append(np.load(path, allow_pickle=True))

        #         sequences.append(window)
        #         labels_encoded.append(label_map[label])

        for label in labels:
            for seq in range(SEQUENCES):
                window = []
                for frame in range(FRAMES):
                    path = os.path.join(DATA_PATH, label, str(seq), f"{frame}.npy")
                    if not os.path.exists(path):
                        raise FileNotFoundError(f"❌ Falta el archivo: {path}")
                    try:
                        data = np.load(path, allow_pickle=True)

                        window.append(data)
                    except Exception as e:
                        raise ValueError(f"❌ Error al leer el archivo {path}: {e}")
                sequences.append(window)
                labels_encoded.append(label_map[label])

        for frame in range(FRAMES):
            path = os.path.join(DATA_PATH, label, str(seq), f"{frame}.npy")
            if not os.path.exists(path):
                raise FileNotFoundError(f"❌ Falta el archivo: {path}")
            if os.path.getsize(path) == 0:
                raise ValueError(f"⚠️ Archivo vacío: {path}")
            try:
                data = np.load(path, allow_pickle=True)
                window.append(data)
            except Exception as e:
                raise ValueError(f"Error al leer el archivo {path}: {e}")

        
        X = np.array(sequences)
        y = to_categorical(labels_encoded).astype(int)
        
        print("Entrenando con:")
        print("X shape:", X.shape)
        print("y shape:", y.shape)
        print("Etiquetas:", label_map)
        
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(FRAMES, X.shape[2])))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(LSTM(64, return_sequences=False, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(len(label_map), activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
        print("X shape:", X.shape)  # Debería ser (N, FRAMES, FEATURES)
        print("y shape:", y.shape)  # Debería ser (N, NUM_LABELS)

        model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))
        print("todo genial")
        os.makedirs("model", exist_ok=True)
        model.save('model/action.h5')

        with open("label_map.json", "w") as f:
            json.dump(label_map, f)

        print("✅ Entrenamiento finalizado y modelo guardado como 'action.h5'")
    except Exception as e:
        print("Error durante el entrenamiento:", e)

    


 #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
@app.websocket("/ws/entrenar")
async def websocket_entrenar(websocket: WebSocket):
    await websocket.accept()
    sequence = 0
    frame_num = 0
    word = None

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while sequence < SEQUENCES:
            try:
                data = await websocket.receive_text()
                json_data = json.loads(data)
                word = json_data["word"]

                b64 = data.split(",")[1]
                b64 += '=' * (-len(b64) % 4)

                img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
                img, results = detect_and_draw(img, holistic)

                if results is not None and results.pose_landmarks:
                    # Extraer pose landmarks
                    pose = []
                    for lm in results.pose_landmarks.landmark:
                        pose.extend([lm.x, lm.y, lm.z])

                    # Verificar que se extrajo data válida
                    if pose and all(p is not None for p in pose):
                        path = f"MP_Data/{word}/{sequence}"
                        os.makedirs(path, exist_ok=True)

                        np.save(f"{path}/{frame_num}.npy", np.array(pose))
                        print(f"📁 Guardado: {path}/{frame_num}.npy")

                        frame_num += 1

                        if frame_num == FRAMES:
                            sequence += 1
                            frame_num = 0
                            print(f"✅ Secuencia {sequence} guardada para la palabra '{word}'")
                    else:
                        print("⚠️ Datos vacíos, no se guarda el frame.")

                # Enviar imagen anotada al cliente
                _, buf = cv2.imencode('.jpg', img)
                await websocket.send_json({
                    "frame": f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}"
                })

            except Exception as e:
                print("Error en la captura:", e)
                continue

        # ✅ Llamar al entrenamiento después de terminar de grabar todas las secuencias
        try:
            print("🚀 Iniciando entrenamiento...")
            entrenar()
        except Exception as e:
            print("Error durante el entrenamiento:", e)




    # DATA_PATH = "MP_Data"
    # labels = os.listdir(DATA_PATH)
    # label_map = {label: idx for idx, label in enumerate(labels)}
    
    # sequences, labels_encoded = [], []

    # for label in labels:
    #     for seq in range(SEQUENCES):
    #         window = []
    #         for frame in range(FRAMES):
    #             path = os.path.join(DATA_PATH, label, str(seq), f"{frame}.npy")
    #             window.append(np.load(path))
    #         sequences.append(window)
    #         labels_encoded.append(label_map[label])

    # X = np.array(sequences)
    # y = to_categorical(labels_encoded).astype(int)

    # model = Sequential()
    # model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(FRAMES, X.shape[2])))
    # model.add(LSTM(128, return_sequences=True, activation='relu'))
    # model.add(LSTM(64, return_sequences=False, activation='relu'))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dense(len(label_map), activation='softmax'))

    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

    # model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))
    # print("todo genial")
    # model.save('action.h5')

    # with open("label_map.json", "w") as f:
    #     json.dump(label_map, f)

    # print("✅ Entrenamiento finalizado y modelo guardado como 'action.h5'")

 #==========
 #FUNCIONA BIEN
 #==========
# @app.websocket("/ws/entrenar")
# async def websocket_entrenar(websocket: WebSocket):
#     await websocket.accept()

#     SEQUENCES = 30
#     FRAMES = 30
#     sequence = 0
#     frame_num = 0
#     word = None # Puedes reemplazar esto si luego usarás json_data["word"]

#     with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#         while sequence < SEQUENCES:
#             try:
#                 data = await websocket.receive_text()
#                 json_data = json.loads(data)  # Parsea el JSON
#                 word = json_data["word"]
                
#                 b64 = data.split(",")[1]
#                 b64 += '=' * (-len(b64) % 4)

#                 img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
#                 img, results = detect_and_draw(img, holistic)

#                 if results is not None and results.pose_landmarks:
#                     path = f"MP_Data/{word}/{sequence}"
#                     os.makedirs(path, exist_ok=True)

#                     # Extraer coordenadas
#                     pose = []
#                     for lm in results.pose_landmarks.landmark:
#                         pose.extend([lm.x, lm.y, lm.z])
#                     np.save(f"{path}/{frame_num}.npy", np.array(pose))

#                     frame_num += 1

#                     if frame_num == FRAMES:
#                         sequence += 1
#                         frame_num = 0
#                         print(f"Secuencia {sequence} guardada para la palabra '{word}'")

#                 # Enviar frame con anotaciones
#                 _, buf = cv2.imencode('.jpg', img)
#                 await websocket.send_json({
#                     "frame": f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}"
#                 })

#             except Exception as e:
#                 print("Error:", e)
#                 continue


 
 
 
 
 #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# @app.websocket("/ws/entrenar")
# async def websocket_entrenar(websocket: WebSocket):
#     await websocket.accept()
#     with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#         while True:
#             try:
#                 data = await websocket.receive_text()
#                 json_data = json.loads(data)  # Parsea el JSON
#                 word = json_data["word"]
                
#                 b64 = data.split(",")[1]
#                 b64 += '=' * (-len(b64) % 4)  # Corrige padding en una sola línea

#                 img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
#                 img, _ = detect_and_draw(img, holistic)

#                 _, buf = cv2.imencode('.jpg', img)
#                 await websocket.send_json({"frame": f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}"})
#             except Exception as e:
             
#                 continue
            
            
        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# @app.websocket("/ws/entrenar")
# async def websocket_entrenar(websocket: WebSocket):
#     await websocket.accept()
#     contador = 0
#     word = None
    
    # with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            # try:
            #     while True:
            #         # try:
            #             data = await websocket.receive_text()
            #             print(data)
            #             json_data = json.loads(data)
            #             frame_b64 = json_data["frame"].split(",")[1]
            #             word = json_data["word"]

                        
            #             #Guardar los puntos en un JSON                        
            #             with open("actions.json", "w") as f:
            #                 json.dump(word, f)
                            
            #             print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
                        
                        
                    #     img = cv2.imdecode(np.frombuffer(base64.b64decode(frame_b64), np.uint8), cv2.IMREAD_COLOR)
                    #     img, landmarks = detect_and_draw(img, holistic)

                    #     if landmarks is not None:
                    #         os.makedirs(f"MP_Data/{word}", exist_ok=True)
                    #         np.save(f"MP_Data/{word}/{contador}.npy", landmarks)
                    #         contador += 1

                    #     _, buf = cv2.imencode('.jpg', img)
                    #     await websocket.send_json({
                    #         "frame": f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}"
                    #     })

                    # except Exception as e:
                    #     print("Error interno:", e)
                    #     # Aquí NO uses continue, porque si fue una desconexión, se repetirá infinitamente
                    #     break  # Sal del bucle si hay error (mejor aún, filtra el tipo si quieres ser más específico)

            # except WebSocketDisconnect:
            #     print("🔌 Cliente desconectado")
