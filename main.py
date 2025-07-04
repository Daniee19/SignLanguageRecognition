from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
# Librería para definir el tipo de respuesta
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, FileResponse
import cv2, base64
from mediapipe_utils import detect_and_draw, extract_keypoints
from model import SignModel
import mediapipe as mp
import numpy as np
import os

app = FastAPI()


# Establecer la ruta de los archivos estáticos y las plantillas

static_path = os.path.join(os.path.dirname(__file__), "static/")
templates_path = os.path.join(os.path.dirname(__file__), "templates/")

# Servir archivos estáticos
# Se pueden servir archivos estáticos como imágenes, CSS, JS, etc.
app.mount("/static", StaticFiles(directory=static_path), name="static")

model = SignModel()
mp_holistic = mp.solutions.holistic

@app.get("/")
async def index():
    return FileResponse("static/client.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            data = await websocket.receive_text()
            header, b64 = data.split(",",1)
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
                "probability": float(prob) if prob is not None else 0.0  # o None si prefieres
            })
