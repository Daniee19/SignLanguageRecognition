import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
#Paso 6 preprocesar datos, crear etiquetas y caracteristicas
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
#Paso 7 Construir y entrenar LSTM Neural Network
from tensorflow.keras.models import Sequential
#LSTM -> Es un modelo que permite hacer detección de acciones
from tensorflow.keras.layers import LSTM, Dense
#Hacer un registro interno del tablero de tensor
from tensorflow.keras.callbacks import TensorBoard
#Para que el entrenamiento se detenga cuando ya no se mejora
from tensorflow.keras.callbacks import EarlyStopping
#Paso 10 para la matriz de confusión, para probar mejor el modelo
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model

to_categorical([0,1,2], num_classes=3)
mp_holistic = mp.solutions.holistic #Modelo holístico (permite establecer los puntos)
mp_drawing = mp.solutions.drawing_utils #Dibujar lineas
mp_face_mesh = mp.solutions.face_mesh #Modelo para reconocer los puntos de la cara

#Direccion de las acciones
DATA_PATH=os.path.join("MP_Data")
#Cada acción tendrá 30 secuencia de video
acciones = np.array(["hola", "gracias", "te amo"])
#Cada video estará conformado de 30 frames
secuencia_video = 30
secuencia_frame = 30


#Crear las carpetas donde se almacenará los puntos entrenados
# for accion in acciones:
#     for sec_video in range(secuencia_video):
#             try:
#                 os.makedirs(os.path.join(DATA_PATH, accion, str(sec_video)))
#             except:
#                 pass


#Lo procesa para ser dibujado
def mediapipe_deteccion(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    #Detectar los puntos del cuerpo, cara, manos
    resultados = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, resultados

#Lo dibuja
def dibujar_landmarks(image, resultados):
    #resultados.pose_landmarks -> Trae los puntos obtenidos de la imagen
    #mp_holistic.POSE_CONNECTIONS -> Establece como se deben de conectar los puntos
    mp_drawing.draw_landmarks(image, resultados.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                              landmark_drawing_spec=mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), #puntos
                              connection_drawing_spec =mp_drawing.DrawingSpec(color=(80,110,10), thickness=1) #líneas
                              ) #modifica la imagen pero no retorna nada
    
    mp_drawing.draw_landmarks(image, resultados.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), #puntos
                              connection_drawing_spec =mp_drawing.DrawingSpec(color=(80,110,10), thickness=1) #líneas
                              )
    mp_drawing.draw_landmarks(image, resultados.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), #puntos
                              connection_drawing_spec =mp_drawing.DrawingSpec(color=(80,110,10), thickness=1) #líneas
                              )
    mp_drawing.draw_landmarks(image, resultados.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), #puntos
                              connection_drawing_spec =mp_drawing.DrawingSpec(color=(80,110,10), thickness=1) #líneas
                              )

def extraer_puntos(results):
    #Cuando no se encuentra ningun punto tanto x, y, z
    #El array debe estar en 1D para manejarlo como red neuronal (El flatten se encarga de hacerlo en 1D)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

            
# #ENTRENAR LAS SEÑAS
#Keypoints using mp holistic
# cap = cv2.VideoCapture(0); #Se escoge la cámara 
# #Asignar el modelo de mediapipe
# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
#     for accion in acciones:
#         for sec_video in range(secuencia_video):
#             for frame_number in range(secuencia_frame):
#                 #Leer por el video
#                 ret, frame = cap.read()
                
#                 #Realizar detecciones
#                 image, resultados = mediapipe_deteccion(frame, holistic)
#                 dibujar_landmarks(image, resultados)
                
#                 if frame_number == 0:
#                     cv2.putText(image, "EMPEZANDO LA RECOLECCION", (120,200),
#                                 cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 4, cv2.LINE_AA)
#                     cv2.putText(image, "Recolectando {} frame Nro: {} Numero de video: {}".format(accion, frame_number, sec_video), (15,12),
#                                 cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
#                     cv2.imshow('OpenCV feed', image)
#                     #Se hace esto para tener tiempo de ubicarnos bien
#                     cv2.waitKey(2000)
#                 else:
#                     cv2.putText(image, "Recolectando {} frame Nro: {} Numero de video: {}".format(accion, frame_number, sec_video), (15,12),
#                                 cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
#                     cv2.imshow('OpenCV feed', image)
            
                
#                 #Exportar puntos claves de cada frame
#                 keypoints = extraer_puntos(resultados)
#                 npy_path = os.path.join(DATA_PATH ,accion, str(sec_video), str(frame_number))
#                 np.save(npy_path, keypoints)
                                
#                 #Si se presiona q se sale del bucle
#                 if cv2.waitKey(10) & 0xFF == ord('q'):
#                     break
    
# cap.release()
# cv2.destroyAllWindows()


####### 127 - 194
#Son las etiquetas
acciones = np.array(["hola", "gracias", "te amo"])
#Devuelve un diccionario de clave valor-> "hola" : 0
label_map = {
    label: num 
    for num, label in enumerate(acciones)
    }
print(label_map)

# #Con los datos obtenidos del entrenamiento se va a realizar una matriz grande que agrupe 30 cuadros de 1662 puntos
sequences, labels = [], []
for accion in acciones:
    
    for sequence in range(secuencia_video):
        #30 x 3 = 90 arrays a crear
        window = []
        for frame_num in range(secuencia_frame):
            res = np.load(os.path.join(DATA_PATH, accion, str(sequence), "{}.npy".format(frame_num)))
            #30 frames de 1662 puntos
            window.append(res)
        #1 secuencia -> 30 frames de 1662 puntos    
        sequences.append(window)
        
        labels.append(label_map[accion]) #Si es hola retorna un 0 #Se repite 30 veces por cada acción (secuencia), 30 veces 0, 30 veces 1, 30 veces 2

# #Datos de entrada
x = np.array(sequences) #.shape -> (90, 30, 1662) #toda la información en un array ejm: [accion=0[sec=0 [frag=0.npy..29.npy]]]
# #Etiquetas
y = to_categorical(labels).astype(int) #se hace esto para trabajar las etiquetas con modelos de deep learning #0-> [1 0 0] #1-> [0 1 0] #2-> [0 0 1]
# #print(y)

# #Entrenamiento y pruebas (testing)
# #x_train, y_train -> Aprende patrones y ajusta sus parámetros
# #x_test, y_test -> Se pone a prueba si aprendió bien
# #ASIGNA NÚMEROS DE MANERA ALEATORIA a las variables
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.05)
# #x_test y y_test -> 90 * 0.05 = 4.5 -> 5 arrays

# #Permite monitorear tu actividad neuronal (el entrenamiento y la precisión)
log_dir = os.path.join("Logs")
tb_callback = TensorBoard(log_dir=log_dir)

# #LSTM se usa dentro de modelos secuenciales como Keras (Sequential())
# #LSTM es una red neuronal secuencial de Keras que permite procesar secuencias temporales de datos 
# #Modelo de secuencia en Keras (apilar capas 1 en 1)
# model = Sequential()
#Cuando retorna true retorna toda la secuencia completa, útil si hay más LSTM
#Agrega una capa de LSTM con 64 dimensiones
#Mientras se agrega más capturará patrones más complejos
#El input_shape sirve como entrada para indicar al Keras que tiene que esperar como entrada (30 secuencias que cada uno contiene 1662 datos)
# model.add(LSTM(64, return_sequences=True, activation="relu", input_shape=(30, 1662)))
# model.add(LSTM(128, return_sequences=True, activation="relu"))
# #El false indica que se va a reducir la dimensión de salida usando a algo compatible como Dense
# #Se baja a 64 para comprimir la información y tomar decisiones
# model.add(LSTM(64, return_sequences=False, activation="relu"))
# #64 neuronas -> No hay una regla exacta. 64 Es un número común que suele funcionar bien
# model.add(Dense(64, activation="relu"))
# model.add(Dense(32, activation="relu"))
# #Convierte la salida en una distribución de probabilidades, adecuada para clasificación multiclase
# #Al final la capa de salida tiene 3 neuronas, el actions.shape[0] obtiene todas las clases
# model.add(Dense(acciones.shape[0], activation= "softmax"))

# #Configura el modelo para entrenar
# model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])
# #Entrenar
# #Detener el entranimiento si no se mejora

# early_stop = EarlyStopping(monitor="val_categorical_accuracy", patience=20, restore_best_weights=True)
# history = model.fit(x_train, y_train, epochs=1000, 
#           validation_split=0.2,
#           callbacks=[early_stop, tb_callback])

# #Resumen del entrenamiento
# model.summary()

# epocas_recorridas=len(history.history["loss"])

# #VIDEO 8
# #Predicción en base al entrenamiento
# #El predict funciona por los pesos que se obtuvieron al entrenar
# #Probabilidades en diferencia
# #Si se detectaron los puntos más en una parte, eso aumenta la probabilidad de que el modelo reconozca bien el gesto
# #Si tenia que haber movimiento al grabar los keymarks #Si me puse quieto por eso no lo reconoce bien y los detecta a ambos en los mismos puntos o similares
# # Tu modelo usa una arquitectura LSTM, que analiza secuencias en el tiempo. No solo se fija en cómo están los puntos ahora, sino en cómo cambian.
# # Entonces:# - Si mantienes una pose estática, aunque sea la correcta (ej. gesto "gracias"),# - Pero no hay movimiento entre frames (los keypoints se mantienen iguales),

# #Acuerdate que es el 5% así que va a dar 5 resultados
# resultado = model.predict(x_test) #Retorna una muestra de 5 resultados con 3 valores, los cuales son comprimidos por el Dense

# print("El x_test es:", x_test)
# print("El resultado es: ", resultado)
# #Se obtiene el valor 
# #El np.argmax() devuelve el índice del resultado mayor de ese array
# print("La acción de resultado[0] es : ", acciones[np.argmax(resultado[0])])  #retorna 1, es decir que la acción detectada es de 1 ("hola") Esto en base a los registros y su precisión

# print(f"Epocas recorridas: {epocas_recorridas} epochs")
# for i in range(5):
#     print(acciones[np.argmax(resultado[i])])
#     print(acciones[np.argmax(y_test[i])])
#     print("\n")

# # #9. Guardar pesos
# model.save("action.h5")
model = load_model("action.h5")

#Cambiar de one-hot-encoding a categorical label (en una misma fila)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat=np.argmax(model.predict(x_test), axis=1).tolist()

print(multilabel_confusion_matrix(ytrue,yhat))



print("")
print("======================")
print("El y_test es:")
print("")
print(y_test)
print("")
print("======================")
print("")
print("El predict es: ", model.predict(x_test))
print("")
print("======================")
print("")
print("El predict con axis=1 es:", np.argmax(model.predict(x_test), axis=1).tolist())
print("")
print("La precisión es: ", accuracy_score(ytrue, yhat)*100,"%")
print("La precisión es: ", accuracy_score(ytrue, yhat))

#VAMOS A PROBAR QUE FUNCIONE =================

#Nuevas variables detectadas
sequence = []
sentence = []
threshold = 0.5 #limite #tolerancia

colores_probabilidad_dinamico = [(245,117,16), (117,245,16), (16,117,245)]

def prob_viz(res, acciones, input_frame, colores ):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colores[num], -1)
        cv2.putText(output_frame, acciones[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame
    
cap = cv2.VideoCapture(0); #Se escoge la cámara 
#Asignar el modelo de mediapipe
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        #Tomar decisiones
        imagen, resultados = mediapipe_deteccion(frame, holistic)
        print(resultados)
        
        #Dibujar puntos de marca
        dibujar_landmarks(imagen, resultados)

        #2. Lógica de predicción continua
        keypoints = extraer_puntos(resultados)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            #solo el primer segundo (30fragmentos = 1 segundo) y de ahí se ejecutará de corrido
            # np.expand_dims convierte la secuencia (30, 1662) en (1, 30, 1662) para que tenga el formato que espera el modelo (batch_size, time_steps, features)
            res = model.predict(np.expand_dims(sequence, axis=0))[0] #Dará el resultado de las probabilidades de las tres clases a lo largo de 30 fragmentos de 1660 puntos
            print(acciones[np.argmax(res)]) #el argmax da el indice que mas probabilidad tenga similitud con la clase de los ULTIMOS 30 fragmentos hayados


            #3. Lógica visual
            if res[np.argmax(res)] > threshold:
                if len(sentence) > 0:
                    if acciones[np.argmax(res)] != sentence[-1]:
                        sentence.append(acciones[np.argmax(res)])
                else:
                    sentence.append(acciones[np.argmax(res)])
        
            if len(sentence) > 5:
                    sentence = sentence[-5:]
            
            #Probabilidad visual
            imagen = prob_viz(res, acciones, imagen, colores_probabilidad_dinamico)
            
        cv2.rectangle(imagen, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(imagen, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
               
        #Mostrar en la pantalla
        cv2.imshow("OpenCV Feed", imagen)
    
        #Si se presiona q se sale del bucle
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break         
                                
    cap.release()
    cv2.destroyAllWindows()            

