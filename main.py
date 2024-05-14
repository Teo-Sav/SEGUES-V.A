# Imporación de librerías ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import cv2
import numpy as np
from ultralytics import YOLO
import random
from tracker import Tracker
import time
from firebase import firebase

# Declaración de las variables ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

update_interval = 5  # Intervalo de actualización de la base de datos en segundos
update_buffer = {f'c{i}': True for i in range(1, 29)}  # Buffer para almacenar los cambios en la base de datos

# Especificamos la cámara que va a detectar o tomará como una ventana emergente
cap = cv2.VideoCapture(1)
ret, frame = cap.read()

# Se establece el modelo de Red neuronal que usaremos (En este caso Yolov8 nano)
model = YOLO("yolov8n.pt")
tracker = Tracker()

# Se establecen colores aleatorios para cada detección
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

# Es el valor de la confianza mínima como para que se pinte el bounding box
detection_threshold = 0.5

# Lista de clases que va a detectar la visión artificial (Los IDs corresponden en la poscición del array por ejemplo person es el id 0.0)
classes = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# Se establece la zona a detectar los espacios (Es la zona de estacionamiento) forma de Lineal empezando desde la izquierda

#F1
estC1 = np.array([[181, 28], [219, 28], [219, 104], [181, 104],])
estC2 = np.array([[223, 28], [260, 28], [260, 104], [223, 104],])
estC3 = np.array([[265, 28], [303, 28], [303, 104], [265, 104],])
estC4 = np.array([[306, 28], [344, 28], [344, 104], [306, 104],])
estC5 = np.array([[347, 28], [385, 28], [385, 104], [347, 104],])
estC6 = np.array([[390, 28], [427, 28], [427, 104], [390, 104],])
estC7 = np.array([[432, 28], [469, 28], [469, 104], [432, 104],])
estC8 = np.array([[473, 28], [512, 28], [512, 104], [473, 104],])
#F2
estC9 = np.array([[181, 114], [218, 114], [218, 188], [181, 188],])
estC10 = np.array([[223, 114], [260, 114], [260, 188], [223, 188],]) 
estC11 = np.array([[264, 114], [301, 114], [301, 188], [264, 188],])
estC12 = np.array([[306, 114], [343, 114], [343, 188], [306, 188],])
estC13 = np.array([[349, 114], [385, 114], [385, 188], [349, 188],])
estC14 = np.array([[390, 114], [426, 114], [426, 188], [390, 188],]) 
estC15 = np.array([[430, 114], [468, 114], [468, 188], [430, 188],])
estC16 = np.array([[472, 114], [510, 114], [510, 188], [472, 188],])
#F3
estC17 = np.array([[9, 280], [48, 280], [48, 355], [9, 355]])
estC18 = np.array([[54, 280], [90, 280], [90, 355], [54, 355]])
estC19 = np.array([[95, 280], [134, 280], [134, 355], [95, 355]])
estC20 = np.array([[138, 280], [174, 280], [174, 355], [138, 355]])
estC21 = np.array([[180, 280], [216, 280], [216, 355], [180, 355]])
estC22 = np.array([[222, 280], [259, 280], [259, 355], [222, 355]])
estC23 = np.array([[264, 280], [300, 280], [300, 355], [264, 355]])
estC24 = np.array([[305, 280], [342, 280], [342, 355], [305, 355]])
estC25 = np.array([[346, 280], [384, 280], [384, 355], [346, 355]])
estC26 = np.array([[388, 280], [425, 280], [425, 355], [388, 355]])
estC27 = np.array([[430, 280], [467, 280], [467, 355], [430, 355]])
estC28 = np.array([[472, 280], [510, 280], [510, 355], [472, 355]])



# Se establece y se configura la base de datos a la que enviaremos datos

firebase = firebase.FirebaseApplication("https://segues-71c59-default-rtdb.firebaseio.com", None)


# Variables que ayudan al conteo de fps
fps = 0
frame_count = 0
start_time = time.time()

# Desarrrollo del programa ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

while ret:
    frame_count += 1

    # Results es una lista que contiene todas las detecciones en una lista de un objeto (Cada objeto es una lista con cada "Results")
    results = model(frame,conf=0.5,verbose=False)

    for result in results:     

        # Se crea el array "Detections" que guardará las futuras detecciones
        detections = []

        for r in result.boxes.data.tolist():

            # x1 es ???, x2 es ???, y1 es ???, y2 es ???, score es el valor de confianza del objeto y class_id es el id de la clase precargada de la IA (Es decir los objetos a clasificar como por ejemplo "car")
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)

            # Este if compara el ID de la clase del objeto "Desencriptado" de results para compararlo con su referencia en el array "Classes" y porteriormente verificar que sea un carro con la clase "car"
            if classes[class_id] == 'car':
                if score > detection_threshold:
                    detections.append([x1, y1, x2, y2, score])
                
        if(detections!=[]):
            tracker.update(frame, detections)
            
            # Especifica un bucle para cada objeto detectado
            for track in tracker.tracks:
                # Se crea la variabla bbox que hace referencia a la esquina de cada rectangulo dibujado para cada objeto y su respectivo track_id
                bbox = track.bbox
                x1, y1, x2, y2 = bbox
                track_id = track.track_id
                
                # Se establece el centro de cada objeto
                xc, yc = int((x1 + x2)/2), int((y1 + y2)/2)

                # Se impriman los valores anteriormente mencionados (Se dibuja el rectangulo en cada objeto, al igual que si ID y su centro)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
                cv2.putText(frame, "ID:"+str(track_id),(int(x1),int(y1)-10),cv2.FONT_HERSHEY_PLAIN,1, (colors[track_id % len(colors)]), 2)
                cv2.circle(img=frame, center=(xc, yc), radius=5, color=(0,255,0), thickness=-1)

                
                # Se pregunta si los puntos se encuentran dentro de los estacionamientos para posteriormente enviar su estado a la base de datos
                print("Contenido de update_buffer antes del bucle:")
                print(update_buffer)

                for i in range(1, 29):
                    key = f'c{i}'
                    estC = eval(f'estC{i}')
                    estadoEst = cv2.pointPolygonTest(np.array(estC, np.int32), (xc, yc), False)

                    if estadoEst >= 0: #Esta libre?
                        #print("Estacionado")
                        #NO
                        if update_buffer[key] != False:
                            update_buffer[key] = False
  
                    else:
                        #print("Libre")
                        #Si
                        if update_buffer[key] != True:
                            update_buffer[key] = True
                            
                            
                            
                print("Contenido de update_buffer al final del bucle:")
                print(update_buffer)

    for estC in [estC1, estC2, estC3, estC4, estC5, estC6, estC7, estC8, estC9, estC10,
             estC11, estC12, estC13, estC14, estC15, estC16, estC17, estC18, estC19, estC20,
             estC21, estC22, estC23, estC24, estC25, estC26, estC27, estC28]:
        
        cv2.polylines(img=frame, pts=[estC], isClosed=True, color=(0,0,255), thickness=3)

    if time.time() - start_time >= update_interval:
            for key, value in update_buffer.items():
                firebase.put('/sectorA1', key, str(value))
            update_buffer.clear()
            start_time = time.time()
# Se cierra el programa ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Se muestra la cantidad fps que tenemos en el video/cámara/programa
    cv2.putText(frame, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("frame", frame)

    # Especificamos que se cierra la ventana de la cámara con el botón "esc"
    if cv2.waitKey(1) & 0xff == 27:
        break


    ret, frame = cap.read()

    # Cálculos de fps
    if time.time() - start_time >= 1:
        fps = frame_count
        frame_count = 0
        start_time = time.time()

cap.release()
cv2.destroyAllWindows()
