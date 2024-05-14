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
update_buffer = {f'c{i}': True for i in range(1, 29)} # Buffer para almacenar los cambios en la base de datos

# Especificamos la cámara que va a detectar o tomará como una ventana emergente
cap = cv2.VideoCapture(0)
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

regions = {
    'c1': [[387, 306], [423, 306], [420, 376], [385, 376]],
    'c2': [[346, 306], [380, 306], [380, 376], [346, 376]],
    'c3': [[306, 306], [341, 306], [341, 376], [306, 376]],
    'c4': [[264, 306], [301, 306], [301, 376], [264, 376]],
    'c5': [[223, 306], [260, 306], [260, 376], [223, 376]],
    'c6': [[183, 306], [219, 306], [219, 376], [183, 376]],
    'c7': [[141, 306], [177, 306], [177, 376], [141, 376]],
    'c8': [[98, 306], [136, 306], [136, 376], [100, 376]],
    'c9': [[389, 222], [426, 222], [426, 295], [389, 295]],
    'c10': [[349, 222], [384, 222], [384, 295], [346, 295]],
    'c11': [[307, 222], [343, 222], [343, 295], [307, 295]],
    'c12': [[265, 222], [303, 222], [303, 295], [265, 295]],
    'c13': [[224, 222], [260, 222], [260, 295], [224, 295]],
    'c14': [[182, 222], [219, 222], [219, 295], [182, 295]],
    'c15': [[139, 222], [176, 222], [176, 295], [139, 295]],
    'c16': [[97, 222], [134, 222], [134, 295], [97, 295]],
    'c17': [[566, 57], [605, 57], [599, 131], [562, 131]],
    'c18': [[522, 57], [561, 57], [556, 131], [517, 131]],
    'c19': [[481, 57], [518, 57], [514, 131], [476, 131]],
    'c20': [[438, 57], [475, 57], [472, 131], [436, 131]],
    'c21': [[393, 56], [432, 56], [429, 131], [393, 131]],
    'c22': [[353, 54], [389, 54], [389, 131], [350, 131]],
    'c23': [[309, 54], [347, 54], [347, 131], [309, 131]],
    'c24': [[265, 51], [304, 52], [304, 131], [265, 131]],
    'c25': [[223, 51], [260, 53], [260, 131], [223, 131]],
    'c26': [[179, 51], [217, 51], [217, 131], [180, 131]],
    'c27': [[136, 51], [174, 51], [176, 131], [138, 131]],
    'c28': [[92, 51], [130, 51], [130, 131], [94, 131]],
}


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
                    space = regions[key]
                    estadoEst = cv2.pointPolygonTest(np.array(space, np.int32), (xc, yc), False)

                    if estadoEst >= 0: #Esta libre?
                        print("Estacionado")
                        #NO
                        if update_buffer.get(key, False) != False:
                            update_buffer[key] = False
  
                    else:
                        print("Libre")
                        #Si
                        if update_buffer.get(key, True) != True:
                            update_buffer[key] = True  
                            
                            
                print("Contenido de update_buffer al final del bucle:")
                print(update_buffer)

    # Imprime las celdas
    for region_points in regions.values():
        cv2.polylines(img=frame, pts=[np.array(region_points, np.int32)], isClosed=True, color=(0, 0, 255), thickness=3)

    #Si llegan las actualizacones de los datos####

    if time.time() - start_time >= update_interval:
            for key, value in update_buffer.items():
                firebase.put('/sectorA1', key, value)
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
