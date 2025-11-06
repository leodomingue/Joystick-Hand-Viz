from ultralytics import YOLO
import mediapipe as mp
import cv2

#Modelos
model = YOLO("best.pt")  # joystick
mp_hands = mp.solutions.hands.Hands(min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results_joystick = model.predict(frame, conf=0.5, verbose=False)
    boxes_joystick = results_joystick[0].boxes.xyxy.cpu().numpy()
    classes_joystick = results_joystick[0].boxes.cls.cpu().numpy()

    #Nombres de las clases
    class_names = model.names

    #Detección manos
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hands = mp_hands.process(rgb)

    if hands.multi_hand_landmarks:
        for hand_landmarks in hands.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
            )

    for i, box in enumerate(boxes_joystick):
        x1, y1, x2, y2 = map(int, box)
        class_id = int(classes_joystick[i])
        class_name = class_names[class_id]
        confidence = results_joystick[0].boxes.conf[i].cpu().numpy()
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name}: {confidence:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 2)
        cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.imshow("Deteccion", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()