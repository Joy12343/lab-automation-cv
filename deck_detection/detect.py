from ultralytics import YOLO
import cv2

model = YOLO("model/best.pt")

cap = cv2.VideoCapture("http://10.63.5.42:5000/video_feed2")  # 0 is usually the default webcam
#cap.set(cv2.CAP_PROP_FRAME_WIDTH)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT)

if not cap.isOpened():
    print("Error: Could not open webcam.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame from webcam.")
    else:
        results = model(frame)

        for result in results:
            rendered_img = result.plot()
            cv2.imshow("result", rendered_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
         break

cap.release()
cv2.destroyAllWindows()
