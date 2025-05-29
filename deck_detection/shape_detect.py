import cv2

cap = cv2.VideoCapture("http://10.63.5.42:5000/video_feed2")  # 0 is usually the default webcam
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def crop_frame(frame):
    h, w, _ = frame.shape
    x_1 = int(w * 0.25)
    x_2 = int(w * 0.95)
    y_2 = int(h * 0.90)
    return frame[:y_2, x_1:x_2]

def camera_preprocess(frame):
    frame = crop_frame(frame)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
    enhanced = clahe.apply(gray)

    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    return blurred

def is_good_rectangle(approx, min_area=10, aspect_range=(1.0, 2.5)):
    if len(approx) != 4:
        return False

    if not cv2.isContourConvex(approx):
        return False

    return True
    '''
    area = cv2.contourArea(approx)
    if area < min_area:
        return False

    # Get bounding box and check aspect ratio
    x, y, w, h = cv2.boundingRect(approx)
    aspect_ratio = w / float(h)
    if not (aspect_range[0] <= aspect_ratio <= aspect_range[1]):
        return False

    # Check angles between edges (should be ~90 degrees)
    def angle(pt1, pt2, pt3):
        a = np.array(pt1) - np.array(pt2)
        b = np.array(pt3) - np.array(pt2)
        cosine_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        return np.degrees(np.arccos(cosine_angle))

    angles = []
    for i in range(4):
        p1 = approx[i][0]
        p2 = approx[(i + 1) % 4][0]
        p3 = approx[(i + 2) % 4][0]
        ang = angle(p1, p2, p3)
        angles.append(ang)

    return all(80 <= ang <= 100 for ang in angles)
    '''


if not cap.isOpened():
    print("Error: Could not open webcam.")

while True:
    ret, frame = cap.read()

    if ret:
        new_frame = camera_preprocess(frame)
        frame_cropped = crop_frame(frame)
    
        # Edge detection
        edges = cv2.Canny(new_frame, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        deck_contour = None
        max_area = 0

        for cnt in contours:

            cv2.drawContours(frame_cropped, [cnt], -1, (0, 0, 255), 3)

            ep = 0.005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, ep, True)
            area = cv2.contourArea(approx)
            
            if 25 <= len(approx) <= 45 and 60 < area < 100:
                cv2.drawContours(frame_cropped, [approx], -1, (0, 255, 0), 3)

            if is_good_rectangle(approx):
                cv2.drawContours(frame_cropped, [approx], -1, (255, 0, 0), 3)

        cv2.imshow('cv2_result', frame_cropped)
        # cv2.imshow('test', blurred)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
