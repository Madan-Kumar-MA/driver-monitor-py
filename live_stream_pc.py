import cv2
import mediapipe as mp

# Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Webcam capture
cap = cv2.VideoCapture(0)

def is_distracted(landmarks, img_w, img_h):
    """
    Simple distraction detection:
    - If eyes closed OR head turned too much = distracted
    """
    left_eye = landmarks[159]  # upper eyelid
    right_eye = landmarks[386]  # upper eyelid

    # Convert to pixel coords
    lx, ly = int(left_eye.x * img_w), int(left_eye.y * img_h)
    rx, ry = int(right_eye.x * img_w), int(right_eye.y * img_h)

    # Eye distance (simple blink detection)
    eye_gap = abs(ly - ry)

    if eye_gap < 5:  # eyes closed
        return True

    # Check if face is tilted left/right too much
    nose = landmarks[1]
    nx = int(nose.x * img_w)

    if nx < img_w * 0.3 or nx > img_w * 0.7:
        return True

    return False


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert to RGB for mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    distracted = False
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1)
            )
            distracted = is_distracted(face_landmarks.landmark, w, h)

    # Show distraction status
    status = "Driver Distracted!" if distracted else "Driver Focused"
    color = (0, 0, 255) if distracted else (0, 255, 0)
    cv2.putText(frame, status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # Show video
    cv2.imshow("Driver Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
