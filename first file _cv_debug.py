# finger_counter_cv_debug.py
import cv2
import numpy as np
import time

try:
    import pyttsx3
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

# ---------- Config ----------
CAMERA_INDEX = 0
ROI_TOP_LEFT = (100, 100)
ROI_BOTTOM_RIGHT = (400, 400)
MIN_CONTOUR_AREA = 1000       # smaller to capture smaller hands
SPEAK = False
SPEAK_ON_CHANGE = True
SMOOTHING_WINDOW = 5
# Adaptive thresholds (good starting values; you can change these)
ANGLE_THRESH_DEG = 80        # angle at defect (deg) must be less than this
DEPTH_RATIO = 0.08           # defect depth in pixels must be > DEPTH_RATIO * bounding_box_height
# ----------------------------

if SPEAK and not TTS_AVAILABLE:
    print("pyttsx3 not found — install it if you want TTS: pip install pyttsx3")
    SPEAK = False

if SPEAK:
    tts = pyttsx3.init()
    tts.setProperty("rate", 150)
else:
    tts = None

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError("Could not open camera. Change CAMERA_INDEX if needed.")

count_history = []
last_spoken = -1

def smooth_count(new_count):
    count_history.append(new_count)
    if len(count_history) > SMOOTHING_WINDOW:
        count_history.pop(0)
    counts, freqs = np.unique(count_history, return_counts=True)
    return int(counts[np.argmax(freqs)])

print("Place your hand inside the square. Press ESC to quit. Press 's' to save ROI and mask images for inspection.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    vis = frame.copy()

    x1, y1 = ROI_TOP_LEFT
    x2, y2 = ROI_BOTTOM_RIGHT
    roi = frame[y1:y2, x1:x2]
    roi_h, roi_w = roi.shape[:2]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 15, 60], dtype=np.uint8)
    upper = np.array([25, 150, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finger_count = 0

    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(max_contour)

        if area > MIN_CONTOUR_AREA:
            cv2.drawContours(roi, [max_contour], -1, (0, 255, 0), 2)

            x, y, w, h = cv2.boundingRect(max_contour)
            # draw bounding rect
            cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 1)

            hull = cv2.convexHull(max_contour, returnPoints=False)
            if hull is not None and len(hull) > 3:
                defects = cv2.convexityDefects(max_contour, hull)
                defect_count = 0
                debug_defects = []
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(max_contour[s][0])
                        end = tuple(max_contour[e][0])
                        far = tuple(max_contour[f][0])

                        # compute triangle side lengths in pixels
                        a = np.linalg.norm(np.array(start) - np.array(end))
                        b = np.linalg.norm(np.array(start) - np.array(far))
                        c = np.linalg.norm(np.array(end) - np.array(far))

                        # cosine rule for angle at 'far' point
                        # clamp the cos value to [-1,1] to avoid numerical errors
                        cos_val = max(-1.0, min(1.0, (b*b + c*c - a*a) / (2*b*c + 1e-6)))
                        angle = np.degrees(np.arccos(cos_val))

                        # convert defect depth to approximate pixels
                        # OpenCV defect depth 'd' is returned scaled (usually 256x), so convert:
                        depth_pixels = d / 256.0

                        debug_defects.append((i, int(d), round(depth_pixels, 2), round(angle,1)))

                        # adaptive test: angle < ANGLE_THRESH and depth_pixels > DEPTH_RATIO * bbox_height
                        if angle < ANGLE_THRESH_DEG and depth_pixels > (DEPTH_RATIO * h):
                            defect_count += 1
                            # draw far point
                            cv2.circle(roi, far, 6, (0, 0, 255), -1)
                            cv2.circle(roi, start, 4, (0,255,255), -1)
                            cv2.circle(roi, end, 4, (0,255,255), -1)
                            cv2.line(roi, start, end, (0,255,255), 1)

                # finger_count heuristic: defects + 1 (bounded to 0..5)
                finger_count = defect_count + 1 if defect_count > 0 else 0

                # print debug info
                print(f"Area: {int(area)}, bbox_h: {h}, defects(total): {defects.shape[0] if defects is not None else 0}, "
                      f"defects_used: {defect_count}, debug_defects: {debug_defects}")

    else:
        finger_count = 0

    smooth = smooth_count(finger_count)

    cv2.rectangle(vis, ROI_TOP_LEFT, ROI_BOTTOM_RIGHT, (255, 0, 0), 2)
    cv2.putText(vis, f"Detected: {smooth}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    combined = np.hstack([cv2.resize(vis, (640, 480)), cv2.resize(mask_bgr, (640, 480))])
    cv2.imshow("Finger Counter - Left:frame Right:mask", combined)

    # optional TTS
    if SPEAK and tts is not None:
        if SPEAK_ON_CHANGE:
            if last_spoken != smooth:
                tts.say(str(int(smooth)))
                tts.runAndWait()
                last_spoken = smooth
        else:
            tts.say(str(int(smooth)))
            tts.runAndWait()
            last_spoken = smooth

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('s'):
        cv2.imwrite("roi_debug.png", roi)
        cv2.imwrite("mask_debug.png", mask)
        print("Saved roi_debug.png and mask_debug.png")

cap.release()
cv2.destroyAllWindows()
