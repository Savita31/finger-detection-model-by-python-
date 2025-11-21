"""
finger_counter_cv.py
Finger counting using classical computer vision (OpenCV).
No MediaPipe required — works on Python 3.13.

How it works (summary):
- Capture webcam frame, crop a fixed ROI where user should place their hand.
- Convert ROI to HSV, apply skin-like color threshold (simple heuristic).
- Morphological ops to clean mask, find largest contour (assumed hand).
- Compute convex hull & convexity defects; number of defects -> finger count estimate.
- Display result and optionally speak it.
"""

import cv2
import numpy as np
import time

# Optional TTS (install pyttsx3 if you want speech)
try:
    import pyttsx3
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

# ---------- Config ----------
CAMERA_INDEX = 0
ROI_TOP_LEFT = (100, 100)     # top-left corner of ROI box (x,y)
ROI_BOTTOM_RIGHT = (400, 400) # bottom-right corner of ROI box (x,y)
MIN_CONTOUR_AREA = 2000       # ignore small contours
SPEAK = False                 # set True to speak the count (requires pyttsx3)
SPEAK_ON_CHANGE = True
SMOOTHING_WINDOW = 5          # smooth over last N detected counts
# ----------------------------

if SPEAK and not TTS_AVAILABLE:
    print("pyttsx3 not found — install it if you want TTS: pip install pyttsx3")
    SPEAK = False

# Init TTS
if SPEAK:
    tts = pyttsx3.init()
    tts.setProperty("rate", 150)
else:
    tts = None

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError("Could not open camera. Change CAMERA_INDEX if needed.")

count_history = []

def smooth_count(new_count):
    """Keep a short history and return the modal (most frequent) value."""
    count_history.append(new_count)
    if len(count_history) > SMOOTHING_WINDOW:
        count_history.pop(0)
    # modal value
    counts, freqs = np.unique(count_history, return_counts=True)
    return counts[np.argmax(freqs)]

last_spoken = -1

print("Place your hand inside the square. Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror for natural interaction
    vis = frame.copy()

    x1, y1 = ROI_TOP_LEFT
    x2, y2 = ROI_BOTTOM_RIGHT
    roi = frame[y1:y2, x1:x2]

    # Convert to HSV for color thresholding (simple skin detection)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # ---- Skin color range (heuristic) ----
    # These ranges are a rough starting point and may need tuning for lighting/skin tones.
    lower = np.array([0, 15, 60], dtype=np.uint8)
    upper = np.array([25, 150, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Alternative: use YCrCb (sometimes more stable). Uncomment to try:
    # ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    # lower = np.array((0, 133, 77))
    # upper = np.array((255, 173, 127))
    # mask = cv2.inRange(ycrcb, lower, upper)

    # Morphology to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    # Find contours in mask
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finger_count = 0

    if contours:
        # pick largest contour by area (likely the hand)
        max_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(max_contour)

        if area > MIN_CONTOUR_AREA:
            # draw contour on ROI for debugging
            cv2.drawContours(roi, [max_contour], -1, (0, 255, 0), 2)

            # convex hull
            hull = cv2.convexHull(max_contour, returnPoints=False)
            if hull is not None and len(hull) > 3:
                defects = cv2.convexityDefects(max_contour, hull)
                if defects is not None:
                    defect_count = 0
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(max_contour[s][0])
                        end = tuple(max_contour[e][0])
                        far = tuple(max_contour[f][0])

                        # compute lengths of triangle sides
                        a = np.linalg.norm(np.array(start) - np.array(end))
                        b = np.linalg.norm(np.array(start) - np.array(far))
                        c = np.linalg.norm(np.array(end) - np.array(far))

                        # apply cosine theorem to find angle at far point
                        # angle in degrees
                        angle = np.degrees(np.arccos((b*b + c*c - a*a) / (2*b*c + 1e-6)))

                        # filter defects by angle and by distance (depth 'd')
                        if angle < 90 and d > 10000:
                            defect_count += 1
                            # draw defect points (optional)
                            cv2.circle(roi, far, 5, (0, 0, 255), -1)
                    # Usually number of fingers = defect_count + 1 (if at least 1 defect)
                    finger_count = min(5, defect_count + 1) if defect_count > 0 else 0

            # fallback heuristic: bounding rectangle height/width ratio can hint open palm
            # (optional improvement)
    else:
        finger_count = 0

    # smoothing & display
    smooth = smooth_count(finger_count)

    # Draw ROI rectangle and info on main frame
    cv2.rectangle(vis, ROI_TOP_LEFT, ROI_BOTTOM_RIGHT, (255, 0, 0), 2)
    cv2.putText(vis, f"Detected: {smooth}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # show mask for debugging
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    combined = np.hstack([vis, cv2.resize(mask_bgr, (vis.shape[1], vis.shape[0]))])
    cv2.imshow("Finger Counter (classical CV) - Left: frame, Right: mask", combined)

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
    if key == 27:  # ESC
        break
    elif key == ord('s'):
        # press 's' to save ROI debug image
        cv2.imwrite("roi_debug.png", roi)
        cv2.imwrite("mask_debug.png", mask)
        print("Saved roi_debug.png and mask_debug.png")

cap.release()
cv2.destroyAllWindows()
