import cv2
import numpy as np

def detect_shape(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    if len(approx) == 3:
        return "Triangle"
    elif len(approx) == 4:
        return "Rectangle"
    else:
        return "Circle"

# HSV renk aralıkları
color_ranges = {
    "Red": [(0, 100, 100), (10, 255, 255)],
    "Red2": [(160, 100, 100), (180, 255, 255)],
    "Green": [(40, 50, 50), (80, 255, 255)],
    "Blue": [(100, 100, 50), (140, 255, 255)]
}

# IP Webcam URL (kendi telefon IP adresinle değiştir)
url = "http://192.168.1.42:8080/video"
cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized = cv2.resize(frame, (640, 640))
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    for color_name in ["Red", "Green", "Blue"]:
        if color_name == "Red":
            mask1 = cv2.inRange(hsv, np.array(color_ranges["Red"][0]), np.array(color_ranges["Red"][1]))
            mask2 = cv2.inRange(hsv, np.array(color_ranges["Red2"][0]), np.array(color_ranges["Red2"][1]))
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv, np.array(color_ranges[color_name][0]), np.array(color_ranges[color_name][1]))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 80:
                shape = detect_shape(cnt)
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(resized, (x, y), (x + w, y + h), (0, 0, 0), 2)
                cv2.putText(resized, f"{color_name} {shape}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.imshow("IP Kamera Şekil ve Renk Tespiti", resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
