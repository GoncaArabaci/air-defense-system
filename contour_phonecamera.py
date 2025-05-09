import cv2
import numpy as np

ip_address = "10.103.134.218:8080"
video_url = f"http://{ip_address}/video"

def nothing(x):
    pass

# Trackbar penceresini oluştur ve boyutunu ayarla
cv2.namedWindow("Trackbar", cv2.WINDOW_NORMAL)

# Trackbar'ları oluştur
cv2.createTrackbar("L-H", "Trackbar", 0, 179, nothing)  
cv2.createTrackbar("L-S", "Trackbar", 0, 255, nothing)  
cv2.createTrackbar("L-V", "Trackbar", 0, 255, nothing)  
cv2.createTrackbar("U-H", "Trackbar", 179, 179, nothing)  
cv2.createTrackbar("U-S", "Trackbar", 255, 255, nothing)  
cv2.createTrackbar("U-V", "Trackbar", 255, 255, nothing)  

cap = cv2.VideoCapture(video_url)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Bağlantı hatası!")
        break
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Trackbar değerlerini al
    l_h = cv2.getTrackbarPos("L-H", "Trackbar")
    l_s = cv2.getTrackbarPos("L-S", "Trackbar")
    l_v = cv2.getTrackbarPos("L-V", "Trackbar")
    u_h = cv2.getTrackbarPos("U-H", "Trackbar")
    u_s = cv2.getTrackbarPos("U-S", "Trackbar")
    u_v = cv2.getTrackbarPos("U-V", "Trackbar")

    # Trackbar değerleri ile maske oluştur
    lower_bound = np.array([l_h, l_s, l_v], dtype=np.uint8)
    upper_bound = np.array([u_h, u_s, u_v], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    mask = cv2.medianBlur(mask, 5)

    # Kontur bulma
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 300:  # Küçük nesneleri filtrele
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            x, y, w, h = cv2.boundingRect(approx)

            # Şekil tespiti
            shape = ""
            if len(approx) == 3:
                shape = "Triangle"
            elif len(approx) == 4:
                aspect_ratio = float(w) / h
                if 0.85 <= aspect_ratio <= 1.15:
                    shape = "Square"
                else:
                    shape = "Rectangle"
            elif len(approx) > 6:
                shape = "Circle"

            # Çizimleri ekle
            cv2.drawContours(frame, [approx], -1, (255, 255, 0), 3)  # Mavi çizim
            cv2.putText(frame, shape, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)  # Metin ekle

    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("Telefon Kamerası - IP Webcam", frame)
    cv2.imshow("Maske", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
