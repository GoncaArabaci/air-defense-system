import cv2
import numpy as np

ip_address = "10.125.68.131:8080"
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

# Trackbar ayarlarını kaydedebileceğimiz dosya
settings_file = "settings.txt"

# Trackbar ayarlarını dosyaya kaydetme
def save_settings():
    l_h = cv2.getTrackbarPos("L-H", "Trackbar")
    l_s = cv2.getTrackbarPos("L-S", "Trackbar")
    l_v = cv2.getTrackbarPos("L-V", "Trackbar")
    u_h = cv2.getTrackbarPos("U-H", "Trackbar")
    u_s = cv2.getTrackbarPos("U-S", "Trackbar")
    u_v = cv2.getTrackbarPos("U-V", "Trackbar")
    
    with open(settings_file, 'w') as f:
        f.write(f"{l_h} {l_s} {l_v} {u_h} {u_s} {u_v}\n")

# Trackbar ayarlarını dosyadan yükleme
def load_settings():
    try:
        with open(settings_file, 'r') as f:
            settings = f.readline().strip().split()
            return list(map(int, settings))
    except FileNotFoundError:
        return [0,68, 129, 119, 226, 255]  # Varsayılan değerler
        # return [0,82, 120, 126, 255, 255]
        
# Ayarları yükle
settings = load_settings()

# Trackbar'lara değerleri yükle
cv2.setTrackbarPos("L-H", "Trackbar", settings[0])
cv2.setTrackbarPos("L-S", "Trackbar", settings[1])
cv2.setTrackbarPos("L-V", "Trackbar", settings[2])
cv2.setTrackbarPos("U-H", "Trackbar", settings[3])
cv2.setTrackbarPos("U-S", "Trackbar", settings[4])
cv2.setTrackbarPos("U-V", "Trackbar", settings[5])

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
    frame_resized = cv2.resize(frame, (640, 480))  # Resize to 640x480 or any size you prefer
    mask_resized = cv2.resize(mask, (640, 480))  # Resize to 640x480 or any size you prefer
    cv2.imshow("Telefon Kamerası - IP Webcam", frame_resized)
    cv2.imshow("Maske", mask_resized)

    # Kaydetme işlemi
    save_settings()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
