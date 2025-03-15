import cv2
import numpy as np
ip_address = "192.168.1.55:8080"
video_url = f"http://{ip_address}/video"

cap = cv2.VideoCapture(video_url)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Bağlantı hatası!")
        break
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Renk aralıklarını belirle (HSV formatında)
    color_ranges = {
        "Red": [(np.array([0, 120, 70]), np.array([10, 255, 255])),
                (np.array([170, 120, 70]), np.array([180, 255, 255]))],  # Kırmızı (2 bölge)

        "Blue": [(np.array([90, 50, 50]), np.array([130, 255, 255]))],  # Mavi

        "Green": [(np.array([35, 50, 50]), np.array([85, 255, 255]))],  # Yeşil

        "Yellow": [(np.array([20, 100, 100]), np.array([30, 255, 255]))],  # Sarı
    }

    # Renklerin BGR karşılıkları
    color_bgr = {
        "Red": (0, 0, 255),      # Kırmızı
        "Blue": (255, 0, 0),     # Mavi
        "Green": (0, 255, 0),    # Yeşil
        "Yellow": (0, 255, 255)  # Sarı
    }

    for color, ranges in color_ranges.items():
        mask = np.zeros_like(hsv[:, :, 0])  # Boş bir maske oluştur
        for lower, upper in ranges:
            mask += cv2.inRange(hsv, lower, upper)  # Belirtilen aralıkları ekle
        
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:  # Gürültüleri filtrelemek için minimum alan sınırı
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr[color], 2)  # Algılanan renkte kutu çiz
                cv2.putText(frame, color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr[color], 2)  # Algılanan renkte yazı yaz


    cv2.imshow("Telefon Kamerası - IP Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
