import cv2
import numpy as np

ip_address = "192.168.1.55:8080"
video_url = f"http://{ip_address}/video"

def nothing(x):
    pass

cv2.namedWindow("Trackbar")
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

    l_h = cv2.getTrackbarPos("L-H", "Trackbar")
    l_s = cv2.getTrackbarPos("L-S", "Trackbar")
    l_v = cv2.getTrackbarPos("L-V", "Trackbar")
    u_h = cv2.getTrackbarPos("U-H", "Trackbar")
    u_s = cv2.getTrackbarPos("U-S", "Trackbar")
    u_v = cv2.getTrackbarPos("U-V", "Trackbar")

    lower_bound = np.array([l_h, l_s, l_v])
    upper_bound = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("Telefon Kamerası - IP Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
