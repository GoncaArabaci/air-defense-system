import cv2
import numpy as np


def detect_shape(image):
    # Görüntüyü gri tonlamaya çevir
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gürültüyü azalt
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold değerini yükselt ve hassasiyeti artır
    _, threshold = cv2.threshold(blurred, 150, 254, cv2.THRESH_BINARY_INV)

    # Morfolojik işlemleri azalt
    kernel = np.ones((3, 3), np.uint8)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations=1)
    threshold = cv2.dilate(threshold, kernel, iterations=1)

    # Konturları bul
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = image.copy()

    for contour in contours:
        area = cv2.contourArea(contour)

        # Alan kontrolünü gevşet
        if area > 1000 and area < 50000:  # Minimum alanı düşürdük
            cv2.drawContours(result, [contour], 0, (0, 255, 0), 2)

            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

            vertices = len(approx)

            # Şeklin merkezini bul
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = 50, 50

            # Basit şekil tespiti
            if vertices == 3:
                cv2.putText(result, "Ucgen", (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            elif vertices == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                if 0.85 <= aspect_ratio <= 1.15:
                    cv2.putText(result, "Kare", (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(result, "Dikdortgen", (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:  # vertices > 4 ise daire olarak kabul et
                cv2.putText(result, "Daire", (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return result, threshold


# Ana program
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result, threshold = detect_shape(frame)

        # Sonuçları göster
        cv2.imshow("Sekil Tespiti", result)
        cv2.imshow("Threshold", threshold)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()