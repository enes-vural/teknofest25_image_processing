import cv2

image = cv2.imread('Sekiller.jpg')

B, G, R = cv2.split(image)

# Eşikleme ve bulanıklaştırma
ret, gray_scale = cv2.threshold(R, 100, 255, cv2.THRESH_BINARY)
blurred = cv2.GaussianBlur(gray_scale, (5, 5), 0)
cv2.imshow("GrayScale", blurred)

# Kenar tespiti
edges = cv2.Canny(blurred, 50, 150)
cv2.imshow("edges", edges)

# Kontur bulma
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    # Küçük gürültüleri filtrele
    area = cv2.contourArea(contour)
    if area < 1000:
        continue

    # Konturu yaklaşık bir çokgene dönüştür
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # 6 köşeli şekilleri tespit et (altıgen)
    if len(approx) == 3:
        cv2.drawContours(image, [approx], 0, (0, 255, 0), 3)
        x, y, w, h = cv2.boundingRect(approx)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        print(approx)

cv2.imshow("image", image)

# Pencerenin açık kalmasını sağla
cv2.waitKey(0)
cv2.destroyAllWindows()
