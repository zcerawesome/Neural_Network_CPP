import cv2
import csv

img = cv2.imread('Drawing/Drawing.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
data = img.astype('float32') / 255.0
data = data.flatten()

with open("image.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(data)