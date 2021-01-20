import numpy as np
import cv2
import requests
import os
import matplotlib.pyplot as plt

#Download data
face = "haarcascade.xml"
if not os.path.isfile(face):
	url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
	content = requests.get(url).content
	f = open(face, "wb")
	f.write(content)

eye = "eyecascade.xml"
if not os.path.isfile(eye):
	url_ = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml"
	content_ = requests.get(url_).content
	f = open(eye, "wb")
	f.write(content_)

image = "photo.jpg"

# if not os.path.isfile(image):
#   use this code if not have image
# 	url_ = "https://www.python.org.co/usuarios/yurs_ksf1/yurley.jpg"
# 	content_ = requests.get(url_).content
# 	f = open(image, "wb")
# 	f.write(content_)


#Classifiers
face_cascade = cv2.CascadeClassifier(face)
eye_cascade = cv2.CascadeClassifier(eye)

#Image we will predict
img = cv2.imread(image)
plt.imshow(img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Detect face
faces = face_cascade.detectMultiScale(gray, 1.3, 5)


for (x, y, w, h) in faces:
	#params = (image, position, width and height, color in RGB scale, and thickness)
	img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
	roi_gray = gray[y:y+h, x:x+w]
	roi_color = img[y:y+h, x:x+w]

	#Detect eyes per face
	eyes = eye_cascade.detectMultiScale(roi_gray)
	for (ex, ey, ew, eh) in eyes:
		cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)


#Show image
cv2.namedWindow("image", cv2.WINDOW_NORMAL)

while 1:
	cv2.imshow("image", img)
	cv2.waitKey(1)

cv2.destroyAllWindows()