import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('')

# color imgs
color = ('b','g','r')
for i, col in enumerate(color):
  histr = cv2.calcHist([img],[i],None,[256],[0, 256])
  plt.plot(histr, color = col)
  plt.xlim([0, 256])
plt.show()

# gray scale imgs
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# set histogram bin
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
hist=hist.reshape((256))

plt.bar(range(0,256), hist)
plt.show()
plt.plot(range(0,256), hist)
plt.show()
