import matplotlib.pyplot as pyplot
import cv2
im = cv2.imread("a.png")
print type(im)

fig = pyplot.figure(figsize=(1, 1))
ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
ax.imshow(im)
pyplot.show()
