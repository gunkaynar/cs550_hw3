from k_means import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('sample.jpg')
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')
plt.show()
plt.savefig('original.png')

k_values = [2, 3, 4, 5, 6]
draw(image, k_values, 2, 3)