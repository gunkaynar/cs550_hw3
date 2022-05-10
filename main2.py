from k_means import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('sample.jpg')

k_values = [2, 3, 4, 5, 6, 8, 10, 16, 32, 64]
find_best_k(image, k_values)

best_k_value = 15
clustered = KMeans(image, k=best_k_value)
history = clustered.train(error_margin=1e-5)

print('Clustering Error:', history['clustering_error'])

clustered_image = clustered.show()
plt.imshow(clustered_image)
plt.axis('off')
plt.title(f'Clustered Image with k={best_k_value}')
plt.show()
plt.savefig('clustered_image.png')
