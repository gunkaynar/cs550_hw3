from ahc import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('sample.jpg')

k_values = [2, 3, 4, 5, 6, 10]
draw2(image, k_values, 2, 3)

k_values = [2, 3, 4, 5, 6, 8, 10, 16, 32, 64]
errors = list()
cls = []
for k in k_values:
    cl = AHC(image, k_first=100, k_last=k)
    history = cl.train(error_margin=0.0001)
    errors.append(history['clustering_error'])
    cls.append(cl)

errors = []
k_values = [2, 3, 4, 5, 6, 8, 10, 16, 32, 64]
for i in range(len(k_values)):
    errors.append(cls[i].find_error())
plt.plot(k_values, errors)
plt.title('Clustering Error over k values')
plt.xlabel('k')
plt.ylabel('Clustering Error')
plt.savefig('clustering_error2.png')
clustered = cls[8].show()
plt.imshow(clustered)
plt.axis('off')
plt.title('Clustered Image with K=32')
plt.savefig('clustered2.png')
k_values = [2, 3, 4, 5, 6]
for i in range(len(k_values)):
    print('Vectors of K=%d' % k_values[i])
    print(np.around((cls[i].centroids * cls[i].std) + cls[i].mean, decimals=2))
