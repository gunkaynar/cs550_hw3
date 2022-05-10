from ahc import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('sample.jpg')

Ks = [2, 3, 4, 5, 6, 10]
draw2(image, Ks, 2, 3)

KList = [2, 3, 4, 5, 6, 8, 10, 16, 32, 64]
errors = list()
cls = []
for k in KList:
    cl = AHC(image, K_init=100, K_final=k)
    history = cl.train(error_margin=0.0001, linkage='complete', verbose=False)
    errors.append(history['clustering_error'])
    cls.append(cl)

errors = []
KList = [2, 3, 4, 5, 6, 8, 10, 16, 32, 64]
for i in range(len(KList)):
    errors.append(cls[i].calcError())
plt.plot(KList, errors)
plt.title('Clustering Error vs K')
plt.xlabel('K')
plt.ylabel('Normalized Clustering Error')
plt.savefig('clustering_error2.png')
clustered = cls[8].visualize()
plt.imshow(clustered)
plt.axis('off')
plt.title('Clustered Image with K=32')
plt.savefig('clustered2.png')
KList = [2, 3, 4, 5, 6]
for i in range(len(KList)):
    print('Vectors of K=%d' % KList[i])
    print(np.around((cls[i].centroids * cls[i].std) + cls[i].mean, decimals=2))
