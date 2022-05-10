import numpy as np
import time
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, data, k=2):
        self.shape = data.shape
        self.features = data.shape[-1]
        self.data = np.reshape(data, (-1, self.features))
        self.mean = np.mean(self.data)
        self.std = np.std(self.data)
        self.data = (self.data - self.mean) / self.std

        self.k = k
        ind = np.random.permutation(self.data.shape[0])
        self.centroids = np.array(self.data[ind][:k, :], copy=True)

        self.whose = 0

    def assign(self):
        distances = np.sqrt(np.sum(((self.data - self.centroids[:, np.newaxis]) ** 2), axis=-1))
        self.whose = np.argmin(distances, axis=0)

    def maximization(self):
        error = 0
        for i in range(self.k):
            self.centroids[i, :] = np.mean(self.data[self.whose == i], axis=0)
            error += np.sum(np.sqrt(np.sum((self.data[self.whose == i] - self.centroids[i, :]) ** 2, axis=-1)))
        error /= self.data.shape[0]
        return error

    def train(self, error_margin=0.001):
        history = dict()
        errors = [np.inf]
        start_time = time.time()
        while True:
            self.assign()
            error = self.maximization()
            print(error)
            if np.abs(error - errors[-1]) <= error_margin:
                break
            errors.append(error)
        stop_time = time.time()
        print('Running time = ' % (stop_time - start_time))
        history['error'] = errors[1:]
        history['time'] = stop_time - start_time
        history['clustering_error'] = (errors[-1])
        return history

    def show(self):
        img = np.ones(self.data.shape)
        for i in range(self.k):
            img[self.whose == i] = self.centroids[i]

        img *= self.std
        img += self.mean
        img = np.round(img).astype(int)
        img = img.reshape(self.shape)
        return img

    def find_centroids(self):
        return self.centroids * self.std + self.mean


def draw(data, k_values, n, m):
    times = []
    errors = []
    cluster_vectors = list()
    images = list()

    for k in k_values:
        cl_ = KMeans(data, k=k)
        hist = cl_.train(error_margin=0.00001)
        times.append(hist['time'])
        errors.append(hist['clustering_error'])

        cluster_vectors.append(cl_.find_centroids())
        images.append(cl_.show())

    plt.figure(figsize=(m * 4, n * 4))
    for i in range(len(k_values)):
        plt.subplot(n, m, i + 1)
        plt.imshow(images[i])
        plt.title('K={}'.format((k_values[i])))
        plt.axis('off')
    plt.savefig("k_means_values.png")




def find_best_k(data, k_values):
    errors = list()
    for k in k_values:
        cl = KMeans(data, k=k)
        history = cl.train(error_margin=0.00001)
        errors.append(history['clustering_error'])
    plt.plot(k_values, errors)
    plt.title('Clustering Error vs K')
    plt.xlabel('K')
    plt.ylabel('Clustering Error')
    plt.savefig("k_means_error.png")
