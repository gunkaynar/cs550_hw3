import numpy as np
import time
from scipy.spatial import distance
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, data, k=2):
        self.shape = data.shape
        self.features = data.shape[-1]

        self.data = np.reshape(data, (-1, self.features))

        self.k = k

        ind = np.random.permutation(self.data.shape[0])
        self.centroids = np.array(self.data[ind][:k, :], copy=True)

        self.membership = 0

    def assign(self):
        distances = np.sqrt(np.sum(((self.data - self.centroids[:, np.newaxis]) ** 2), axis=-1))
        self.membership = np.argmin(distances, axis=0)

    def maximization(self):
        error = 0
        for i in range(self.k):
            self.centroids[i, :] = np.mean(self.data[self.membership == i], axis=0)
            error += np.sum(np.sqrt(np.sum((self.data[self.membership == i] - self.centroids[i, :]) ** 2, axis=-1)))
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

        return self.centroids, self.membership

    def show(self):
        img = np.ones(self.data.shape)
        for i in range(self.k):
            img[self.membership == i] = self.centroids[i]

        img *= self.std
        img += self.mean
        img = np.round(img).astype(int)
        img = img.reshape(self.shape)
        return img

    def find_centroids(self):
        return self.centroids * self.std + self.mean


class AHC:
    def __init__(self, data, k_last=2, k_first=100):
        self.shape = data.shape
        self.features = data.shape[-1]

        self.data = np.reshape(data, (-1, self.features))

        self.mean = np.mean(self.data)
        self.std = np.std(self.data)
        self.data = (self.data - self.mean) / self.std

        self.k_last = k_last
        self.k_first = k_first

        self.centroids = 0
        self.membership = 0

        self.k = 0
        self.k_values = 0
        self.distanceMatrix = 0

    def find_distance(self, c1, c2):
        distances = distance.cdist(c1, c2, 'euclidean')
        return np.max(distances)

    def distance_matrix(self):
        distances = np.zeros((self.k, self.k))
        for i in range(self.k):
            for j in range(self.k):
                print('Distance (%d,%d)' % (i, j))
                if i != j and distances[i, j] == 0 and distances[j, i] == 0:
                    dist = self.find_distance(self.data[self.membership == j],
                                              self.data[self.membership == i])
                    distances[i, j] = dist
                    distances[j, i] = dist
                elif distances[i, j] != 0:
                    distances[j, i] = distances[i, j]
                elif distances[j, i] != 0:
                    distances[i, j] = distances[j, i]
        self.distanceMatrix = distances

    def iterate(self):
        indices = np.argwhere(self.distanceMatrix == np.amin(self.distanceMatrix[self.distanceMatrix != 0]))[0]
        ind1 = indices[0]
        ind2 = indices[1]

        self.membership[self.membership == self.k_values[ind2]] = self.k_values[ind1]
        self.distanceMatrix[ind1, :] = np.maximum(self.distanceMatrix[ind2, :], self.distanceMatrix[ind1, :])
        self.distanceMatrix[:, ind1] = np.maximum(self.distanceMatrix[:, ind2], self.distanceMatrix[:, ind1])
        self.distanceMatrix[ind1, ind1] = 0
        self.distanceMatrix = np.delete(self.distanceMatrix, ind2, axis=0)
        self.distanceMatrix = np.delete(self.distanceMatrix, ind2, axis=1)

        self.k -= 1
        removed = self.k_values[ind2]
        self.k_values = self.k_values[self.k_values != removed]

        return removed

    def assign(self):
        self.centroids = np.ones((self.k_last, self.features))
        for x, i in enumerate(self.k_values):
            self.centroids[x, :] = np.mean(self.data[self.membership == i], axis=0)

    def find_error(self):
        error = 0
        for i, cl in enumerate(self.k_values):
            error += np.sum(np.sqrt(np.sum((self.data[self.membership == cl] - self.centroids[i, :]) ** 2, axis=-1)))
        error /= self.data.shape[0]
        return error

    def train(self, error_margin=0.001):
        history = dict()

        start_time = time.time()
        kmc = KMeans(self.data, k=self.k_first)
        self.centroids, self.membership = kmc.train(error_margin)
        self.k = self.k_first
        self.k_values = np.unique(self.membership)

        self.distance_matrix()

        while self.k > self.k_last:
            rem = self.iterate()

        stop_time = time.time()

        self.assign()

        history['time'] = (stop_time - start_time)
        history['clustering_error'] = self.find_error()
        history['centroids'] = self.find_centroids()

        return history

    def show(self):
        img = np.ones(self.data.shape)

        for x, i in enumerate(self.k_values):
            img[self.membership == i] = self.centroids[x]
        img *= self.std
        img += self.mean
        img = np.round(img).astype(int)
        img = img.reshape(self.shape)
        return img

    def find_centroids(self):
        return self.centroids * self.std + self.mean


def draw2(data, k_values, n, m):
    times = []
    errors = []
    cluster_vectors = list()
    images = list()

    for k in k_values:
        cl = AHC(data, k_first=100, k_last=k)
        history = cl.train(error_margin=0.00)

        times.append(history['time'])
        errors.append(history['clustering_error'])
        cluster_vectors.append('centroids')

        images.append(cl.show())

    plt.figure(figsize=(m * 4, n * 4))
    for i in range(len(k_values)):
        plt.subplot(n, m, i + 1)
        plt.imshow(images[i])
        plt.title('K={}'.format((k_values[i])))
        plt.axis('off')
    plt.savefig("draw2.png")
