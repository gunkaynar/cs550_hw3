import numpy as np
import time
from scipy.spatial import distance
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, data, K=2, normalize=True):
        self.shape = data.shape
        self.features = data.shape[-1]

        self.data = np.reshape(data, (-1, self.features))

        self.K = K

        ind = np.random.permutation(self.data.shape[0])
        self.centroids = np.array(self.data[ind][:K, :], copy=True)

        self.membership = 0

    def expectation(self):
        distances = np.sqrt(np.sum(((self.data - self.centroids[:, np.newaxis]) ** 2), axis=-1))
        self.membership = np.argmin(distances, axis=0)

    def maximization(self):
        error = 0
        for i in range(self.K):
            self.centroids[i, :] = np.mean(self.data[self.membership == i], axis=0)
            error += np.sum(np.sqrt(np.sum((self.data[self.membership == i] - self.centroids[i, :]) ** 2, axis=-1)))
        error /= self.data.shape[0]
        return error

    def train(self, error_margin=0.001, verbose=False):
        history = dict()
        errors = [np.inf]
        start_time = time.time()
        while True:
            self.expectation()
            error = self.maximization()
            if verbose:
                print(error)
            if np.abs(error - errors[-1]) <= error_margin:
                break
            errors.append(error)
        stop_time = time.time()
        if verbose:
            print('Elapsed time: %.5fs' % (stop_time - start_time))

        return self.centroids, self.membership

    def visualize(self):
        img = np.ones(self.data.shape)
        for i in range(self.K):
            img[self.membership == i] = self.centroids[i]
        if self.normalize:
            img *= self.std
            img += self.mean
        img = np.round(img).astype(int)
        img = img.reshape(self.shape)
        return img

    def getCentroids(self):
        return self.centroids * self.std + self.mean


class AHC:
    def __init__(self, data, K_final=2, K_init=100, normalize=True):
        self.shape = data.shape
        self.features = data.shape[-1]

        self.data = np.reshape(data, (-1, self.features))

        self.mean = np.mean(self.data)
        self.std = np.std(self.data)
        self.normalize = normalize
        if normalize:
            self.data = (self.data - self.mean) / self.std

        self.K_final = K_final
        self.K_init = K_init

        self.centroids = 0
        self.membership = 0

        self.k = 0
        self.klist = 0
        self.distanceMatrix = 0

    def calcDistance(self, c1, c2):
        distances = distance.cdist(c1, c2, 'euclidean')
        return np.max(distances)


    def calcDistanceMatrix(self, verbose):
        distances = np.zeros((self.k, self.k))
        for i in range(self.k):
            for j in range(self.k):
                if verbose:
                    print('Distance (%d,%d)' % (i, j))
                if i != j and distances[i, j] == 0 and distances[j, i] == 0:
                    dist = self.calcDistance(self.data[self.membership == j],
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

        self.membership[self.membership == self.klist[ind2]] = self.klist[ind1]
        self.distanceMatrix[ind1, :] = np.maximum(self.distanceMatrix[ind2, :], self.distanceMatrix[ind1, :])
        self.distanceMatrix[:, ind1] = np.maximum(self.distanceMatrix[:, ind2], self.distanceMatrix[:, ind1])
        self.distanceMatrix[ind1, ind1] = 0
        self.distanceMatrix = np.delete(self.distanceMatrix, ind2, axis=0)
        self.distanceMatrix = np.delete(self.distanceMatrix, ind2, axis=1)

        self.k -= 1
        removed = self.klist[ind2]
        self.klist = self.klist[self.klist != removed]

        return removed

    def setCentroids(self):
        self.centroids = np.ones((self.K_final, self.features))
        for x, i in enumerate(self.klist):
            self.centroids[x, :] = np.mean(self.data[self.membership == i], axis=0)

    def calcError(self):
        error = 0
        for i, cl in enumerate(self.klist):
            error += np.sum(np.sqrt(np.sum((self.data[self.membership == cl] - self.centroids[i, :]) ** 2, axis=-1)))
        error /= self.data.shape[0]
        return error

    def train(self, error_margin=0.001, verbose=False):
        history = dict()

        start_time = time.time()
        kmc = KMeans(self.data, K=self.K_init, normalize=False)
        self.centroids, self.membership = kmc.train(error_margin, verbose)
        self.k = self.K_init
        self.klist = np.unique(self.membership)

        self.calcDistanceMatrix(verbose)

        while self.k > self.K_final:
            rem = self.iterate()
            if verbose:
                print('%d merged, %d clusters remain' % (rem, self.k))

        stop_time = time.time()

        self.setCentroids()

        history['time'] = (stop_time - start_time)
        history['clustering_error'] = self.calcError()
        history['centroids'] = self.getCentroids()

        return history

    def visualize(self):
        img = np.ones(self.data.shape)

        for x, i in enumerate(self.klist):
            img[self.membership == i] = self.centroids[x]
        if self.normalize:
            img *= self.std
            img += self.mean
        img = np.round(img).astype(int)
        img = img.reshape(self.shape)
        return img

    def getCentroids(self):
        return self.centroids * self.std + self.mean


def draw2(data, KList, n, m):
    times = []
    errors = []
    cluster_vectors = list()
    images = list()

    for k in KList:
        cl = AHC(data, K_init=100, K_final=k)
        history = cl.train(error_margin=0.00, verbose=False)

        times.append(history['time'])
        errors.append(history['clustering_error'])
        cluster_vectors.append('centroids')

        images.append(cl.visualize())

    plt.figure(figsize=(m * 4, n * 4))
    for i in range(len(KList)):
        plt.subplot(n, m, i + 1)
        plt.imshow(images[i])
        plt.title('K={}'.format((KList[i])))
        plt.axis('off')
    plt.savefig("draw2.png")
