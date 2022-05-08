import numpy as np
import time
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, data, K=2, normalize=True):
        self.shape = data.shape
        self.features = data.shape[-1]
        self.data = np.reshape(data, (-1, self.features))
        self.mean = np.mean(self.data)
        self.std = np.std(self.data)
        self.normalize = normalize
        if normalize:
            self.data = (self.data - self.mean) / self.std

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
        history['error'] = errors[1:]
        history['time'] = stop_time - start_time
        history['clustering_error'] = (errors[-1])
        return history

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


def draw(data, k_values, n, m):
    times = []
    errors = []
    cluster_vectors = list()
    images = list()

    for k in k_values:
        cl_ = KMeans(data, K=k)
        hist = cl_.train(error_margin=0.00001)
        times.append(hist['time'])
        errors.append(hist['clustering_error'])

        cluster_vectors.append(cl_.getCentroids())
        images.append(cl_.visualize())

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
        cl = KMeans(data, K=k)
        history = cl.train(error_margin=0.00001)
        errors.append(history['clustering_error'])
    plt.plot(k_values, errors)
    plt.title('Clustering Error vs K')
    plt.xlabel('K')
    plt.ylabel('Normalized Clustering Error')
    plt.savefig("k_means_error.png")
