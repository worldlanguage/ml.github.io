""" Import the basic requirements package """ 
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

""" Dataset export function """

def read_csv():

    X = np.random.randint(low=1, high=100, size=(800, 3))

    print(X.shape)  # Print feature set and label set length

    return X

""" Euclidean distance function """

def distance(x, y):

    y = y.values

    x_C_dis = np.sqrt(np.sum(np.power(x - y, 2), axis=1))

    return x_C_dis


def cacl_k_vectors(data, k):

    C_set = []
    for c in range(k):
        C_set.append(data[data['C'] == c][['x', 'y', 'z']].mean().values)

    k_vectors = np.vstack(tuple(C_set))

    return k_vectors

# create model
def KMeans(k=10):

    kmeans_params = {
        'k': k,                      # Set the base of the kmeans logarithm
    }
    return kmeans_params

# fit model
def fit(kmeans_params, X):

    data = pd.DataFrame(X, columns=['x', 'y', 'z'])

    k_index = data.sample(n=kmeans_params['k'], random_state=1).index
    k_vector = data[['x', 'y', 'z']].iloc[k_index].values

    ax = Axes3D(plt.figure(figsize=(9, 7.5)))

    plt.ion()

    for i in range(100):

        data['C'] = data[['x', 'y', 'z']].apply(lambda row: np.argmin(distance(k_vector, row)), axis=1)

        k_vector = cacl_k_vectors(data, kmeans_params['k'])

        ax.cla()

        for c in range(kmeans_params['k']):
            tmp = data[data['C'] == c]
            ax.scatter(tmp['x'], tmp['y'], tmp['z'], label='C' + str(c))

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_title('KMeans training...' + str(i) + ':')

        plt.legend()
        plt.pause(0.6)

    plt.ioff()
    plt.show()

    return data


""" KMeans model training host process """

if __name__ == '__main__':
    sta_time = time.time()

    X = read_csv()

    model = KMeans(k=4)

    KMeans = fit(model, X)

    print("Time:", time.time() - sta_time)