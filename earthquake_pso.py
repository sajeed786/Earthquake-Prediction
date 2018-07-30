from datetime import datetime
import math
import numpy as np
import copy
import pandas as pd
import sklearn.metrics
import pso
import functools

class Dataset:
    @staticmethod
    def load_from_file(filename):
        """
        Load and return data from file
        :param filename: path of the database.csv file
        :return: (date, latitude, longitude, magnitude) (np.array)
        """
        date, latitude, longitude, magnitude = [], [], [], []
        df = pd.read_csv(filename, engine='python')
        df['Magnitude'] = df['Magnitude'].apply(pd.to_numeric)
        df['Latitude'] = df['Latitude'].apply(pd.to_numeric)
        df['Longitude'] = df['Longitude'].apply(pd.to_numeric)
        df['Date'] = df['Date'] + df['Time']
        del df['Time']
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y%H:%M:%S')
        date = df['Date'].tolist()
        latitude = df['Latitude'].tolist()
        longitude = df['Longitude'].tolist()
        magnitude = df['Magnitude'].tolist()
        return np.array(date), np.float32(latitude), np.float32(longitude), np.float32(magnitude)

    @staticmethod
    def normalize_date(array):
        """
        Normalize datetime array
        :param array: array to normalize
        :return: normalized array (np.array)
        """
        min_data = min(array)
        max_data = max(array)
        delta = max_data - min_data
        return np.float32([(d - min_data).total_seconds() / delta.total_seconds() for d in array])

    @staticmethod
    def normalize_cord(latitude, longitude):
        """
        Normalize GPS cord array, assuming the earth is shpherical
        :param latitude: latitude array to normalize
        :param longitude: longitude array to normalize
        :return: normalized arrays (np.array)
        """
        rad_lat = np.deg2rad(latitude)
        rad_lon = np.deg2rad(longitude)

        x = np.cos(rad_lat) * np.cos(rad_lon)
        y = np.cos(rad_lat) * np.sin(rad_lon)
        z = np.sin(rad_lat)

        return x, y, z

    @staticmethod
    def vectorize(date, latitude, longitude):
        """
        Transform given array in a vectors to feed NN
        :param date: date array
        :param latitude: latitude array
        :param longitude: longitude array
        :return: np.array
        """
        return np.concatenate(Dataset.normalize_cord(latitude, longitude) + (Dataset.normalize_date(date),)) \
            .reshape((4, len(date))) \
            .swapaxes(0, 1)


class Math:
    @staticmethod
    def sigmoid(x, deriv=False):
        """
        Sigmoïd function
        :param x: np.array
        :param deriv: derivate wanted ?
        :return:
        """
        if deriv:
            return x * (1 - x)

        return 1 / (1 + np.exp(-x))

    @staticmethod
    def relu(x, deriv=False):
        """
        Rectifier function
        :param x: np.array
        :param deriv: derivate wanted ?
        :return:
        """
        if deriv:
            return np.ones_like(x) * (x > 0)

        return x * (x > 0)

    @staticmethod
    def new_parameters(x, x_min, x_max, radius):
        """
        Generate new random parameters in the sphere of center and radius given
        :param x: center on the sphere
        :param x_min: minmium value returned
        :param x_max: maximum value returned
        :param radius: radius
        :return: new parameter
        """
        alpha = 2 * np.random.random() - 1
        new_x = x + radius * alpha

        if new_x < x_min:
            return x_min
        elif new_x > x_max:
            return x_max

        return new_x


class Generator:
    @staticmethod
    def gen_random_batch(batch_size, X, Y):
        """
        Generator for random batch
        :param batch_size: size or the returned batches
        :param X: X array
        :param Y: Y array
        :return: random batches of the given size
        """
        while True:
            index = np.arange(X.shape[0])
            np.random.shuffle(index)

            s_X, s_Y = X[index], Y[index]
            for i in range(X.shape[0] // batch_size):
                yield (X[i * batch_size:(i + 1) * batch_size], Y[i * batch_size:(i + 1) * batch_size])

    @staticmethod
    def get_batch(batch_size, X, Y):
        """
        Generator to split givens arrays in smaller batches
        :param batch_size: size or the returned batches
        :param X: X array
        :param Y: Y array
        :return: random batches of the given size
        """
        if X.shape[0] % batch_size != 0:
            print("[/!\ Warning /!\] the full set will not be executed because of a poor choice of batch_size")

        for i in range(X.shape[0] // batch_size):
            yield X[i * batch_size:(i + 1) * batch_size], Y[i * batch_size:(i + 1) * batch_size]


if __name__ == "__main__":

    def dim_weights(shape):
        dim = 0
        for i in range(len(shape)-1):
            dim = dim + (shape[i] + 1) * shape[i+1]
        return dim
	
	def weights_to_vector(weights):
        w = np.asarray([])
        for i in range(len(weights)):
            v = weights[i].flatten()
            w = np.append(w, v)
        return w

    def vector_to_weights(vector, shape):
        weights = []
        idx = 0
        for i in range(len(shape)-1):
            r = shape[i] + 1
            c = shape[i+1]
            idx_min = idx
            idx_max = idx + r*c
            W = vector[idx_min:idx_max].reshape(r,c)
            weights.append(W)
        return weights
	
    def eval_network(weights,shape,X,y):
        mse=np.asarray([])
        for w in weights:
            weights = vector_to_weights(w,shape)
        l0=X
        l1=Math.sigmoid(np.dot(l0,weights[0]))
        m=np.ones((l1.shape[0],1))
        l1=np.c_[m,l1]
        l2=Math.relu(np.dot(weights[1].T,l1.T))
        l2=l2.T
        mse = np.append(mse, sklearn.metrics.mean_squared_error(y, l2))
        return mse
    # Load and prepare data
    date, latitude, longitude, magnitude = Dataset.load_from_file("database_original.csv")
    data_size = len(date)
    vectorsX, vectorsY = Dataset.vectorize(date, latitude, longitude), magnitude.reshape((data_size, 1))

    # Split vectors into train / eval sets
    eval_set_size = int(0.1 * data_size)
    index = np.arange(data_size)
    np.random.shuffle(index)
    trainX, trainY = vectorsX[index[eval_set_size:]], vectorsY[index[eval_set_size:]]
    evalX, evalY = vectorsX[index[:eval_set_size]], vectorsY[index[:eval_set_size]]

        # Hyperparameters
    batch_size = 128 
    
    max_epochs = 100
        #defining shape of neural net
    shape = (4,32,1)

    # feed forward
    X=trainX
    v=np.ones((X.shape[0],1))
    X=np.c_[v,X]
    cost_fn = functools.partial(eval_network, shape=shape, X=X, y=trainY)
    swarm = pso.ParticleSwarm(cost_fn, dim=dim_weights(shape), size=50)
            
    # Train...
    i = 0
    best_scores = [swarm.best_score]
    print ("Before updation: ",best_scores[-1]) #printing the last element of the list
    while swarm.best_score > 1e-6 and i < 500:
        swarm.update()
        i = i+1
        #print(swarm.best_score)
        if swarm.best_score < best_scores[-1]:
            best_scores.append( swarm.best_score )
            print ("Updating: ",best_scores[-1]) #printing the last element of the list
    print (best_scores)
	#finding best set of weights
	best_weights = vector_to_weights(swarm.g, shape)
 