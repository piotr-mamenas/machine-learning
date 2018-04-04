# -*- coding: utf-8 -*-
import os
import tarfile
import numpy as np
import pandas as pd

from six.moves import urllib

class Loader:
    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    HOUSING_PATH = "datasets/housing"
    MNIST_PATH = "datasets/mnist"
    POKEMON_PATH = "datasets/pokemon"
    LEARN_PROCESS_PATH = "datasets/language-learn"
    HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

    def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
        if not os.path.isdir(housing_path):
            os.makedirs(housing_path)
            tgz_path = os.path.join(housing_path, "housing.tgz")
            urllib.request.urlretrieve(housing_url, tgz_path)
            housing_tgz = tarfile.open(tgz_path)
            housing_tgz.extractall(path=housing_path)
            housing_tgz.close()

    def load_housing_data(housing_path=HOUSING_PATH):
        csv_path = os.path.join(housing_path, "housing.csv")
        return pd.read_csv(csv_path)

    def load_mnist_train(limit=None, mnist_path=MNIST_PATH):
        print("Reading and transforming MNIST...")
        csv_path = os.path.join(mnist_path, "train.csv")
        df = pd.read_csv(csv_path)
        data = df.as_matrix()
        np.random.shuffle(data)
        X = data[:, 1:] / 255.0 # data is from 0..255
        Y = data[:, 0]
        if limit is not None:
            X, Y = X[:limit], Y[:limit]
        return X, Y

    def load_pokemon_train(limit=None, pokemon_path=POKEMON_PATH):
        csv_path = os.path.join(pokemon_path, "pokemon.csv")
        return pd.read_csv(csv_path)

    def load_pokemon_combat(limit=None, pokemon_path=POKEMON_PATH):
        csv_path = os.path.join(pokemon_path, "combats.csv")
        return pd.read_csv(csv_path)

    def load_pokemon_tests(limit=None, pokemon_path=POKEMON_PATH):
        csv_path = os.path.join(pokemon_path, "tests.csv")
        return pd.read_csv(csv_path)

    def load_learn_process(limit=None, learn_process_path=LEARN_PROCESS_PATH):
        csv_path = os.path.join(learn_process_path, "learn_process.csv")
        return pd.read_csv(csv_path)

    def load_image_recognition(balance_ones=True):
        y = []
        x = []
        first = True
        for line in open('datasets/fer2013.csv'):
            if first:
                first = False
            else:
                row = line.split(',')
                y.append(int(row[0]))
                x.append([int(p) for p in row[1].split()])

        x, y = np.array(x) / 255.0, np.array(y)

        if balance_ones:
            x0, y0 = x[y != 1, :], y[y != 1]
            x1 = x[y==1, :]
            x1 = np.repeat(x1, 9, axis=0)
            x = np.vstack([x0, x1])
            y = np.concatenate((y0, [1]*len(x1)))

        return x, y

    def get_image_data():
        x, y = load_image_recognition()
        n, d = x.shape
        d = int(np.sqrt(d))
        x = x.reshape(n, 1, d, d)
        return x, y

    def get_binary_image_data():
        x = []
        y = []
        first = True
        for line in open('datasets/fer2013.csv'):
            if first:
                first = False
            else:
                row = line.split(',')
                y = int(row[0])
                if y == 0 or y == 1:
                    y.append(y)
                    x.append([int(p) for p in row[1].split()])
        return np.array(x) / 255.0, np.array(y)
