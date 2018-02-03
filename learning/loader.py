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
