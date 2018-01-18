# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

from loader import Loader as ld
from train_test_split import TrainTestSplit as tts

ld.fetch_housing_data()
housing = ld.load_housing_data()

housing.info()
housing.hist(bins=50, figsize=(20,15))

plt.show()

housing_with_id = housing.reset_index()
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = tts.split_train_test_by_id(housing_with_id, 0.2, "id")

train_set.info()