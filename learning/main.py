# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

from loader import Loader as ld
from train_test_split import TrainTestSplit as tts
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def discovery():
    ld.fetch_housing_data()
    housing = ld.load_housing_data()

    housing.info()
    housing.hist(bins=50, figsize=(20,15))

    plt.show()

    housing_with_id = housing.reset_index()
    housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
    train_set, test_set = tts.split_train_test_by_id(housing_with_id, 0.2, "id")

    train_set.info()

    X = ld.load_mnist_train()
    print(X)
    
if __name__ == '__main__':    
    lbc = load_breast_cancer()
    
    print(lbc)
    print(lbc.data)
    print(lbc.target)
    
    x = lbc.data
    y = lbc.target
    
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=44)
    
    rf = RandomForestClassifier()
    rf.fit(x_train,y_train)
    score = rf.score(x_test,y_test)
    print(score)