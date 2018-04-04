import numpy as np
import pandas as pd

class TfUtils(object):
    def init_weight_and_biases(sizeX, sizeY):
        weight = np.random.randn(sizeX, sizeY) / np.sqrt(sizeX, sizeY)
        bias = np.zeros(sizeY)
        return weight.astype(np.float32), bias.astype(np.float32)

    def relu(weights):
        return weights * (weights > 0)

    def sigmoid(weights):
        return 1 / (1 + np.exp(-weights))

    def softmax(weights):
        expA = np.exp(weights)
        return expA / expA.sum(axis=1, keepdims=True)

    def error_rate(targets, predictions):
        return np.mean(targets != predictions)

    def targets_to_indicator(targets):
        sizeX = len(targets)
        sizeY = len(set(targets))
        indicator = np.zeros((sizeX,sizeY))
        for i in range(sizeX):
            indicator[i, targets[i]] = 1
        return indicator
