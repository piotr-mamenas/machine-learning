import numpy as np

import matplotlib.pyplot as plt
from loader import Loader as ld


if __name__ == '__main__': 
    combats = ld.load_pokemon_combat()
    train = ld.load_pokemon_train()
    tests = ld.load_pokemon_tests()
    
    combats.info()
    train.info()
    
    print(train.head())
    train.set_index("#")
    print(train.head())