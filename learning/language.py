import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from loader import Loader as ld


if __name__ == '__main__':
    process = ld.load_learn_process()

    process.info()
    print(process.head())
    
    melted_data = pd.melt(process, value_vars=['english','deutsch'], var_name='language', value_name='word')
    numberof = melted_data.groupby(by=['language','word'])['word'].count().orderby()
    print(numberof)
    