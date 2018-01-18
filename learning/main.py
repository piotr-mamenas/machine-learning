from loader import Loader

Loader.fetch_housing_data()
dt = Loader.load_housing_data()

dt.info()