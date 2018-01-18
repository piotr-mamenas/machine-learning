def get_answer():
    return True

def de_mean(x):
    xmean = mean(x)
    return [xi - xmean for xi in x]

def covariance(x,y):
    n = len(x)
    return dot(de_mean(x), de_mean(y)) / (n-1)

def correlation(x,y):
    stddev_x = x.std()
    stddev_y = y.std()
    return covariance(x,y) / stddev_x / stddev_y