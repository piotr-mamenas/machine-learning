import numpy as np

class Logistic(object):
    
    def sigmoid(self,z):
        return 1/(1 + np.exp(-z))
        
    def calculate_sigmoid(self,N,D,X):
        ones = np.array([[1]*N]).T
        Xb = np.concatenate((ones, X), axis=1)
        w = np.random.randn(D + 1)
        z = Xb.dot(w)
        return self.sigmoid(z)
        
    def cross_entropy(self,T, Y, N):
        E = 0
        for i in range(N):
            if T[i] == 1:
                E -= np.log(Y[i])
            else:
                E -= np.log(1 - Y[i])
        return E

    def example_cross_entropy(self,N,D,X):
        X[:50, :] = X[:50,:] - 2*np.ones((50,D))
        X[50:, :] = X[50:, :] + 2*np.ones((50,D))

        T = np.array([0]*50 + [1]*50)
        Y = self.calculate_sigmoid(N,D,X)
        print(self.cross_entropy(T,Y,N))
            
if __name__ == '__main__':
    N = 100
    D = 2
    X = np.random.randn(N,D)
    
    model = Logistic()
    model.example_cross_entropy(N,D,X)