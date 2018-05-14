import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class Playground(object):

    def get_data(self):
        return load_breast_cancer()




if __name__ == '__main__':
    playground = Playground();
    cancer = playground.get_data()
    print(cancer.target_names)
    print(cancer.data[cancer.target == 0])
        
    scaler = StandardScaler()
    scaler.fit(cancer.data)
    scaled_data = scaler.transform(cancer.data)
    
    pca = PCA(n_components=2)
    pca.fit(scaled_data)
    scaled_pca = pca.transform(scaled_data)
    
    print(scaled_data.shape)
    print(scaled_pca.shape)
    
    plt.scatter(scaled_pca[:,0], scaled_pca[:,1], s=100, color=['red','blue'], alpha=0.5)
    plt.show()