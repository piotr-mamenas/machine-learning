import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.neural_network import MLPClassifier

class Playground(object):
    
    def pca_run(self):
        cancer = load_breast_cancer()
        print(cancer.target_names)
        print(cancer.data[cancer.target == 0])
    
        scaler = StandardScaler()
        scaler.fit(cancer.data)
        scaled_data = scaler.transform(cancer.data)
        
        pca = PCA(n_components=5)
        pca.fit(scaled_data)
        scaled_pca = pca.transform(scaled_data)
        
        print(scaled_data.shape)
        print(scaled_pca.shape)
    
        plt.scatter(scaled_pca[:,0], scaled_pca[:,1], s=100, color=['red','blue'], alpha=0.5)
        plt.show()
        
    def univar_run(self):
        iris = load_iris()
        data, targets = iris.data, iris.target
        print(data.shape)
        
        dataTransformed = SelectKBest(chi2, k=2).fit_transform(data,targets)
        print(dataTransformed.shape)
        return data, targets, dataTransformed
        
if __name__ == '__main__':
    playground = Playground();
    playground.pca_run();
    data, targets, dataTransformed = playground.univar_run();
    
    scaler = StandardScaler()
    
    x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size=0.33, random_state=42)
    x2_train, x2_test, y2_train, y2_test = train_test_split(dataTransformed, targets, test_size=0.33, random_state=42)
    
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    x2_train = scaler.fit_transform(x2_train)
    x2_test = scaler.fit_transform(x2_test)
    
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10,), random_state=1)
    clf.fit(x_train,y_train)
    score = clf.score(x_test,y_test)
    print(score)
    
    clf2 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10,), random_state=1)
    clf.fit(x2_train, y2_train)
    score2 = clf.score(x2_test, y2_test)
    print(score2)
    
    
    
    
    
    