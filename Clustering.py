import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import cm

class Clusters(object):
    """This class perfoms clustering and scatterplot them
        Input arguments:
        - eigenvalues: eigenvalues of Laplacian metrix computed before
        - eigenvectors: eigenvectors of Laplacian metrix computed before
        - M: number of clusters decided
        - dataset: dataset used
     """
    def __init__(self, eigenvalues: np.array, eigenvectors: np.array,M: int, dataset: np.ndarray):
        self.eigenvalues= eigenvalues
        self.eigenvectors= eigenvectors
        self.M= M
        self.df= dataset
    
    def rotation_matrix(self):
        """Compute rotation matrix U appending eigenvectors of Laplacian metrix as columns
        """
        sorted_eigenvectors_idx= np.argsort(self.eigenvalues)[:self.M]
        return self.eigenvectors[:,sorted_eigenvectors_idx].T
    
    def clusters_plot(self,U: np.ndarray, dataset_name: str,labels: np.array = None):
        """Plot clusters
        """
        y= U.T
        fig, ax = plt.subplots(1,figsize = (10,3), subplot_kw = {"projection" : "3d"})
        if self.M == 1:
            print("You cant't perform scatter plot with 1 cluster")
        elif self.M == 2:
            ax.scatter(y[:, 0],y[:,1])
        elif self.M == 0:
            raise("You must have at least 1 Cluster since there is at least one eigenvalue = 0")
        else:
            ax.scatter(y[:, 0],y[:,1],y[:,2])
        ax.grid()
        plt.show()

        if dataset_name == 'circle':
            kmeans = KMeans(n_clusters = self.M)
            clusters = kmeans.fit_predict(y)
            cmap = cm.Set1.colors
            color_to_clusters = {i : cmap[i] for i in range(self.M)}
            color_array = np.array([color_to_clusters[cluster] for cluster in clusters])
            plt.figure()
            plt.scatter(self.df.values[:, 0], self.df.values[:, 1], c = color_array, s = 50)
            plt.title("Colors by spectral clustering")
            plt.grid()
            plt.show()
            return np.array(clusters)
        
        elif dataset_name == 'spiral':
            kmeans = KMeans(n_clusters = self.M)
            clusters = kmeans.fit_predict(y)
            cmap = cm.Set1.colors
            color_to_clusters = {i : cmap[i] for i in range(self.M)}
            color_array = np.array([color_to_clusters[cluster] for cluster in clusters])
            fig, ax= plt.subplots(1,2, figsize= (10,4))
            ax[0].scatter(self.df.values[:, 0], self.df.values[:, 1], c = labels, s = 50)
            ax[0].set_title("Colors by lables")
            ax[1].scatter(self.df.values[:, 0], self.df.values[:, 1], c = color_array, s = 50)
            ax[1].set_title("Colors by spectral clustering")
            plt.grid()
            plt.show()
            return np.array(clusters)