import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import cm
from Laplacian import Laplacian
from EigenMethods import EigenMethods

class SpectralClustering(object):
    """This class perfoms clustering and scatterplot them
        Input arguments:
        - eigenvalues: eigenvalues of Laplacian metrix computed before
        - eigenvectors: eigenvectors of Laplacian metrix computed before
        - M: number of clusters decided
        - dataset: dataset used
     """
    def __init__(self, n_nearest : int = 10, similarity : str = "exp", sparse : bool= True):
        self.laplacian = Laplacian(similarity, sigma = 1)
        self.eigen = EigenMethods(eigenval_method = "shifting", eigenvec_method = "shifting")
        self.n_nearest = n_nearest
        self.sparse = sparse
    
    def fit_predict(self, X, player_mode = False):
        try:
            X = X.values
        except:
            X = X
        L,W,D = self.laplacian.LWD(X, self.n_nearest, sparse_cond = self.sparse)
        M = self.select_M()
        eigenval, eigenvec = self.eigen.eigencompute(L,5)
        U = self.rotation_matrix(eigenval, eigenvec, M)
        if player_mode:
            return U, self.KMeans(U, M), eigenval
        return self.KMeans(U, M)

    def select_M(self):
        match self.n_nearest:
            case 10:
                return 3
            case 20:
                return 3
            case 40:
                return 2
        raise RuntimeError("Algorithm cannot manage this case")
    
    def rotation_matrix(self, eigenvalues, eigenvectors, M):
        """Compute rotation matrix U appending eigenvectors of Laplacian matrix as columns
        """
        sorted_eigenvectors_idx= np.argsort(eigenvalues)[:M]
        return eigenvectors[:,sorted_eigenvectors_idx]

    def KMeans(self, U : np.ndarray, M):
        kmeans = KMeans(n_clusters = M)
        return kmeans.fit_predict(U)

    
    def clusters_plot(self,U: np.ndarray, dataset_name: str,labels: np.array = None):
        """Plot clusters
        """
        y = U
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
            ax[0].set_title("Colors by labels")
            ax[1].scatter(self.df.values[:, 0], self.df.values[:, 1], c = color_array, s = 50)
            ax[1].set_title("Colors by spectral clustering")
            plt.grid()
            plt.show()
            return np.array(clusters)