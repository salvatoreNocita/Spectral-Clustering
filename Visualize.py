from Open_data import load_data
import matplotlib.pyplot as plt
import numpy as np
from Laplacian import Laplacian
from EigenMethods import EigenMethods
from SpectralClustering import SpectralClustering
import pandas as pd
from matplotlib import cm
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.metrics.pairwise import euclidean_distances

class Visualize():
    def __init__(self, k, interactive):
        self.interactive = interactive
        self.k = k
        self.data = None
        self.n_clusters = None

    def decide(self, data, similarity_rule = "exp", sigma = 1, labels = None):
        laplace= Laplacian(similarity_rule, sigma)

        try:
            data = data.values
        except:
            data = data
        
        self.data = data
        self.scatter(data, title = "Original Data")

        L,W,D= laplace.LWD(data, self.k, sparse_cond = True)

        eigenval_method, eigenvec_method = self.choose_eigen_method()
        eigen = EigenMethods(eigenval_method, eigenvec_method)
        eigenval, eigenvec = eigen.eigencompute(L, 5)

        self.plot_line(eigenval, title = "Eigenvalues")

        M = int(input('Insert number of clusters: '))

        spect_cls = SpectralClustering(n_nearest = self.k)
        U = spect_cls.rotation_matrix(eigenval, eigenvec, M = M)
        self.scatter(U, title = "Projected Data")

        clusters = spect_cls.KMeans(U, M)

        self.scatter(data, clusters, labels, title = "Clustered Data")
    
    def player(self, data, labels = None):
        try:
            data = data.values
        except:
            data = data
        
        self.data = data
        self.scatter(data, title = "Original Data")
        spect_cls = SpectralClustering(n_nearest = self.k)
        U, clusters, eigenval = spect_cls.fit_predict(data, player_mode = True)
        self.n_clusters = np.unique(clusters).size
        self.plot_line(eigenval, title = "Eigenvalues")
        self.scatter(U, title = "Projected Data")
        self.scatter(data, clusters, labels, title = "Clustered Data")
    
    def other(self, k_means, dbscan, sklearn):
        if k_means:
            kmeans20 = KMeans(n_clusters = self.n_clusters)
            k_means_clusters = kmeans20.fit_predict(self.data)
            self.scatter(self.data, k_means_clusters, title = f"KMeans Clusters, k : {self.k}")
        if dbscan:
            if self.interactive:
                distances =  np.sort(euclidean_distances(self.data), axis = 1)
                self.plot_line(np.sort(distances[:, self.k]), title = f"{self.k}-th neighbor distance")
                eps = int(input("Choose an eps for DBSCAN: "))
                dbscan = DBSCAN(eps = eps, min_samples = self.k)
            else:
                if self.k == 10:
                    eps = 0.75
                elif self.k == 20:
                    eps = 1
                elif self.k == 40:
                    eps = 1.5
                else:
                    raise RuntimeError(f"You should pass to interactive mode for k : {self.k}")
                
            dbscan_clusters = dbscan.fit_predict(self.data)
            self.scatter(self.data, dbscan_clusters, title = f"DBSCAN Clusters eps : {eps}, thresh : {self.k}")
        if sklearn:
            sklearn = SpectralClustering(n_clusters = self.n_clusters)
            sklearn_clusters = sklearn.fit_predict(self.data)
            self.scatter(self.data, sklearn_clusters, title = f"Sklearn's Spectral Clustering, k : {self.k}")
                
                
    
    def scatter(self, data, clusters = None, labels = None, title = "Cluster/Label Plot"):
        cmap = cm.Set1.colors
        if isinstance(clusters, np.ndarray):
            color_to_clusters = {i : cmap[i] for i in np.unique(clusters)}
            color_array = np.array([color_to_clusters[cluster] for cluster in clusters])
            legend_elements = [
            Line2D([0], [0], color=color, lw=4, label=category)
            for category, color in color_to_clusters.items()
            ]
        else:
            legend_elements = False
            color_array = [cmap[1] for i in range(data.shape[0])]
 
        if data.shape[1] <= 2:
            values = np.hstack((data, np.zeros((data.shape[0], 2 - data.shape[1]))))
            plt.figure()
            plt.scatter(values[:, 0], values[:, 1], c = color_array, s = 50)
            plt.title(title)
            plt.grid()
            if isinstance(legend_elements, list):
                plt.legend(handles=legend_elements, title="Legend")
        else:
            fig, ax = plt.subplots(1,figsize = (10,3), subplot_kw = {"projection" : "3d"})
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c = color_array)
            ax.set_title(title)
            ax.grid()
            if isinstance(legend_elements, list):
                ax.legend(handles=legend_elements, title="Legend")
        
        plt.show()
        
        if labels != None:
            self.scatter(data, labels, title = "Labeled Data")
        

    def choose_eigen_method(self):
        methods = ["shifting", "deflation"]
        flag1 = False
        flag2 = False

        while flag1 == False:
            decided_eigenvalues_method= input("Decide how to compute eigenvalues between deflation and shifting: ")
            if decided_eigenvalues_method not in methods:
                print()
                print('---Wrong name of the method, please insert value between deflation or shifting---')
                print()
                flag1 = False
            else:
                flag1 = True 
                print()   

        while flag2 == False:
            decided_eigenvectors_method= input("Decide how to compute eigenvectors between deflation and shifting: ")
            if decided_eigenvectors_method not in methods:
                print()
                print('---Wrong name of the method, please insert value between deflation or shifting---')
                print()
                flag2 = False
            else:
                flag2 = True
                print()
        return decided_eigenvalues_method, decided_eigenvectors_method
    
    def plot_line(self, series, title = "Title"):
        plt.figure()
        plt.plot(np.arange(series.size) ,series)
        plt.xticks(np.arange(series.size))
        plt.yticks(series)
        plt.title(title)
        plt.tight_layout()
        plt.grid()
        plt.show()
    
    """
    def scatter(self, data):
        
        Scatter plot of data.
        Input_argument:
        - dataset = data to plot
        
        plt.figure()
        plt.scatter(data[:,0],data[:,1])
        plt.grid()
        plt.show()
    """