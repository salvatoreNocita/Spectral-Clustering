from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pylab as plt
import matplotlib.cm as cm
import numpy as np

class OtherMethods(object):
    """Perform clustering with other given methods"""
    def __init__(self, clusters: np.array,df: np.ndarray):
        self.clusters= clusters
        self.df= df

    def K_Means_silhouette(self, M: int):
        """Compare our results with k-means resluts using shilouette score"""
        personal_score = silhouette_score(self.df.values, self.clusters)
        kmeans = KMeans(n_clusters = M)
        testclusters_kmeans = kmeans.fit_predict(self.df.values)
        kmeans_score = silhouette_score(self.df.values, testclusters_kmeans)
        print(f"Spectral Clustering Score: {personal_score}, K-Means {M} clusters Score: {kmeans_score}")
    
    def DB_Scan(self,k: int,decide: str, player: str):
        distances =  np.sort(euclidean_distances(self.df.values), axis = 1)
        k_th_neighbors = k

        if decide == 'on':
            plt.figure()
            plt.plot(np.arange(distances.shape[0]), np.sort(distances[:, k]))
            plt.grid()
            plt.title(f"{k}-th neighbor distance")
            plt.tight_layout()
            plt.show()
            print()
            eps= float(input('Give thresholds for DBscan: '))
        else:
            eps= 0.8

        
        dbscan = DBSCAN(eps = eps, min_samples = k)
        testclusters_dbscan = dbscan.fit_predict(self.df.values)

        if player == 'on':
            fig, ax = plt.subplots(1, 2, figsize = (13, 4))
            cmap = cm.Set1.colors
            color_to_clusters = {i : cmap[i] for i in np.unique(testclusters_dbscan)}
            color_array = np.array([color_to_clusters[cluster] for cluster in testclusters_dbscan])
            ax[0].plot(np.arange(distances.shape[0]), np.sort(distances[:, k]))
            ax[0].axhline(y=eps, color='r', linestyle='--', label=f'eps = {eps}')
            ax[0].grid()
            ax[0].set_title(f"{k}-th neighbor distance")
            ax[1].scatter(self.df.values[:, 0], self.df.values[:, 1], c = color_array, s = 50)
            ax[1].grid()
            ax[1].set_title(f"eps, min_samples: {eps, k}")
            fig.suptitle("DBSCAN")
            plt.show()
        
        personal_score = silhouette_score(self.df.values, self.clusters)
        dbscan = DBSCAN(min_samples = k, eps = eps)
        testclusters_dbscan = dbscan.fit_predict(self.df.values)
        dbscan_score = silhouette_score(self.df.values, testclusters_dbscan)
        print()
        print(f"Spectral Clustering Score: {personal_score}, DBScan thresh : {k} clusters Score: {dbscan_score}")
        print()