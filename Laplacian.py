import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse import csr_matrix

class Laplacian(object):
    """This clas creates Laplacian Matrix, and return Weighted matrix and Degree matrix
        Input_arguments:
        - dataset = data for which want to create Laplacian matrix
        - similarity= type of similarity function that want to use to create similarity matrix
    """
    def __init__(self, similarity: str= 'exp',sigma: int= 1):
        self.similarity= similarity
        self.sigma= sigma
    
    def LWD(self, X : np.ndarray, nearest_neighbors: int, sparse_cond: bool = True):
        """Create and return L,D,W
            Input_arguments:
          - Nearest Neighborhood = measure of how big the neighbor must to be
          - Sparse_cond = Bool to make sparse matrices (Default = True)
          """
        W = self._weighted_matrix(X, nearest_neighbors)
        D = self._degree_matrix(W)
        
        if sparse_cond == True:
            W = self._make_sparse(W)
            D = self._make_sparse(D)
        
        L = D - W
        
        return L,W,D 

    def _weighted_matrix(self, X : np.ndarray, k : int):
        """Create first  weighted graph and next weighted matrix, using similarity matrix computed by
            proximity matrix and with a similarity function
        """
        proximity_matrix = euclidean_distances(X)
        similarity_matrix =  self._similarity(proximity_matrix)
        S = similarity_matrix.copy()
        n = S.shape[0]
        
        nearest_neighbors = np.argsort(S, axis = 1)[: , ::-1][:, : k]
        for i, row in enumerate(S):
            for j in range(row.size):
                if j not in nearest_neighbors[i, :]:
                    S[i, j] = 0 if i not in nearest_neighbors[j, :] else S[i, j]
        W = S - np.eye(n)
        
        return  W
    
    def _similarity(self, P : np.ndarray):
        """Compute similarity function with spiecified type of similarity function
        """
        match self.similarity:
            case "exp":
                return np.exp(- P ** 2 / (2 * self.sigma ** 2))

    def _degree_matrix(self,W: np.ndarray):
        """Compute Degree matrix by definition
        """
        return np.diag(W.sum(axis=1))
    
    def _make_sparse(self,A: np.ndarray):
        return csr_matrix(A)