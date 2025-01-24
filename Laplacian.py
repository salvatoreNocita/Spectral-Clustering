import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse import csr_matrix

class Laplacian_matrix(object):
    """This clas creates Laplacian Matrix, and return Weighted matrix and Degree matrix
        Input_arguments:
        - dataset = data for which want to create Laplacian matrix
        - similarity= type of similarity function that want to use to create similarity matrix
    """
    def __init__(self,X: np.ndarray, similarity: str= 'exp',sigma: int= 1):
        self.X= X
        self.similarity= similarity
        self.sigma= sigma
    
    def LWD(self,nearest_neighbors: int, sparse_cond: bool = True):
        """Create and return L,D,W
            Input_arguments:
          - Nearest Neighborhood = measure of how big the neighbor must to be
          - Sparse_cond = Bool to make sparse matrices (Default = True)
          """
        W = self._weighted_matrix(nearest_neighbors)
        D= self._degree_matrix(W)
        L= D-W
        if sparse_cond == True:
            L,W,D= self._make_sparse(L,W,D)
        
        return L,W,D 

    def _weighted_matrix(self,k : int):
        """Create first  weighted graph and next weighted matrix, using similarity matrix computed by
            proximity matrix and with a similarity function
        """
        proximity_matrix = euclidean_distances(self.X)
        similarity_matrix =  self._similarity(proximity_matrix)
        S = similarity_matrix.copy()
        n = S.shape[0]
        
        n_neighbors = np.argsort(S, axis = 1)[: ,::-1][:, :k]
        for i, row in enumerate(S):
            for j in range(row.size):
                if j not in n_neighbors[j, :]:
                    S[i, j] = 0 if i not in n_neighbors[j, :] else S[i, j]
        W= S - np.eye(n)
        
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
    
    def _make_sparse(self,A: np.ndarray, B: np.ndarray, C: np.ndarray):
        return csr_matrix(A), csr_matrix(B), csr_matrix(C)