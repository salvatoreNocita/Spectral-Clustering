import numpy as np

class EigenMethods(object):
    """This class contains principals method for eigenvalue and eigenvector
    """
    def __init__(self):
        pass


    def power_method(self,A, mu = 0, tol = 1e-8, max_iter = 1000):
        """
        Inverse power method applied to the matrix A shifted by a factor mu*I. 
        Returns the smallest eigenvalue of A and the associated eigenvector
        """
        n = A.shape[0]
        v = np.random.rand(n, 1)
        v = v / np.linalg.norm(v)
        I = np.eye(n)
    
        for k in range(max_iter):
            #Solve (A - mu * I) * v_new = v_old
            y= (A - mu * I) @ v
            v_new = y / np.linalg.norm(y)
            lambda_k = (v_new.T @ A @ v_new).item()            #copy an element of an array in standard python scalar and return it
            v_new =v_new.reshape((n,1))
            if np.linalg.norm(A @ v_new - lambda_k * v_new) < tol:
                return lambda_k, v_new
                break
            else:
                v = v_new
                continue


    def inverse_power_method(self,A, mu= 0, tol = 1e-8, max_iter = 1000):
        """
        Inverse power method applied to the matrix A shifted by a factor mu*I. 
        Returns the smallest eigenvalue of A and the associated eigenvector
        """
        n = A.shape[0]
        v = np.random.rand(n)
        v = v / np.linalg.norm(v)
        I = np.eye(n)
    
        for k in range(max_iter):
            y = np.linalg.solve(A - mu * I, v)                          #Solve (A - mu * I) * v_new = v_old
            v_new = y / np.linalg.norm(y)
            lambda_k = (v_new.T @ A @ v_new).item()
            if np.linalg.norm(A @ v_new - lambda_k * v_new) < tol:      #We accept just a treshold of equality due numerical cancellation
                return lambda_k, v
            v = v_new 


    def shifting_small_method(self,A, M, tol = 1e-8):
        """
        Shifting method applied to the symmetric matrix A in order to find the smallest M eigenvalues and 
        corresponding eigenvectors of A.
        A: symmetric matrix for which we want to return the eigenvalues
        M: number of small eigenvalues to compute
        tol: tolerance of the inverse power method
        Returns the M smallest eigenvalues and the corresponding eigenvectors.
        """
        eigenvalues = []
        eigenvectors = []
        A_current = A.copy()
        n = A.shape[0]
        """
        try:
            gamma, _ = power_method(A_current, 0, tol, max_iter = 1000)
        except np.linalg.LinAlgError:
            mu = 1e-6
            gamma, _ = power_method(A_current, mu, tol, max_iter = 1000)
        """
        gamma = 100
        if M > n:
            raise np.linalg.LinAlgError(f"{n}-squared cannot have {M} eigenvalues")

        for i in range(M):
            try:
            #first try to apply inverse power method without shifting the matrix A (mu = 0)
                eigenvalue_i, eigenvector_i = self.inverse_power_method(A_current, 0, tol, max_iter = 1000) 
            except np.linalg.LinAlgError:
            #if LinAlgError occurs (singular matrix, eigenvalue equal to 0), then matrix A is shifted by a 
            # small quantity mu to allow the computation
                mu = 1e-6
                eigenvalue_i, eigenvector_i = self.inverse_power_method(A_current, mu, tol, max_iter = 1000) 
            eigenvalues.append(eigenvalue_i)
            eigenvectors.append(eigenvector_i)

            #delete the contribute along the rank 1 matrix eigenvector_i*eigenvector_i.T
            A_current = A_current  - eigenvalue_i * np.outer(eigenvector_i, eigenvector_i) 
            #it could be a problem for the conditioning number of the next iterations (no prob if i compute the maximum eigenvalue)
            A_current = A_current  + gamma * np.outer(eigenvector_i, eigenvector_i)
    
        return np.array(eigenvalues), np.array(eigenvectors).T


    def deflation_inverse_power_method(self,A,M=4,mu=0,tol=1e-8,max_iter=1000, compute_eigenvectors= False):
        n= A.shape[0]
        e1= np.zeros(n).T
        e1[0]= 1
        Actual_A= A.copy()
        E= np.zeros((n,n))
        B_prec= np.zeros((n,n))
        P_prec= np.zeros((n,n))
        Actual_vector= np.zeros((1,n))
        eigenvalues= []
        eigenvectors= []
        if compute_eigenvectors == False:
            P= np.zeros((n,n))
            eigenvalues= []
            for i in range(M):
                min_lamda_i, x_i = self.inverse_power_method(Actual_A,mu)
                eigenvalues.append(min_lamda_i)
                P_bar_i= np.eye(n-i) - 2* (np.outer(x_i + e1[:n-i],x_i + e1[:n-i])/(np.linalg.norm(x_i + e1[:n-i])**2))
                B_bar_i= P_bar_i @ Actual_A @ P_bar_i
                Actual_A= B_bar_i[1:,1:]
                P[i,i]= min_lamda_i
            return eigenvalues, None
        else:
            for i in range(M):
                if i == 0:
                    try:
                        min_eigenvalue_i, x_i= self.inverse_power_method(A, mu=0)
                    except np.linalg.LinAlgError:
                        min_eigenvalue_i, x_i= self.inverse_power_method(A, mu=1e-6)
                    eigenvalues.append(min_eigenvalue_i)
                    eigenvectors.append(x_i)
                    Actual_vector = x_i
                    P_bar_i= np.eye(n-i) - 2* (np.outer(x_i + e1[:n-i],x_i + e1[:n-i])/(np.linalg.norm(x_i + e1[:n-i])**2))
                    B_bar_i= P_bar_i @ Actual_A @ P_bar_i
                    Actual_A= B_bar_i[1:,1:]
                    E[i,i]= min_eigenvalue_i
                    P_prec = P_bar_i
                    B_prec = B_bar_i
                else:
                    E[i,i]= min_eigenvalue_i
                    P_bar_i= np.eye(n-i) - 2* (np.outer(x_i + e1[:n-i],x_i + e1[:n-i])/(np.linalg.norm(x_i + e1[:n-i])**2))
                    B_bar_i= P_bar_i @ Actual_A @ P_bar_i
                    Actual_A= B_bar_i[1:,1:]
                    try:
                        min_eigenvalue_i,_= self.inverse_power_method(Actual_A,mu=0)
                    except np.linalg.LinAlgError:
                        min_eigenvalue_i,_= self.inverse_power_method(Actual_A, mu=1e-6)
                    eigenvalues.append(min_eigenvalue_i)
                    P_i= np.zeros((n,n))
                    for k in range(0,i):
                        P_i[k,k]= 1
                    P_i[i+1:,i+1:]= P_bar_i
                    B_i= P_i @ B_prec @ P_i
                    first_parameter= B_prec[i-1,i:]*Actual_vector/(eigenvalues[i-1]-min_eigenvalue_i)
                    new_vector= np.zeros((1,n-i-1))
                    new_vector[0]= first_parameter
                    new_vector[1:]= Actual_vector
                    Actual_A= new_vector
                    for j in range(0,i-1,-1):
                        parameter= B_prec[j,j:]*Actual_A/(eigenvalues[j-1]-min_eigenvalue_i)
                        new_vector= np.zeros((1,j-1))
                        new_vector[0]= parameter
                        new_vector[1:]= Actual_vector
                        Actual_vector= new_vector
                    eigenvectors.append(Actual_vector)
            return eigenvalues,eigenvectors