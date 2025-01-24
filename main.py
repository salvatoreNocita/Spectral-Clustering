import pandas as pd
import numpy as np
from Open_data import load_data
import matplotlib.pyplot as plt
from Laplacian import Laplacian_matrix
from Eigen_Methods import EigenMethods
from Clustering import Clusters
from Other_Clustering_Methods import OtherMethods

def main():
    player = 'on'
    decide= 'on'

    ## Load dataset

    if decide == 'on':
        names= ['spiral','circle']
        flag1= False
        while flag1 == False:
            print()
            nome_dataset= input("Give name dataset between spiral and circle: ")
            print()
            data= load_data(nome_dataset)
            try:
                dataset, labels= data.open_data()
                flag1= True
            except Exception as e:
                print()
                print("----Name of dataset not valid, please insert value between spiral and circle----")
                print()
                flag1= False
    else:
        nome_dataset = 'circle'
        data= load_data()
        dataset, labels= data.open_data()
    
    ## Visualize scatterplot of data
    if player == 'on':
        data.visualize(dataset)
    
    ## Compute Laplacian, Weighted, Degree Metrices

    if decide == 'on':
        print()
        k= int(input('Give value (k) of the neighbor to use: '))
        print()
    else:
        k=10

    laplace= Laplacian_matrix(dataset)
    L,W,D= laplace.LWD(k,sparse_cond=True)

    ## Compute eigenvalues,eigenvectors
    eigen= EigenMethods()
    M= 5    #Test number
    if decide == 'on':
        methods= ['deflation','shifting']
        flag2 = False
        flag3 = False
        while flag2 == False:
            decided_eigenvalues_method= input("Decide how to compute eigenvalues between deflation and shifting: ")
            if decided_eigenvalues_method not in methods:
                print()
                print('---Wrong name of the method, please insert value between deflation or shifting---')
                print()
                flag2 = False
            else:
                flag2 = True 
                print()               
        while flag3 == False:
            decided_eigenvectors_method= input("Decide how to compute eigenvectors between deflation and shifting: ")
            if decided_eigenvectors_method not in methods:
                print()
                print('---Wrong name of the method, please insert value between deflation or shifting---')
                print()
                flag3 = False
            else:
                flag3 = True
                print()

        if decided_eigenvalues_method == 'deflation' and decided_eigenvectors_method =='deflation':
            decided_eigenvalues,decided_eigenvectors= eigen.deflation_inverse_power_method(L,M, compute_eigenvectors= True)
        elif decided_eigenvalues_method == 'shifting' and decided_eigenvectors_method == 'shifting':
            decided_eigenvalues,decided_eigenvectors= eigen.shifting_small_method(L,M)
        elif decided_eigenvalues_method == 'deflation' and decided_eigenvectors_method == 'shifting':
            decided_eigenvalues,_= eigen.deflation_inverse_power_method(L,M, compute_eigenvectors= False)
            _,decided_eigenvectors= eigen.shifting_small_method(L,M)
        elif decided_eigenvalues_method == 'shifting' and decided_eigenvectors_method == 'deflation':
            decided_eigenvalues,_= eigen.shifting_small_method(L,M)
            _,decided_eigenvectors= eigen.deflation_inverse_power_method(L,M,compute_eigenvectors= True)

    else:
        decided_eigenvalues,_= eigen.deflation_inverse_power_method(L,M,compute_eigenvectors=False)
        _,decided_eigenvectors= eigen.shifting_small_method(L,M)


    ## Decide number of clusters and M smallest eigenvalues,eigenvectors
    if player == 'on':
        plt.figure()
        plt.plot(np.arange(len(decided_eigenvalues)),decided_eigenvalues)
        plt.yticks(decided_eigenvalues)
        plt.tight_layout()
        plt.grid()
        plt.show()

        if decide == 'on':
            decided_M= int(input('Insert number of clusters: '))
        else:
            decided_M = 3

    else:
        decided_M= 3
    
    defl_compute_eigenvectors= False    #Set True if you want deflaction eigenvectors
    small_eigenvalues,small_eigenvectors= eigen.shifting_small_method(L,decided_M)
    small_eigenvalues_defl,small_eigenvectors_defl= eigen.deflation_inverse_power_method(L,decided_M, defl_compute_eigenvectors)
    
    ## Perform Clustering
    if defl_compute_eigenvectors == False:
        clustering= Clusters(small_eigenvalues_defl, small_eigenvectors,decided_M,dataset)
    elif defl_compute_eigenvectors == True:
        clustering= Clusters(small_eigenvectors_defl,small_eigenvectors_defl,decided_M,dataset)
    
    U= clustering.rotation_matrix()
    if nome_dataset == 'circle':
        clusters= clustering.clusters_plot(U,nome_dataset)
    elif nome_dataset == 'spiral':
        clusters= clustering.clusters_plot(U,nome_dataset,labels= labels)

    ## Perform other clustering methods
    o_methods= OtherMethods(clusters,dataset)
    print()
    o_methods.K_Means_silhouette(decided_M)
    print()
    o_methods.DB_Scan(k,decide=decide,player=player)


main()
