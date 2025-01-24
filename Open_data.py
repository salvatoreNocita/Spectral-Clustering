import pandas as pd
import numpy as np
import matplotlib.pylab as plt

class load_data():
    """This class open data and returns dataset to work with.
        Input_argument:
        - name_data = the name of the problem to handle 
    """
    def __init__(self,name_data = 'circle'):
        self.name_data= name_data
    
    def open_data(self):
        if self.name_data == 'circle':
            dataset= pd.read_csv('Datasets/Circle.csv', names=['x1','x2'])
            print(dataset)
            return dataset, None
        elif self.name_data == 'spiral':
            dataset= pd.read_csv('Datasets/Spiral.csv', names=['x1','x2','labels'])
            labels= dataset['labels'].values
            data= dataset.copy().drop(columns=['labels'])
            print(data)
            return data,labels
    
    def visualize(self,dataset):
        """
        Scatter plot of data.
        Input_argument:
        - dataset = data to plot
        """
        plt.figure()
        plt.scatter(dataset.values[:,0],dataset.values[:,1])
        plt.grid()
        plt.show()