import pandas as pd
from SpectralClustering.Visualize import Visualize

interactive = False
k = 10
dataset = "Circle.csv"

other_kmeans = True
other_dbscan = True
other_sklearn = True


if dataset == "Circle.csv":
    data = pd.read_csv("Datasets/Circle.csv")
    labels = None
elif dataset == "Spiral.csv":
    spiral = pd.read_csv("Datasets/Spiral.csv")
    data = spiral.iloc[:, :2]
    labels = spiral.iloc[:, 2]
elif dataset == "3D_Dataset.csv":
    new_data = pd.read_csv("Datasets/3D_Dataset.csv")
    data = new_data.iloc[:,:3]
    labels = new_data.iloc[:,3]

visual = Visualize(k, interactive, dataset)

if interactive:
    visual.decide(data, similarity_rule = "exp", sigma = 1, labels = labels)
else:
    visual.player(data, labels = labels)


visual.other(other_kmeans, other_dbscan, other_sklearn)

