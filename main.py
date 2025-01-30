import pandas as pd
from Visualize import Visualize

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
elif dataset == "Chameleon.csv":
    data = pd.read_csv("Datasets/Chameleon.csv")
    labels = None

visual = Visualize(k, interactive, dataset)

if interactive:
    visual.decide(data, similarity_rule = "exp", sigma = 1, labels = labels)
else:
    visual.player(data, labels = labels)


visual.other(other_kmeans, other_dbscan, other_sklearn)

