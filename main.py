import pandas as pd
from Visualize import Visualize

interactive = False
k = 10
dataset = "Circle.csv"

other_kmeans = False
other_dbscan = False
other_sklearn = False

if dataset == "Circle.csv":
    data = pd.read_csv("Datasets\\Circle.csv")
    labels = None
elif dataset == "Spiral.csv":
    spiral = pd.read_csv("Datasets\\Spiral.csv")
    data = spiral.iloc[:, :2]
    labels = spiral.iloc[:, 2]

visual = Visualize(k, interactive)

if interactive:
    visual.decide(data, similarity_rule = "exp", sigma = 1, labels = labels)
else:
    visual.player(data, 10, labels = labels)


visual.other(other_kmeans, other_dbscan, other_sklearn)

