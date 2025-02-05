## README
The code consists of four classes and a main file. The datasets are located in the "Datasets" folder and are accessible.

In the main, there is a "dataset" variable where the dataset name must be entered exactly as it appears in the "Datasets" folder.

Additionally, some controls are available:

- k to define the neighborhood size (10, 20, 40).
- The "other" variables allow visualizing other clustering methods (set them to False if you do not want to see them).
- The "interactive" option can be set to True, allowing you to manually enter the required parameters via the terminal, including:

    1. Selection of the eigenvalue computation method between shifting and deflation.
    2. Selection of the eigenvector computation method between shifting and deflation.
    3. Choice of the number of clusters.
    4. Selection of the ε (eps) radius for DBSCAN.

If "interactive" is set to False, the code will run the dataset specified in the "dataset" variable in default mode.