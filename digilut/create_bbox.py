import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def plot_clusters(X, labels, path_name):
    # Generate a colormap
    num_labels = len(set(labels)) - (1 if -1 in labels else 0)
    colors = plt.get_cmap("tab20", num_labels + 1)  # +1 for noisy points
    cmap = ListedColormap(colors(np.arange(num_labels + 1)))

    # Plotting
    plt.figure()
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap=cmap, edgecolor="k", s=50)  # noqa: F841

    # Create a legend for the clusters
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=cmap(i),
            markersize=10,
            label=f"Cluster {i}",
        )
        for i in range(num_clusters)
    ]
    handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="black",
            markersize=10,
            label="Noise",
        )
    )
    plt.legend(handles=handles, title="Clusters")

    plt.title(f"Clustering for {path_name}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


def get_nb_occurences(labels):
    occurences = {}
    for label in labels:
        occurences[label] = occurences.get(label, 0) + 1
    return occurences


def get_n_most_recurrent_from_dict(d, n):
    # Sort the dictionary items by value (occurrences) in descending order
    sorted_items = sorted(d.items(), key=lambda item: item[1], reverse=True)

    # Get the top n items
    top_n_items = sorted_items[:n]

    # Extract just the keys (elements) from the top n items
    result = [item[0] for item in top_n_items]

    return result
