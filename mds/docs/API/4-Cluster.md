# Cluster
This section introduces the class `SingleCluster` and `ClusterList`.

## SingleCluster
The class `SingleCluster` is an internal class for TrendFlow. It represents the clusters of article after the clustering procedure in TrendFlow. It stores the keyphrases of the cluster.

### Instance Attributes
- '``top_5_keyphrases (list)`: the top-5 keyphrases of the cluster


### Instance Functions
- `get_elements(self) -> List`: return a list of article indexes of this cluster. The indexes are with respect to the associated `ArticleList` object returned by TrendFlow.
- `get_keyphrases(self) -> List`: return a list of (keyphrase, count of keyphrase).
- `print_keyphrases(self)`: print all the keyphrases


## ClusterList
The list representation of clusters.

### Instance Functions
- `sort(self)`: sort the clusters according to the size of each cluster (i.e. the number of articles within the cluster) descendingly


