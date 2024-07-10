"""
K-nearest neighbors graph
    

Author(s): LelandBarnard
"""
#REQUIRED MODULES
#built-in
from typing import Union, Callable, List
#third-party
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors
#local
from ._adjacency_graph import _AdjacencyGraph
from svlite.data_structures import VectorTable, AnnotationTable


class KNeighbors(_AdjacencyGraph):
    """
    k-nearest neighbors model implemented as an adjacency graph.  Extends networkx.DiGraph.
    """

    """
    Parameters
    ----------
    n_neighbors : int
        Number of neighbors
    metric: str or callable
        Distance metric. If str, uses corresponding metric from sklearn.metrics.pairwise.distance_metrics, if callable must have signature
        fn(np.ndarray, np.ndarray) -> float.  Assumed to be symmetric.
    incoming_graph_data: nx.DiGraph, default None
        Optional arg for building a KNeighbors graph from an existing KNeighbors graph.  Should be None in most cases.
    """
    
    def __init__(self, 
        n_neighbors: int, 
        metric: Union[str, Callable[[np.ndarray, np.ndarray], float]], 
        incoming_graph_data: nx.DiGraph = None
        ):
        
        super().__init__(incoming_graph_data = incoming_graph_data)
        self.n_neighbors = n_neighbors
        self.metric = metric

        
    def populate(self, 
        node_vectors: VectorTable, 
        node_labels: AnnotationTable = None
        ):
        """
        Populate the KNeighbors directed graph with support nodes.  Each support node will have directed edges toward its k neighboring support nodes.

        Parameters
        ----------
        node_vectors: VectorTable
            VectorTable containing feature vectors to be used to define distances between support nodes
        node_labels: AnnotationTable, default None
            Optional AnnotationTable used to annotate nodes
        """
        
        self._load_vectors(node_vectors)
        self.nn_search = NearestNeighbors(n_neighbors = self.n_neighbors, metric = self.metric)
        self.nn_search.fit(self.support_features)
        self.K = self.nn_search.kneighbors_graph(mode = 'connectivity')
        self._populate()
        
        if node_labels is not None:
            self.label_support_nodes(node_labels)

        
    def embed(self, node_vectors: VectorTable):
        """
        Embed new nodes into the graph.  Embedded nodes will have directed edges toward their k neighboring support nodes, but will not have edges 
        to any other embedded nodes, and support nodes will not have edges toward embedded nodes.

        Parameters
        ----------
        node_vectors: VectorTable
            VectorTable containing feature vectors for embedded nodes, used to define distances from support nodes
        """

        
        embed_features = np.stack(node_vectors.data[node_vectors.feature_col])
        nb_lists = self.nn_search.kneighbors(embed_features)[1]
        new_edge_list = [[node_vectors.data.index[i], self.support_nodes[nb]] for i in range(nb_lists.shape[0]) for nb in nb_lists[i]]
        self.add_nodes_from(node_vectors.data.index)
        self.add_edges_from(new_edge_list)
        
        for node in node_vectors.data.index:
            self.nodes[node]['node_type'] = 'embedded'
            self.nodes[node]['node_color'] = 'unlabeled'
            
            
    def drop_support_nodes(self, nodes_to_drop: List[str], embed_dropped: bool = False) -> _AdjacencyGraph:
        """
        Drop support nodes from the graph. 
        Create new VectorTable and AnnotationTable from the remaining support nodes.
        Initialize a new KNeighbors graph from the new VectorTable and AnnotationTable.
        Return the new KNeighbors graph.
        
        Parameters:
        ----------
        nodes_to_drop: list[str]
            List of support nodes to drop from the graph.

        embed_dropped: bool, default None
            Specify whether or not to embed the dropped nodes in the resulting graph
        
        Returns:
        ----------
        _AdjacencyGraph
            New KNeighbors graph with dropped support nodes.
        """

        
        vt_in, at_in, vt_out, at_out = self._create_vector_table_annotation_table(nodes_to_drop=nodes_to_drop, return_dropped = embed_dropped)
        
        graph = KNeighbors(n_neighbors = self.n_neighbors, 
                           metric = self.metric,
                           incoming_graph_data = None)
        
        graph.populate(node_vectors = vt_in, node_labels = at_in)
        if embed_dropped:
            graph.embed(vt_out)

        return graph