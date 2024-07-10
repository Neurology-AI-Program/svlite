"""
Base graphs and graph extensions
    

Author(s): LelandBarnard
"""
#REQUIRED MODULES
#built-in
#third-party
import pandas as pd
import numpy as np
import networkx as nx
import scipy.sparse
import scipy.stats
from datetime import datetime
from typing import Union, List
#local
from svlite.data_structures import VectorTable, AnnotationTable

class _AdjacencyGraph(nx.DiGraph):
    
    def __init__(self, incoming_graph_data = None):
        
        super().__init__(incoming_graph_data = incoming_graph_data)
        self.populated = False
        self.labeled = False
        self.incoming_graph_data = incoming_graph_data


    def _load_vectors(self, node_vectors: VectorTable):

        self.node_vectors = node_vectors.copy()
        self.support_nodes = self.node_vectors.data.index
        self.support_features = self.node_vectors.to_numpy()
        # np.stack(self.node_vectors.data[self.node_vectors.feature_col])
        self._record_table_attribute(self.node_vectors, '_subjects')
        self._record_table_attribute(self.node_vectors, '_incident_dates')

    
    def _color_nodes(self, r, color_by):
        
        if color_by == 'rarest':
            binary_values = r.loc[self.prevalence.index]
            if binary_values.sum() == 0:
                return 'unlabeled'
            else:
                return sorted(zip(binary_values.items(), self.prevalence.items()), key = lambda t: (-t[0][1], t[1][1]))[0][0][0]    
            
    
    def _populate(self):
        
        if isinstance(self.K, scipy.sparse.csr_matrix):
            self.support_edges = [[self.support_nodes[i], self.support_nodes[j]] for i, j in zip(*self.K.nonzero())]
        else:
            self.support_edges = [[self.support_nodes[i], self.support_nodes[j]] for i, j in np.argwhere((self.K).astype('int'))]
        
        self.add_nodes_from(self.support_nodes)
        self.add_edges_from(self.support_edges)
        
        for node in self.support_nodes:
            self.nodes[node]['node_type'] = 'support'
            self._apply_subject_date(node)
            
        self.populated = True


    def _record_table_attribute(self, table, attr, overwrite = False):

        if not hasattr(self, attr) or overwrite:
            if attr == '_subjects' and table.subject_col is not None:
                self._subjects = {k : v for k, v in table.data[table.subject_col].items()}
                self._subject_field = table.subject_col
            elif attr == '_incident_dates' and table.incident_date_col is not None:
                self._incident_dates = {k : v for k, v in table.data[table.incident_date_col].items()}
                self._incident_date_field = table.incident_date_col


    def _apply_subject_date(self, node):

        if hasattr(self, '_subjects'):
            self.nodes[node][self._subject_field] = self._subjects[node]
        if hasattr(self, '_incident_dates'):
            self.nodes[node][self._incident_date_field] = datetime.strftime(self._incident_dates[node], "%Y-%m-%d")

        
    def label_support_nodes(self, support_labels: AnnotationTable, color_by = 'rarest'):

        set_diff = set(self.support_nodes) ^ set(support_labels.data.index)

        if any(set_diff):
            raise ValueError(f'set difference found between provided index of support_labels and support nodes')
        
        self.support_labels = support_labels.copy()
        # self.support_labels.binary_annotation_cols = support_labels.binary_annotation_cols
        self._record_table_attribute(support_labels, '_subjects')
        self._record_table_attribute(support_labels, '_incident_dates')
        self.prevalence = self.support_labels.data[self.support_labels.binary_annotation_cols].mean()
        
        for node in self.support_nodes:
            for k, v in self.support_labels.data.loc[node, self.support_labels.binary_annotation_cols].items():
                self.nodes[node][k] = int(v)
            for k, v in self.support_labels.data.loc[node, self.support_labels.continuous_annotation_cols].items():
                self.nodes[node][k] = float(v)
            for k, v in self.support_labels.data.loc[node, self.support_labels.supplemental_cols].items():
                if isinstance(v, str):
                    self.nodes[node][k] = v
                else:
                    self.nodes[node][k] = v.item()
            if color_by is not None:
                self.nodes[node]['node_color'] = self._color_nodes(self.support_labels.data.loc[node], color_by)
            self._apply_subject_date(node)
                
        self.labeled = True
            
        
    def neighbor_votes(
        self, 
        nodes = None, 
        metric = 'fraction', 
        output = 'raw', 
        threshold = None, 
        threshold_direction = 'greater',
        fisher_alternative = 'two-sided',
        fraction_alternative = 'in_nb',
        top_n = 1,
        annotate_results = True):
        
        assert self.labeled, "Support nodes must be labeled for neighbor voting"

        def _crosstabs(nbs, node):
                
            intab = self.support_labels.data[self.support_labels.binary_annotation_cols].loc[nbs]
            extab = self.support_labels.data[self.support_labels.binary_annotation_cols].drop(nbs + [node], errors = 'ignore')
            
            c00 = intab.sum() + 0.5
            c01 = len(intab) - intab.sum() + 0.5
            c10 = extab.sum() + 0.5
            c11 = len(extab) - extab.sum() + 0.5
            
            return c00, c01, c10, c11
            
        
        if nodes is None:
            nodes = list(self.nodes)
        
        votes = pd.DataFrame(index = nodes, data = np.zeros((len(nodes), len(self.support_labels.binary_annotation_cols))), columns = self.support_labels.binary_annotation_cols)
        
        for node in votes.index:
            nbs = list(self.neighbors(node))
            scores = np.zeros(len(self.support_labels.binary_annotation_cols))
            if metric == 'fraction':
                if fraction_alternative == 'in_nb':
                    scores = self.support_labels.data[self.support_labels.binary_annotation_cols].loc[nbs].mean()
                elif fraction_alternative == 'ex_nb':
                    scores = self.support_labels.data[self.support_labels.binary_annotation_cols].drop(nbs + [node], errors = 'ignore').mean()
                else:
                    raise ValueError(f'fraction_alternative must be in ["in_nb", "ex_nb"], got {fraction_alternative}')
                
            elif metric == 'odds_ratio':
                
                c00, c01, c10, c11 = _crosstabs(nbs, node)
                
                scores = (c00*c11)/(c10*c01)
                
            elif metric == 'fisher':
                
                if fisher_alternative not in ['less', 'greater', 'two-sided']:
                    raise ValueError(f'fisher_alternative must be in ["less", "greater", "two-sided"], got {fisher_alternative}')
                
                c00, c01, c10, c11 = _crosstabs(nbs, node)
                func1d = lambda arr: scipy.stats.fisher_exact(arr.reshape(2, 2), alternative = fisher_alternative)[1]
                
                scores = np.apply_along_axis(func1d = func1d, axis = 0, arr = np.stack([c00, c01, c10, c11]))
                
            else:
                raise ValueError(f'metric must be one of ["fraction", "odds_ratio", "fisher"], got {metric}')
                
            if output == 'raw':
                
                votes.loc[node, :] = scores
                
            elif output == 'winner':

                if not isinstance(top_n, int):
                    raise ValueError(f'top_n must an integer, got {top_n}')
                
                if top_n < 1:
                    raise ValueError(f'top_n must be positive, got {top_n}')
                
                if metric == 'fisher':
                    scores = -scores
                
                winners = [t[1][0] for t in sorted(zip(scores, self.prevalence.items()), key = lambda t: (-t[0], t[1][1]))[:top_n]]
                
                votes.loc[node, winners] = 1
                
            elif output == 'threshold':
                
                try:
                    threshold = float(threshold)
                except:
                    raise ValueError(f'to output thresholded results, a numeric value must be passed for "threshold", got {threshold}')
                
                if threshold_direction == 'greater':
                    votes.loc[node, :] = (scores >= threshold).astype('int')
                elif threshold_direction == 'less':
                    votes.loc[node, :] = (scores <= threshold).astype('int')
                else:
                    raise ValueError(f'direction for thresholding must be either "greater" or "less", got {threshold_direction}')
                    
            else:
                raise ValueError(f'"output" must be one of ["raw", "winner", "threshold"], got {output}')
                
            if annotate_results:
                for k, v in votes.loc[node, :].items():
                    self.nodes[node][f'{k}_{metric}_{output}'] = v
                if output == 'winner':
                    self.nodes[node][f'{metric}_{output}'] = winners[0]
        
        return votes
    
    
    def node_edge_dfs(self, nodes = None, fields = None, return_value = 'both'):

        if nodes is None:
            nodes = list(self.nodes)
        node_df = pd.DataFrame([self.nodes[n] for n in nodes], index = nodes).fillna(-1)
        
        if fields is not None:
            node_df = node_df[fields]

        base_graph = nx.DiGraph(incoming_graph_data = self)
        subgraph = base_graph.subgraph(nodes)
        edge_df = pd.DataFrame(subgraph.edges, columns = ['source', 'target'])

        if return_value == 'both':
            return node_df, edge_df
        elif return_value == 'nodes':
            return node_df
        elif return_value == 'edges':
            return edge_df
        else:
            raise ValueError(f'return_value must be one of "nodes", "edges", or "both", got {return_value}')
        
    
    def neighborhood_view(self, node):

        nbs = list(self.neighbors(node))
        base_graph = nx.DiGraph(incoming_graph_data = self)
        
        return base_graph.subgraph(nbs + [node])
    

    def _create_vector_table_annotation_table(self, nodes_to_drop: List[str], return_dropped: bool = False) -> Union[VectorTable, AnnotationTable]:
        """
        Removes support nodes from VectorTable and AnnotationTable, 
        and returns them a new VectorTable or AnnotationTable.
        
        Parameters
        ----------
        nodes_to_drop : List[str]
            List of support nodes to drop.
        return_dropped: bool (default False)
            Specify whether to return the VectorTable and AnnotationTable of the dropped nodes
        
        Returns
        -------
            Union[VectorTable, AnnotationTable]
        """
        new_vt = None 
        new_at = None
        dropped_vt = None
        dropped_at = None
        
        # drop nodes from VectorTable and copy to new VectorTable
        new_vt = self.node_vectors.copy()
        new_vt.data.drop(nodes_to_drop, errors='raise', inplace=True)
        if return_dropped:
            dropped_vt = self.node_vectors.copy()
            dropped_vt.data = dropped_vt.data.loc[nodes_to_drop]

        # drop nodes from AnnotationTable
        if self.labeled:
            new_at = self.support_labels.copy()
            new_at.data.drop(nodes_to_drop, errors='raise', inplace=True)
            if return_dropped:
                dropped_at = self.support_labels.copy()
                dropped_at.data = dropped_at.data.loc[nodes_to_drop]
            
        return new_vt, new_at, dropped_vt, dropped_at
    

    def create_neighborhood_centers(self, 
                                    n_representatives: int = 5,
                                    **neighborhood_votes_args) -> Union[VectorTable, dict]:
        
        """
        Returns the centers of the top enriched nodes in each neighborhood for each support label.
        
        Parameters
        ----------
        n_representatives : int
            number of representatives to keep for each label
            
        neighborhood_votes_args : Dict
            arguments to pass to self.neighborhood_votes() 
        
        
        Returns
        -------
            group_centers: VectorTable with the calculated group centers of n_representatives for each label
            group_dict: Dictionary mapping the group names to the feature names
        """
        
        if not neighborhood_votes_args:
            metric = 'fraction'
            votes = self.neighbor_votes()
        else:
            metric = neighborhood_votes_args['metric']
            votes = self.neighbor_votes(**neighborhood_votes_args)
                    
        # for each column in votes, sort by that column, keep top n_representatives
        groups = []
        group_names = []
        for column in votes.columns:
            # When sorting, if metric = 'fisher' it is pvalue so sort ascending, else sort descending
            if metric == 'fisher':
                top_indexes = votes.nsmallest(n_representatives, column).index.tolist()
            else:
                top_indexes = votes.nlargest(n_representatives, column).index.tolist()
                
            groups.append(top_indexes)
            group_names.append(column)
            
        centers_vt, centers = self.node_vectors.calc_group_centers(groups = groups, metric = self.metric, group_names = group_names)
        
        return centers_vt, centers
        
    