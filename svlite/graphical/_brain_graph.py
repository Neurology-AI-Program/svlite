import pandas as pd
import numpy as np
import networkx as nx
import json
import matplotlib as mpl
import itertools

class BrainGraph:
    
    def __init__(self, graph, require_subject = True, require_incident_date = True, node_color_map = None):
        
        self.graph = graph
        if require_subject:
            assert hasattr(self.graph, '_subject_field'), 'Source graph is missing subject field'
        if require_incident_date:
            assert hasattr(self.graph, '_incident_date_field'), 'Source graph is missing subject field'
            
        if node_color_map is None:
            labels_needing_colors = [l for l in self.graph.support_labels.binary_annotation_cols] + ['unlabeled', 'query_node']
            color_cycle = itertools.cycle(mpl.colormaps['tab20'].colors)
            self.node_color_map = {k : v for k, v in zip(labels_needing_colors, color_cycle)}
        else:
            self.node_color_map = node_color_map

    def to_df(self):
        
        nodes_data = [{'label' : i, **self.graph.nodes[i]} for i in self.graph.nodes]
        
        return pd.DataFrame(nodes_data).set_index('label')        
        
    def neighbor_analysis(self, nodes = None):
        
        def _flat_table(df, value_name):
            
            return pd.melt(
                df,
                ignore_index = False,
                var_name = 'label',
                value_name = value_name
            )
        
        node_info = self.graph.node_edge_dfs(
            nodes = nodes, 
            fields = [self.graph._subject_field, self.graph._incident_date_field, 'node_color'],
            return_value = 'nodes'
        )
        
        log_odds = _flat_table(
            np.log(self.graph.neighbor_votes(nodes = nodes, metric = 'odds_ratio', annotate_results = False)), 
            value_name = 'log_odds'
        )
        
        pvals = _flat_table(
            self.graph.neighbor_votes(nodes = nodes, metric = 'fisher', annotate_results = False), 
            value_name = 'pvalue'
        )
        
        frac_in_nb = _flat_table(
            self.graph.neighbor_votes(nodes = nodes, metric = 'fraction', annotate_results = False), 
            value_name = 'fraction_inside_nb'
        
        )
        
        frac_ex_nb = _flat_table(
            self.graph.neighbor_votes(nodes = nodes, metric = 'fraction', fraction_alternative = 'ex_nb', annotate_results = False),
            value_name = 'fraction_outside_nb'
        )
        
        stats = frac_in_nb.reset_index()\
            .merge(frac_ex_nb.reset_index(), on = ['index', 'label'], how = 'left')\
            .merge(log_odds.reset_index(), on = ['index', 'label'], how = 'left')\
            .merge(pvals.reset_index(), on = ['index', 'label'], how = 'left')\
            .set_index('index')
        
        stats['direction'] = stats.apply(lambda r: 'enriched' if r['log_odds'] > 0 else 'depleted', axis = 1)


        return node_info.join(stats, how = 'left')
    
    
    def neighborhood_report(self, query_node, pvalue_threshold = 0.1):
        
        nb_associations = self.neighbor_analysis(nodes = [query_node])\
            .drop(columns = ['subject', 'scan_date', 'node_color'])\
            .query(f'pvalue < {pvalue_threshold}')\
            .sort_values('pvalue')\
            .reset_index(drop = True)\
            .to_json(orient = 'records')
        
        query_node_attributes = self.graph.nodes[query_node]
        
        if self.graph.nodes[query_node]['node_type'] == 'embedded':
            query_node_attributes['query_node_labels'] = []
        else:
            query_node_attributes['query_node_labels'] = [k for k in self.graph.support_labels.binary_annotation_cols if query_node_attributes[k] == 1]
        
        graph_legend = {'query_node' : self.node_color_map['query_node']}
        nb_view = self.graph.neighborhood_view(query_node)
        nb_graph = nx.DiGraph(nb_view)
        
        for node in nb_graph.nodes:
            node_attrs = nb_graph.nodes[node]
            if node != query_node:
                graph_legend[node_attrs['node_color']] = self.node_color_map[node_attrs['node_color']]
                node_attrs['color'] = self.node_color_map[node_attrs['node_color']]
            else:
                node_attrs['color'] = self.node_color_map['query_node']
                
        graph_data = nx.node_link_data(nb_graph)
        
        return {
            'query_node' : query_node,
            'query_node_attributes' : self.graph.nodes[query_node],
            'associations': json.loads(nb_associations),
            'graph_data' : graph_data,
            'graph_legend' : graph_legend
        }
        
    