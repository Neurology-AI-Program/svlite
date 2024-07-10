import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score, auc, roc_curve

def _nan_safe(v, default):
    if np.isinf(v) or np.isnan(v):
        return default
    else:
        return v


def get_calls(graph, threshold = 1, top_n = 1, nodes = None):

    winners = graph.neighbor_votes(nodes = nodes, metric = 'fisher', fisher_alternative='greater', output='winner', top_n = top_n)
    enriched = graph.neighbor_votes(nodes = nodes, metric = 'odds_ratio', output = 'threshold', threshold_direction = 'greater', threshold = 1)
    thresh = graph.neighbor_votes(nodes = nodes, metric = 'fisher', fisher_alternative='greater', output='threshold', threshold_direction='less', threshold = threshold)

    return winners*enriched*thresh


def sensitivity(labels, calls):

    tp = labels*calls.values
    sens = tp.sum()/labels.sum()
    sens['total'] = tp.sum().sum()/labels.sum().sum()
    
    return dict(sens)


def precision(labels, calls):

    tp = labels*calls.values
    prec = tp.sum()/calls.sum()
    prec['total'] = tp.sum().sum()/calls.sum().sum()

    prec = dict(prec)
    for k in prec:
        prec[k] = _nan_safe(prec[k], 0)

    return prec


def false_discovery_rate(labels, calls):

    fp = (1 - labels)*calls.values
    fdr = fp.sum()/calls.sum()
    fdr['total'] = fp.sum().sum()/calls.sum().sum()

    fdr = dict(fdr)
    for k in fdr:
        fdr[k] = _nan_safe(fdr[k], 1)

    return fdr


def error_rate(labels, calls):

    any_detection = calls.apply(lambda r: any(r), axis = 1)
    fn = labels*(1 - calls.values)
    errors = fn*any_detection.values.reshape(-1, 1)
    error_rate = errors.sum()/labels.sum()
    error_rate['total'] = errors.sum().sum()/labels.sum().sum()
    
    return dict(error_rate)


def alt_discovery_rate(labels, calls):
    
    all_alt_scores = []
    for pt in labels.columns:
        labels_for_pt = labels[calls[pt].astype('bool')]
        alt_rate = labels_for_pt.sum()/labels_for_pt.shape[0]
        all_alt_scores.append((pt, alt_rate))
        
    adr = {}
    for pt, s in all_alt_scores:
        adr[pt] = {}
        for k, v in s.items():
            adr[pt][f'{k}_adr'] = _nan_safe(v, 1/(labels.shape[1] - 1))
        
    return adr


def rocs(labels, scores):

    macro = roc_auc_score(labels, scores, average = 'macro')

    roc_vals = {}
    for pt in labels.columns:
        roc_vals[pt] = {}
        fpr, tpr, _ = roc_curve(labels[pt], scores[pt])
        roc_vals[pt]['tpr'] = tpr.tolist()
        roc_vals[pt]['fpr'] = fpr.tolist()
        roc_vals[pt]['auc'] = auc(fpr, tpr)

    return {
        **roc_vals,
        'total' : {
            'auc' : macro
        }
    }


def report_metrics(labels, calls, scores):

    sens = sensitivity(labels, calls)
    prec = precision(labels, calls)
    fdr = false_discovery_rate(labels, calls)
    er = error_rate(labels, calls)
    adr = alt_discovery_rate(labels, calls)
    roc = rocs(labels, scores)

    metrics = {}

    for k in sens:
        metrics[k] = {}
        metrics[k]['sens'] = sens[k]
        metrics[k]['prec'] = prec[k]
        metrics[k]['fdr'] = fdr[k]
        metrics[k]['err'] = er[k]
        if k in labels.columns:
            metrics[k]['roc_tpr'] = roc[k]['tpr']
            metrics[k]['roc_fpr'] = roc[k]['fpr']
        metrics[k]['roc_auc'] = roc[k]['auc']

    for k in adr:
        metrics[k] = {
            **metrics[k],
            **adr[k]
        }
    
    return metrics


def score_holdout(graph, n_per_class, top_n, threshold):
    
    nodes = np.array(graph.nodes)
    np.random.shuffle(nodes)
    
    c = defaultdict(int)
    subset = []
    
    for node in nodes:
        row = graph.support_labels.data.loc[node, graph.support_labels.binary_annotation_cols]
        col = np.random.choice(row.index[np.argwhere(row.values).flatten()])
        if c[col] < n_per_class:
            subset.append(node)
            c[col] += 1
            
    sampled_graph = graph.drop_support_nodes(subset, embed_dropped = True)
    
    calls = get_calls(sampled_graph, top_n = top_n, nodes = subset, threshold = threshold)
    scores = sampled_graph.neighbor_votes(metric = 'odds_ratio', nodes = subset)
    labels = graph.support_labels.data.loc[subset, graph.support_labels.binary_annotation_cols]
    
    metrics = report_metrics(labels, calls, scores)
    
    return metrics
