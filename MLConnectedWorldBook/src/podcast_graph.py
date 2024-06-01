import json
from collections import defaultdict
from datetime import timedelta

import networkx as nx
import numpy as np
import pandas as pd
from cachier import cachier
from yake import yake
from tqdm.auto import tqdm
import os

DESCRIPTION = "description_translated"
TITLE = "title_transliterated"
PODCAST = "PODCAST"


@cachier(stale_after=timedelta(days=1))
def build_podcast_and_keywords_graph(
    df: pd.DataFrame,
    n_grams: int = 2,
    top: int = 10,
    node_type1: str = PODCAST,
) -> nx.Graph:

    kw_extractor = yake.KeywordExtractor(
        lan="en",
        n=n_grams,  # up to n-grams
        dedupLim=0.9,  # two words are considered the same if the similarity is higher than this
        dedupFunc="seqm",  # similarity function. Options are seqm, jaccard, cosine, levenshtein, jaro, jaro_winkler, monge_elkan, keywordset
        windowsSize=1,  # number of words in the window
        top=top,  # number of keywords to extract
        features=None,  # list of features to extract. If None, it uses all the features. Options are: ngram, sif, first, freq
    )
    graph = nx.Graph()
    for _, row in tqdm(
        df.iterrows(), total=len(df), desc="Building graph", leave=False
    ):
        title = row[TITLE]
        description = row[DESCRIPTION]
        if not description:
            continue
        overview = str(description).lower()
        keywords = kw_extractor.extract_keywords(overview)
        if title not in graph:
            graph.add_node(
                title, label=title, type=node_type1, count=1
            )
        else:
            graph.nodes[title]["count"] += 1
        for keyword, _ in keywords:
            if keyword not in graph:
                graph.add_node(
                    keyword, label=keyword, type="KEYWORD", count=1
                )
            else:
                graph.nodes[keyword]["count"] += 1

            weight = description.lower().count(keyword)
            if graph.has_edge(title, keyword):
                graph.edges[title, keyword]["weight"] += weight
            else:
                graph.add_edge(title, keyword, weight=weight)
    return graph




def rank_keywords(graph):
    keyword_weights = defaultdict(int)

    # Iterate through the edges
    for podcast, keyword, weight in graph.edges(data=True):
        keyword_weights[keyword] += weight["weight"]

    # Sort the dictionary by values (sum of weights) in descending order
    ranked_keywords = sorted(keyword_weights.items(), key=lambda x: x[1], reverse=True)

    return ranked_keywords

def get_smallest_connected_component_graph(graph: nx.Graph) -> nx.Graph:
    largest_connected_component = min(nx.connected_components(graph), key=len)
    return graph.subgraph(largest_connected_component).copy()

def get_connected_component_graph(graph: nx.Graph, size_min, size_max) -> nx.Graph:
    connected_components = nx.connected_components(graph)
    for component in connected_components:
        if len(component) < size_max and len(component) > size_min:
            return graph.subgraph(component).copy()
    largest_connected_component = max(nx.connected_components(graph), key=len)
    return graph.subgraph(largest_connected_component).copy()


def remove_isolated_nodes(graph: nx.Graph):
    nodes_to_remove = [n for n in graph.nodes if graph.degree(n) == 0]
    ret = graph.copy()
    ret.remove_nodes_from(nodes_to_remove)
    return ret


def get_largest_connected_component_graph(graph: nx.Graph) -> nx.Graph:
    largest_connected_component = max(nx.connected_components(graph), key=len)
    return graph.subgraph(largest_connected_component).copy()


def cleanup_nodes_by_percentile(
    graph: nx.Graph,
    node_attr: str = "count",
    percentile: float = 1.0,
    node_type: str = None,
):
    if node_type:
        nodes = [n for n, d in graph.nodes(data=True) if d["type"] == node_type]
    else:
        nodes = list(graph.nodes)
    counts = np.array([graph.nodes[n][node_attr] for n in nodes])
    threshold = np.percentile(counts, 100 - percentile)
    nodes_to_remove = [n for n in nodes if graph.nodes[n][node_attr] < threshold]
    ret = graph.copy()
    ret.remove_nodes_from(nodes_to_remove)
    return ret

