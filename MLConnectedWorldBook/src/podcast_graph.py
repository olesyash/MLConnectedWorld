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


def get_graph_with_credit_info(
    g_movies_and_keywords: nx.Graph,
    df_credits: pd.DataFrame,
    top_n_cast: int = 20,
) -> nx.MultiGraph:

    credit_edges = []
    # Process credits to extract relevant information
    for _, row in tqdm(
        df_credits.iterrows(), total=len(df_credits), desc="Processing credits"
    ):
        cast = json.loads(row.cast)
        for c in cast:
            if c["order"] > top_n_cast:
                continue
            credit_edges.append(
                {
                    "MOVIE": row["title"],
                    "PERSON": c["name"],
                    "type": "PARTICIPATED_IN",
                }
            )
        crew = json.loads(row.crew)
        for c in crew:
            job = c["job"]
            if job not in ["Director", "Producer", "Writer", "Screenplay"]:
                continue
            credit_edges.append(
                {"MOVIE": row["title"], "PERSON": c["name"], "type": "WORKED_ON"}
            )

    df_credit_edges = pd.DataFrame(credit_edges)

    g_multi = nx.MultiGraph()

    # Preserve node attributes including type
    for node, data in g_movies_and_keywords.nodes(data=True):
        g_multi.add_node(node, **data)

    for u, v, data in g_movies_and_keywords.edges(data=True):
        g_multi.add_edge(u, v, key="HAS_KEYWORD", **data)

    # Add the edges to the graph, and add nodes if they don't exist
    for _, row in tqdm(
        df_credit_edges.iterrows(), total=len(df_credit_edges), desc="Adding edges"
    ):
        movie = row["MOVIE"]
        person = row["PERSON"]
        edge_type = row["type"]

        if movie not in g_multi.nodes:
            g_multi.add_node(movie, type="MOVIE")
        if person not in g_multi.nodes:
            g_multi.add_node(person, type="PERSON")

        # Check if the edge already exists with the same type before adding
        if not g_multi.has_edge(movie, person, key=edge_type):
            g_multi.add_edge(movie, person, key=edge_type)

    return g_multi


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



dir_data = os.path.abspath("C:\olesya\MLConnectedWorld\MLConnectedWorldBook\data")
fn = os.path.join(dir_data, "podcasts.json")
df = pd.read_json(fn, orient="records", lines=True)
G = build_podcast_and_keywords_graph(df = df)