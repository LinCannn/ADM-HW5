import pandas as pd
import networkx as nx
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import Row
from typing import List, Tuple


conf = SparkConf().setAppName("FlightNetworkConnectedComponents").setMaster("local[*]")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)


class ConnectedComponents():

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_nodes_and_edges(
        self,
        origin_airports: pd.Series,
        destination_airports: pd.Series,
    ) -> None:
        """
        Add nodes and edges.Optionally, distances are included as edge weights.
        """

        airports = set(pd.concat([origin_airports, destination_airports]).unique())
        self.graph.add_nodes_from(airports)
        edges = list(zip(origin_airports, destination_airports))
        self.graph.add_edges_from(edges)


