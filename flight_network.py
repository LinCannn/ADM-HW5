import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
import networkx as nx
from typing import Dict, List, Tuple

class FlightNetwork:

    
    def __init__(self):
        
        self.graph = nx.DiGraph()

    def add_nodes_and_edges(self, origin_airports: pd.Series, destination_airports: pd.Series) -> None:
        
        self.graph.add_nodes_from(origin_airports.unique())
        self.graph.add_nodes_from(destination_airports.unique())
        edges = list(zip(origin_airports, destination_airports))
        self.graph.add_edges_from(edges)

    @property
    def nodes(self) -> List[str]:
        
        return list(self.graph.nodes())

    @property
    def edges(self) -> List[tuple]:
       
        return list(self.graph.edges())

