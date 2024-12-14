#flight_network.py
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import pandas as pd

class FlightNetwork:
    def __init__(self):
        self.nodes: Set[str] = set()
        self.edges: Set[Tuple[str, str]] = set()
        self.in_edges: Dict[str, List[str]] = defaultdict(list)
        self.out_edges: Dict[str, List[str]] = defaultdict(list)

    def add_nodes_and_edges(self, origin_airports: pd.Series, destination_airports: pd.Series) -> None:
        self.nodes.update(origin_airports)
        self.nodes.update(destination_airports)
        self.edges.update(zip(origin_airports, destination_airports))
        
        for origin, destination in zip(origin_airports, destination_airports):
            self.in_edges[destination].append(origin)
            self.out_edges[origin].append(destination)

    def in_degree(self, node: str) -> int:
        return len(self.in_edges[node])

    def out_degree(self, node: str) -> int:
        return len(self.out_edges[node])
