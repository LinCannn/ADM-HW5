import pandas as pd
import networkx as nx
from typing import List, Tuple

class FlightNetwork:
    

    def __init__(self):
        
        self.graph = nx.DiGraph()

    def add_nodes_and_edges(
        self,
        origin_airports: pd.Series,
        destination_airports: pd.Series,
        distances: pd.Series = None
    ) -> None:
        
        
        all_airports = set(origin_airports).union(set(destination_airports))
        self.graph.add_nodes_from(all_airports)

        
        if distances is not None:
            edges = zip(origin_airports, destination_airports, distances)
            for origin, destination, distance in edges:
                self.graph.add_edge(origin, destination, distance=distance)
        else:
            edges = zip(origin_airports, destination_airports)
            self.graph.add_edges_from(edges)

    @property
    def nodes(self) -> List[str]:
        
        # Usa il set combinato dei nodi per maggiore robustezza
        all_airports = set(self.graph.nodes)
        return list(all_airports)

    @property
    def edges(self) -> List[Tuple[str, str, dict]]:
        
        return list(self.graph.edges(data=True))
