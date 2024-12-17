import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
import networkx as nx
from typing import Dict, List, Tuple

import networkx as nx
from typing import List, Tuple
import matplotlib.pyplot as plt

class FlightNetwork:

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_nodes_and_edges(
        self,
        origin_airports: pd.Series,
        destination_airports: pd.Series,
        distances: pd.Series = None
    ) -> None:
        """
        Add nodes and edges. Optionally, distances are included as edge weights.
        """
        self.graph.add_nodes_from(origin_airports.unique())
        self.graph.add_nodes_from(destination_airports.unique())
        
        if distances is not None:
            edges = list(zip(origin_airports, destination_airports, distances))
            for origin, destination, distance in edges:
                self.graph.add_edge(origin, destination, distance=distance)
        else:
            edges = list(zip(origin_airports, destination_airports))
            self.graph.add_edges_from(edges)

    @property
    def nodes(self) -> List[str]:
        return list(self.graph.nodes())

    @property
    def edges(self) -> List[tuple]:
        return list(self.graph.edges(data=True))  # Include edge attributes

    def disconnect_flight_network(self) -> Tuple[List[Tuple[str, str]], Tuple[set, set]]:
        """
        Finds and removes the minimum number of flights to disconnect the flight network into two subgraphs.
    
        Returns:
            - removed_edges: List of edges removed to disconnect the graph.
            - partitions: Two sets of nodes representing the two disjoint subgraphs.
        """
        # Convert the directed graph to an undirected graph for min-cut
        undirected_graph = self.graph.to_undirected()

        # Ensure all edges have a finite capacity
        for u, v in undirected_graph.edges:
            if 'capacity' not in undirected_graph[u][v]:
                undirected_graph[u][v]['capacity'] = 1  # Assign a default capacity
        # Select two arbitrary nodes as source and sink
        nodes = list(self.graph.nodes)
        if len(nodes) < 2:
            raise ValueError("The graph must have at least two nodes to calculate a minimum cut.")
        s, t = nodes[0], nodes[1]  # Arbitrary choice of source and sink nodes

        # Compute the minimum cut
        cut_value, partition = nx.minimum_cut(undirected_graph, s, t)
        reachable, non_reachable = partition

        # Identify the edges crossing the cut
        cut_edges = []
        for u in reachable:
            for v in self.graph.neighbors(u):
                if v in non_reachable:
                    cut_edges.append((u, v))

        # Remove the cut edges from the graph
        self.graph.remove_edges_from(cut_edges)

        return cut_edges, (reachable, non_reachable)


    def visualize_graph(self, title: str, partitions: Tuple[set, set] = None) -> None:
        """
        Visualizza il grafo con un layout e colorazione opzionale.
    
        Args:
        title: Titolo del grafo.
        partitions: Tuple opzionale contenente due insiemi di nodi da colorare.
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        #Usa le distanze come pesi per il layout
        edge_weights = nx.get_edge_attributes(self.graph, 'distance')
        pos = nx.spring_layout(self.graph, seed=42, weight='distance') if edge_weights else nx.spring_layout(self.graph, seed=42)

        if partitions:
            colors = ['red', 'blue']
            for i, partition in enumerate(partitions):
                nx.draw_networkx_nodes(self.graph, pos, nodelist=list(partition), node_color=colors[i], alpha=0.8)
        else:
            nx.draw_networkx_nodes(self.graph, pos, node_color='lightblue', alpha=0.8)

        # Disegna gli archi
        nx.draw_networkx_edges(self.graph, pos, edge_color='gray', alpha=0.7)

        # Aggiungi etichette ai nodi
        nx.draw_networkx_labels(self.graph, pos, font_size=10, font_color='black')

        # Aggiungi le distanze come etichette degli archi
        edge_labels = nx.get_edge_attributes(self.graph, 'distance')  # Ottieni l'attributo 'distance'
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=9)
        plt.title(title)
        plt.show()



