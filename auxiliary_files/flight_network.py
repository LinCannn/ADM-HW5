import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
import networkx as nx
from typing import Dict, List, Tuple, Set
from collections import deque

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
                self.graph.add_edge(origin, destination, distance=distance,capacity=1)
        else:
            edges = list(zip(origin_airports, destination_airports))
            for origin, destination, distance in edges:
                self.graph.add_edge(origin, destination, distance=distance,capacity=1)


    @property
    def nodes(self) -> List[str]:
        return list(self.graph.nodes())

    @property
    def edges(self) -> List[tuple]:
        return list(self.graph.edges(data=True))  # Include edge attributes

    def bfs(self, residual_graph, source, sink, parent):
        """
        Executes a BFS to find an augmenting path in the residual graph.
        Writing this function is a prerequisite for using min_cut.
        """
        visited = set()
        queue = deque([source])
        visited.add(source)
        parent.clear()

        while queue:
            current = queue.popleft()
            for neighbor in residual_graph[current]:
                capacity = residual_graph[current][neighbor]["capacity"]
                if neighbor not in visited and capacity > 0:
                    queue.append(neighbor)
                    visited.add(neighbor)
                    parent[neighbor] = current
                    if neighbor == sink:
                        return True
        return False

    def min_cut(self, source: str, sink: str):
        """
        This manual implementation of min_cut uses the Edmonds-Karp method.
        It gives back removed edges and nodes partitions.
        """
        # Create a copy of the original graph for the residual one
        residual_graph = nx.DiGraph()
        for u, v, data in self.graph.edges(data=True):
            residual_graph.add_edge(u, v, capacity=data["capacity"])
            if not residual_graph.has_edge(v, u):
                residual_graph.add_edge(v, u, capacity=0)

        parent = {}
        max_flow = 0

        # Step 1: Find te maximum flow
        while self.bfs(residual_graph, source, sink, parent):
            # Find the minimum residual capacity over the path
            path_flow = float("Inf")
            v = sink
            while v != source:
                u = parent[v]
                path_flow = min(path_flow, residual_graph[u][v]["capacity"])
                v = u

            # Update the capacities in the residual graph
            v = sink
            while v != source:
                u = parent[v]
                residual_graph[u][v]["capacity"] -= path_flow
                residual_graph[v][u]["capacity"] += path_flow
                v = u

            max_flow += path_flow

        # Step 2: Find the minimum cut
        visited = set()
        queue = deque([source])
        visited.add(source)

        while queue:
            current = queue.popleft()
            for neighbor in residual_graph[current]:
                if neighbor not in visited and residual_graph[current][neighbor]["capacity"] > 0:
                    queue.append(neighbor)
                    visited.add(neighbor)

        # Partitions
        reachable = visited
        non_reachable = set(self.graph.nodes) - reachable

        # Identify the cut edges
        cut_edges = []
        for u in reachable:
            for v in self.graph.successors(u):
                if v in non_reachable:
                    cut_edges.append((u, v))

        return cut_edges, (reachable, non_reachable)
    
    #FOR RQ4:

    def add_nodes_and_edges2(
        self,
        origin_airports: pd.Series,
        destination_airports: pd.Series
    ) -> None:
        """
        Add nodes and edges, this time putting capacity=1 by default.
        """
        self.graph.add_nodes_from(origin_airports.unique())
        self.graph.add_nodes_from(destination_airports.unique())
        
        edges = list(zip(origin_airports, destination_airports))
    
        # Add edges to the graph with capacity = 1
        for u, v in edges:
            self.graph.add_edge(u, v, capacity=1)





