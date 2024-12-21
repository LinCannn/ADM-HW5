import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import heapq
import networkx as nx
from typing import Dict, List, Tuple
import tqdm

class FlightNetwork:

    def __init__(self):
        self.graph = nx.DiGraph()
        self.all_shortest_path = {}
        self.centrality_measure = None

    def add_nodes_and_edges(
        self,
        origin_airports: pd.Series,
        destination_airports: pd.Series,
        distances: pd.Series = None
    ) -> None:
        """
        Add nodes and edges.Optionally, distances are included as edge weights.
        """

        airports = set(pd.concat([origin_airports, destination_airports]).unique())
        
        self.graph.add_nodes_from(airports)
        
        
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

    

    def get_all_shortest_path(self):
        """
        Computes the shortest paths between all pairs of nodes using Dijkstra's algorithm.
        For each source node, it calculates the distances and paths to all other nodes.
        
        Returns:
            all_shortest_path (dict): A dictionary where each source node has a sub-dictionary
                                    with 'distances' (shortest distances) and 'paths' 
                                    (actual shortest paths to other nodes).
        """
        all_shortest_path = {}
        airports = self.nodes  # Get all nodes in the graph

        # Compute shortest paths for each source node
        for source in airports:
            # Initialize distances and paths
            distances = {node: float('inf') for node in airports}
            paths = {node: [] for node in airports}
            distances[source] = 0
            paths[source] = [source]

            # Priority queue for Dijkstra's algorithm
            queue = [(0, source)]  # (distance, node)

            while queue:
                current_distance, current_node = heapq.heappop(queue)

                # Skip if we already found a shorter path
                if current_distance > distances[current_node]:
                    continue

                # Iterate over all neighbors of the current node
                for adjacent in self.graph.successors(current_node):
                    # Fetch edge weight, defaulting to 1 for unweighted graphs
                    edge_weight = self.graph[current_node][adjacent].get('distance', 1)

                    total_distance = current_distance + edge_weight

                    # Relaxation step: Check if we found a shorter path
                    if total_distance < distances[adjacent]:
                        distances[adjacent] = total_distance
                        paths[adjacent] = paths[current_node] + [adjacent]
                        heapq.heappush(queue, (total_distance, adjacent))

            # Store results for the current source node

            all_shortest_path[source] = {
                'distances': distances, 
                'paths': paths
            }

        self.all_shortest_path = all_shortest_path
        return all_shortest_path


    def analyze_centrality(self, airport: str) -> Dict[str, float]:
        """
        Computes centrality measures for a given airport using precomputed shortest paths.
        
        Parameters:
        - airport (str): The specific airport for which to compute centrality measures.

        Returns:
        - Dict[str, float]: Dictionary of centrality measures.
        """

        if not self.all_shortest_path:
            raise ValueError("Shortest paths are not computed. Call get_all_shortest_path() first.")

        if airport not in self.nodes:
            raise ValueError(f"Airport '{airport}' not found in the network.")

            # Betweenness Centrality
        def compute_betweenness_centrality():
            betweenness = 0
            total_nodes = len(self.nodes)

            for src in self.all_shortest_path:
                for dest in self.all_shortest_path:
                    if src == dest or src == airport or dest == airport:
                        continue
                    # Check if airport is on the shortest path between src and dest
                    shortest_path = self.all_shortest_path[src]['paths'][dest]
                    if airport in shortest_path[1:-1]:  # Exclude src and dest
                        betweenness += 1

            # Normalize the betweenness score
            normalization_factor = (total_nodes - 1) * (total_nodes - 2)
            if normalization_factor > 0:
                betweenness /= normalization_factor

            return betweenness

        # Closeness Centrality
        def compute_closeness_centrality():
            distances = self.all_shortest_path[airport]['distances']
            total_distance = sum(dist for dist in distances.values() if dist < float('inf'))
            if total_distance == 0:
                return 0
            return (len(self.nodes) - 1) / total_distance

        # Degree Centrality
        def compute_degree_centrality():
            return len(list(self.graph.successors(airport))) / (len(self.nodes) - 1)

        # PageRank
        def compute_pagerank(alpha=0.85, iterations=100):
            pagerank = {node: 1 / len(self.nodes) for node in self.nodes}
            for _ in range(iterations):
                new_pagerank = {n: (1 - alpha) / len(self.nodes) for n in self.nodes}
                for node in self.nodes:
                    for neighbor in self.graph.successors(node):
                        new_pagerank[neighbor] += alpha * pagerank[node] / len(list(self.graph.successors(node)))
                pagerank = new_pagerank
            return pagerank[airport]

        # Compute all centrality measures
        centrality_measures = {
            "betweenness_centrality": compute_betweenness_centrality(),
            "closeness_centrality": compute_closeness_centrality(),
            "degree_centrality": compute_degree_centrality(),
            "pagerank": compute_pagerank(),
        }

        return centrality_measures

    def compare_centralities(self):
        """
        Compare centrality values for all nodes in the graph.
        Plot centrality distributions and return the top 5 airports for each measure.
        
        Parameters:
        - flight_network (FlightNetwork): An instance of the FlightNetwork class.
        
        Returns:
        - Dict[str, List[Tuple[str, float]]]: Top 5 airports for each centrality measure.
        """
        if not self.all_shortest_path:
            self.get_all_shortest_path()

        centrality_data = {
            "betweenness_centrality": {},
            "closeness_centrality": {},
            "degree_centrality": {},
            "pagerank": {},
        }

        for airport in tqdm.tqdm(self.nodes):
            centrality_measures = self.analyze_centrality(airport)
            for measure, value in centrality_measures.items():
                centrality_data[measure][airport] = value

        # Convert centrality data to a DataFrame for analysis
        centrality_df = pd.DataFrame(centrality_data)
                                    
        # Get top 5 airports for each centrality measure
        top_airports = {}
        for measure in centrality_df.columns:
            top_airports[measure] = (
                centrality_df[measure]
                .sort_values(ascending=False)
                .head(5)
                .reset_index()
                .to_dict(orient="records")
            )

        # Plot distributions for each centrality measure
        plt.figure(figsize=(16, 12))
        for i, measure in enumerate(centrality_df.columns, start=1):
            plt.subplot(2, 2, i)
            sns.histplot(centrality_df[measure], kde=True, bins=20)
            plt.title(f"Distribution of {measure.replace('_', ' ').title()}")
            plt.xlabel(measure.replace('_', ' ').title())
            plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

        self.centrality_measure = centrality_df
        return top_airports
