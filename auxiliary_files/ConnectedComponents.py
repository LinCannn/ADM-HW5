from collections import defaultdict  # Ensure defaultdict is imported

import pandas as pd
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from typing import List, Tuple

class UnionFind:
    def __init__(self, nodes):
        # Initialize Union-Find data structure
        self.parent = {node: node for node in nodes}
        self.rank = {node: 0 for node in nodes}
        self.size = {node: 1 for node in nodes}
    
    def find(self, node):
        # Find with path compression
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]
    
    def union(self, node1, node2):
        # Union by rank
        root1 = self.find(node1)
        root2 = self.find(node2)
        
        if root1 != root2:
            # Union by rank
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
                self.size[root1] += self.size[root2]
            elif self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
                self.size[root2] += self.size[root1]
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1
                self.size[root1] += self.size[root2]
    
    def get_components(self):
        # Get connected components
        components = defaultdict(list)
        for node in self.parent:
            root = self.find(node)
            components[root].append(node)
        return components

def find_connected_components(csv_path: str, start_date: str, end_date: str) -> Tuple[int, List[int], str]:
    # Initialize Spark session
    spark = SparkSession.builder.appName("UnionFindParallel").getOrCreate()
    
    # Load the dataset
    df = pd.read_csv(csv_path)
    
    # Filter by date range
    df = df[(df['Fly_date'] >= start_date) & (df['Fly_date'] <= end_date)]
    df = df[['Origin_airport', 'Destination_airport', 'Fly_date']]
    
    # Extract nodes (airports)
    nodes = list(pd.concat([df['Origin_airport'], df['Destination_airport']]).unique())
    print("Total nodes: ", len(nodes))
    
    # Create edges (undirected graph: Origin to Destination and Destination to Origin)
    edges = list(set(edge for edge in zip(df['Origin_airport'], df['Destination_airport'])))
    print("Total edges: ", len(edges))
    
    # Initialize UnionFind for each node
    uf = UnionFind(nodes)
    
    # Function to apply union operation to pairs
    def union_op(edge):
        origin, destination = edge
        uf.union(origin, destination)
        return (origin, uf.find(origin)), (destination, uf.find(destination))
    
    # Apply union operation in parallel using map
    rdd = spark.sparkContext.parallelize(edges)
    union_rdd = rdd.flatMap(union_op)
    
    # Collect the results and merge them
    result = union_rdd.collect()
    
    # Create a dictionary of parent and sizes for each node
    parent_dict = {}
    for origin, parent in result:
        parent_dict[origin] = parent
    
    # Ensure all nodes are added to the UnionFind even if they aren't part of any edges
    for node in nodes:
        if node not in parent_dict:
            parent_dict[node] = node
    
    # Get the connected components
    components = defaultdict(list)
    for node, parent in parent_dict.items():
        components[parent].append(node)
    
    # Number of connected components
    num_components = len(components)
    
    # Sizes of each connected component
    component_sizes = [len(component) for component in components.values()]
    
    # Find the largest connected component
    largest_component = max(components.values(), key=len)
    
    # Stop the Spark session
    spark.stop()
    
    # Return the results
    return num_components, component_sizes, largest_component
