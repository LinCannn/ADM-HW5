import pandas as pd
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_set, explode, array
from typing import List, Tuple
from collections import defaultdict
import itertools

def find_connected_components(csv_path: str, start_date: str, end_date: str) -> Tuple[int, List[int], str]:
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("ConnectedComponents") \
        .getOrCreate()

    # Read CSV file into Spark DataFrame
    df = spark.read.csv(csv_path, header=True)
    
    # Filter by date range
    df = df.filter((df.Fly_date >= start_date) & (df.Fly_date <= end_date))
    df = df.select("Origin_airport", "Destination_airport", "Fly_date")

    # Function to emit edges in both directions
    def emit_bidirectional_edges(row):
        origin = row.Origin_airport
        dest = row.Destination_airport
        return [(origin, dest), (dest, origin)]

    # Convert to RDD and create bidirectional edges
    edges_rdd = df.rdd.flatMap(emit_bidirectional_edges).distinct()

    # Function to update component IDs
    def update_components(edges):
        # Create local UnionFind structure
        components = {}
        
        def find(node):
            if node not in components:
                components[node] = node
            if components[node] != node:
                components[node] = find(components[node])
            return components[node]
        
        def union(node1, node2):
            root1, root2 = find(node1), find(node2)
            if root1 != root2:
                components[root2] = root1
        
        # Process edges
        for origin, dest in edges:
            union(origin, dest)
        
        # Emit (node, component_id) pairs
        return [(node, find(node)) for node in components.keys()]

    # Initial component assignment
    components_rdd = edges_rdd.mapPartitions(lambda x: update_components(x))

    # Iteratively update components until convergence
    prev_components = None
    curr_components = components_rdd.collectAsMap()
    
    while prev_components != curr_components:
        prev_components = curr_components
        
        # Update component assignments
        components_rdd = edges_rdd.map(
            lambda edge: (edge[1], curr_components.get(edge[0]))
        ).filter(
            lambda x: x[1] is not None
        ).reduceByKey(
            lambda x, y: min(x, y)
        )
        
        curr_components = components_rdd.collectAsMap()

    # Convert final results to DataFrame for analysis
    final_components_df = spark.createDataFrame(
        components_rdd.map(lambda x: (x[0], x[1])).collect(),
        ["node", "component_id"]
    )

    # Calculate component sizes
    component_sizes = final_components_df.groupBy("component_id").count().collect()
    sizes = [row["count"] for row in component_sizes]
    
    # Find largest component
    largest_component_id = max(component_sizes, key=lambda x: x["count"])["component_id"]
    largest_component = final_components_df.filter(
        final_components_df.component_id == largest_component_id
    ).select("node").rdd.map(lambda x: x[0]).collect()

    # Cleanup
    spark.stop()

    return len(component_sizes), sizes, largest_component

