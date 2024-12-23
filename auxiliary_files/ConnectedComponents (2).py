import pandas as pd
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_set, explode, array
from typing import List, Tuple
from collections import defaultdict
import itertools


def find_connected_connections_spark(csv_path: str, start_date: str, end_date: str) -> dict:
    
    spark = SparkSession.builder \
        .appName("ConnectedComponents") \
        .getOrCreate()

    print(f"Analyzing flight network from {start_date} to {end_date}")
    print("Maximum iterations set to: 10")
    
    
    print("Loading and filtering flight network data...")
    df = spark.read.csv(csv_path, header=True)
    
    
    df = df.filter((df.Fly_date >= start_date) & (df.Fly_date <= end_date))
    df = df.select("Origin_airport", "Destination_airport", "Fly_date")
    
    
    total_flights = df.count()
    unique_airports = df.select("Origin_airport").distinct().union(df.select("Destination_airport").distinct()).distinct()
    total_airports = unique_airports.count()

    print(f"Total flights in the period: {total_flights}")
    print(f"Unique airports: {total_airports}")
    
    
    def emit_bidirectional_edges(row):
        origin = row.Origin_airport
        dest = row.Destination_airport
        return [(origin, dest), (dest, origin)]

    
    edges_rdd = df.rdd.flatMap(emit_bidirectional_edges).distinct()

    
    def update_components(edges):
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
        
        
        for origin, dest in edges:
            union(origin, dest)
        
        
        return [(node, find(node)) for node in components.keys()]

    
    components_rdd = edges_rdd.mapPartitions(lambda x: update_components(x))

    
    prev_components = None
    curr_components = components_rdd.collectAsMap()

    iteration = 1
    while prev_components != curr_components and iteration <= 10:  # Limita a 10 iterazioni
        print(f"Iteration {iteration}")
        prev_components = curr_components
        components_rdd = edges_rdd.map(
            lambda edge: (edge[1], curr_components.get(edge[0]))
        ).filter(
            lambda x: x[1] is not None
        ).reduceByKey(
            lambda x, y: min(x, y)
        )
        
        curr_components = components_rdd.collectAsMap()
        iteration += 1

    
    if prev_components == curr_components:
        print(f"Converged at iteration {iteration - 1}")
    else:
        print("Reached maximum iterations without convergence.")

    
    final_components_df = spark.createDataFrame(
        components_rdd.map(lambda x: (x[0], x[1])).collect(),
        ["node", "component_id"]
    )

    
    component_sizes = final_components_df.groupBy("component_id").count().collect()
    sizes = [row["count"] for row in component_sizes]
    
    
    largest_component_id = max(component_sizes, key=lambda x: x["count"])["component_id"]
    largest_component = final_components_df.filter(
        final_components_df.component_id == largest_component_id
    ).select("node").rdd.map(lambda x: x[0]).collect()

    
    result = {
        'number_of_components': len(component_sizes),
        'component_sizes': sizes,
        'largest_component_size': max(sizes)
    }

    
    spark.stop()

    return result
def process_in_parallel(csv_path: str, start_date: str, end_date: str, num_partitions: int = None):
    
    try:
        # Configure Spark with more memory and cores if needed
        conf = SparkConf().setAppName("ParallelConnectedComponents")
        if num_partitions:
            conf = conf.set("spark.default.parallelism", str(num_partitions))
        
        # Create Spark context
        sc = SparkContext.getOrCreate(conf=conf)
        
        # Run the analysis
        result = find_connected_connections_spark(csv_path, start_date, end_date)
        
        return result
        
    except Exception as e:
        print(f"Error in parallel processing: {str(e)}")
        raise
    finally:
        # Clean up Spark context
        sc.stop()

