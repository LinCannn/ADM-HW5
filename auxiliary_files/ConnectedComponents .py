
import time
from collections import defaultdict
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from graphframes import GraphFrame
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

@dataclass
class TimingMetrics:
    setup_time: float = 0
    data_loading_time: float = 0
    processing_time: float = 0
    component_analysis_time: float = 0
    total_time: float = 0

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    return wrapper


class UnionFind:
    def __init__(self, nodes):
        self.parent = {node: node for node in nodes}
        self.rank = {node: 0 for node in nodes}
        self.size = {node: 1 for node in nodes}
    
    def find(self, node):
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]
    
    def union(self, node1, node2):
        root1 = self.find(node1)
        root2 = self.find(node2)
        
        if root1 != root2:
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
        components = defaultdict(list)
        for node in self.parent:
            root = self.find(node)
            components[root].append(node)
        return components


def find_connected_components_with_timing(csv_path: str, start_date: str, end_date: str) -> Dict[str, Any]:
    metrics = TimingMetrics()
    total_start = time.time()

    # Setup timing
    setup_start = time.time()
    
    spark = SparkSession.builder \
        .appName("UnionFindParallel") \
        .getOrCreate()
    
    spark.sparkContext.addFile('/content/drive/MyDrive/ADM-HW5/auxiliary_files', recursive=True)
    df = pd.read_csv(csv_path)
    metrics.setup_time = time.time() - setup_start

    # Data loading timing
    loading_start = time.time()
    df = pd.read_csv(csv_path)
    df = df[(df['Fly_date'] >= start_date) & (df['Fly_date'] <= end_date)]
    df = df[['Origin_airport', 'Destination_airport', 'Fly_date']]
    metrics.data_loading_time = time.time() - loading_start

    # Extract nodes and create edges
    nodes = list(pd.concat([df['Origin_airport'], df['Destination_airport']]).unique())
    edges = list(set(edge for edge in zip(df['Origin_airport'], df['Destination_airport'])))

    # Initialize UnionFind
    uf = UnionFind(nodes)

    # Processing timing
    processing_start = time.time()
    def union_op(edge):
        origin, destination = edge
        uf.union(origin, destination)
        return (origin, uf.find(origin)), (destination, uf.find(destination))

    # Apply union operation in parallel
    rdd = spark.sparkContext.parallelize(edges)
    union_rdd = rdd.flatMap(union_op)
    
    # Collect and process results
    result = union_rdd.collect()
    parent_dict = {}
    for origin, parent in result:
        parent_dict[origin] = parent

    for node in nodes:
        if node not in parent_dict:
            parent_dict[node] = node

    components = defaultdict(list)
    for node, parent in parent_dict.items():
        components[parent].append(node)

    metrics.processing_time = time.time() - processing_start

    # Component analysis timing
    analysis_start = time.time()
    component_sizes = [len(component) for component in components.values()]
    largest_component = max(components.values(), key=len)
    metrics.component_analysis_time = time.time() - analysis_start

    metrics.total_time = time.time() - total_start

    result = {
        'number_of_components': len(components),
        'component_sizes': component_sizes,
        'largest_component_size': len(largest_component),
        'largest_component_airports': largest_component,
        'timing_metrics': {
            'setup_time': metrics.setup_time,
            'data_loading_time': metrics.data_loading_time,
            'processing_time': metrics.processing_time,
            'component_analysis_time': metrics.component_analysis_time,
            'total_time': metrics.total_time
        }
    }

    spark.stop()
    return result


def find_connected_components_graph_with_timing(csv_path: str, start_date: str, end_date: str) -> Dict[str, Any]:
    metrics = TimingMetrics()
    total_start = time.time()

    # Setup timing
    setup_start = time.time()
    spark = SparkSession.builder \
        .appName("GraphFramesConnectedComponents") \
        .config("spark.jars.packages", "graphframes:graphframes-0.8.3-spark3.5-s_2.12") \
        .getOrCreate()
    spark.sparkContext.setCheckpointDir("/tmp/checkpoints")
    spark.sparkContext.addFile('/content/drive/MyDrive/ADM-HW5/auxiliary_files', recursive=True)
    df = pd.read_csv(csv_path)
    metrics.setup_time = time.time() - setup_start

    # Data loading timing
    loading_start = time.time()
    df = pd.read_csv(csv_path)
    df = df[(df['Fly_date'] >= start_date) & (df['Fly_date'] <= end_date)]
    df = df[['Origin_airport', 'Destination_airport', 'Fly_date']]
    metrics.data_loading_time = time.time() - loading_start

    # Create vertices (airports) and edges (flights)
    vertices = pd.concat([df['Origin_airport'], df['Destination_airport']]).unique()
    vertices_df = spark.createDataFrame([(v,) for v in vertices], ['id'])
    
    edges_df = spark.createDataFrame(
        [(row['Origin_airport'], row['Destination_airport']) for _, row in df.iterrows()],
        ['src', 'dst']
    )
    
    # Processing timing
    processing_start = time.time()
    g = GraphFrame(vertices_df, edges_df)
    result = g.connectedComponents()
    metrics.processing_time = time.time() - processing_start

    # Component analysis timing
    analysis_start = time.time()
    component_sizes = result.groupBy('component').count().collect()
    sizes = [size['count'] for size in component_sizes]
    largest_component_id = max(component_sizes, key=lambda x: x['count'])['component']
    largest_component_airports = result.filter(result['component'] == largest_component_id) \
                                        .select('id').rdd.flatMap(lambda x: x).collect()
    metrics.component_analysis_time = time.time() - analysis_start

    metrics.total_time = time.time() - total_start

    result = {
        'number_of_components': len(component_sizes),
        'component_sizes': sizes,
        'largest_component_size': max(sizes),
        'largest_component_airports': largest_component_airports,
        'timing_metrics': {
            'setup_time': metrics.setup_time,
            'data_loading_time': metrics.data_loading_time,
            'processing_time': metrics.processing_time,
            'component_analysis_time': metrics.component_analysis_time,
            'total_time': metrics.total_time
        }
    }

    spark.stop()
    return result

