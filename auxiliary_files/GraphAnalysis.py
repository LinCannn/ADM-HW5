import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
from typing import Dict, List, Tuple
from auxiliary_files.flight_network import FlightNetwork

class GraphAnalysis:
    def __init__(self, network: FlightNetwork):
        
        self.graph = network.graph

    def calculate_graph_density(self, n_nodes: int, n_edges: int) -> float:
        
        if n_nodes <= 1:
            return 0.0
        return (2 * n_edges) / (n_nodes * (n_nodes - 1))

    def get_degree_metrics(self) -> Tuple[Dict[str, int], Dict[str, int]]:
        
        in_degrees = dict(self.graph.in_degree())
        out_degrees = dict(self.graph.out_degree())
        return in_degrees, out_degrees

    def identify_hub_airports(self, percentile: float = 90) -> List[str]:
        
        degrees = dict(self.graph.degree())
        threshold = np.percentile(list(degrees.values()), percentile)
        return [node for node, degree in degrees.items() if degree > threshold]

    def create_hubs_table(
        self, hubs: List[str], in_degrees: Dict[str, int], out_degrees: Dict[str, int]
    ) -> pd.DataFrame:
        
        total_degrees = {node: in_degrees.get(node, 0) + out_degrees.get(node, 0) for node in hubs}
        hubs_table = pd.DataFrame({
            "Airport": hubs,
            "In-degree": [in_degrees.get(node, 0) for node in hubs],
            "Out-degree": [out_degrees.get(node, 0) for node in hubs],
            "Total degree": [total_degrees[node] for node in hubs],
        }).sort_values(by="Total degree", ascending=False)
        return hubs_table

    def plot_degree_distribution(self, degrees: List[int], degree_type="Degree", max_degree_limit=500):
        
        degrees = [degree for degree in degrees if degree <= max_degree_limit]

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.histplot(degrees, kde=True, color='skyblue', ax=ax)

        ax.set_xlim(0, max_degree_limit)
        ax.set_title(f'{degree_type} Distribution')
        ax.set_xlabel(f'{degree_type}')
        ax.set_ylabel('Frequency')
        ax.grid(True)

        return fig

    def get_image_from_figure(self, fig) -> str:
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{image_base64}"
