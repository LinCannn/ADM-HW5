import pandas as pd
import heapq
from collections import defaultdict

class FlightNetwork:
    def __init__(self):
        self.graph = defaultdict(list)  

    def add_flight(self, origin, destination, distance):
        self.graph[origin].append((destination, distance))

    def dijkstra(self, start, end):
        heap = [(0, start, [start])] 
        visited = set()

        while heap:
            current_distance, current_node, path = heapq.heappop(heap)

            if current_node in visited:
                continue
            visited.add(current_node)

            if current_node == end:
                return path, current_distance  

            for neighbor, weight in self.graph[current_node]:
                if neighbor not in visited:
                    heapq.heappush(heap, (current_distance + weight, neighbor, path + [neighbor]))

        return "No route found", float('inf')

def load_and_build_network(file_path, date):

    df = pd.read_csv(file_path)

    required_columns = {'Fly_date', 'Origin_city', 'Destination_city', 'Origin_airport', 'Destination_airport', 'Distance'}
    if not required_columns.issubset(set(df.columns)):
        raise KeyError(f"Dataset is missing required columns. Found columns: {df.columns}")

    df['Fly_date'] = pd.to_datetime(df['Fly_date']).dt.date

    df_date = df[df['Fly_date'] == pd.to_datetime(date).date()]
    print(f"Number of flights on {date}: {len(df_date)}")

    if df_date.empty:
        raise ValueError(f"No flights found for the date {date}.")

    network = FlightNetwork()

    for _, row in df_date.iterrows():
        network.add_flight(row['Origin_airport'], row['Destination_airport'], row['Distance'])

    return network, df


def find_best_routes(network, df, origin_city, destination_city):

    origin_airports = df[df['Origin_city'] == origin_city]['Origin_airport'].unique()
    destination_airports = df[df['Destination_city'] == destination_city]['Destination_airport'].unique()

    results = []

    for origin_airport in origin_airports:
        for destination_airport in destination_airports:
            path, distance = network.dijkstra(origin_airport, destination_airport)
            results.append({
                'Origin_city_airport': origin_airport,
                'Destination_city_airport': destination_airport,
                'Best_route': ' → '.join(path) if isinstance(path, list) else path
            })
    return pd.DataFrame(results)