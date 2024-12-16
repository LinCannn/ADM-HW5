import sys
import csv
import asyncio
from opencage.geocoder import OpenCageGeocode

API_KEY = 'e4451f9958ca4dfbabb459c61f922c3a'
INFILE = 'airports_without_coordinates.csv' 
OUTFILE = 'airport_codes_coordinates_.csv'  
MAX_ITEMS = 0  
NUM_WORKERS = 3  
RATE_LIMIT_DELAY = 1

# Write one geocoding result to the output CSV file
async def write_one_geocoding_result(geocoding_result, airport_code, csv_writer):
    if geocoding_result and len(geocoding_result) > 0:
        geocoding_result = geocoding_result[0]
        city = geocoding_result['components'].get('city', '')
        region = geocoding_result['components'].get('state', '')
        country = geocoding_result['components'].get('country', '')
        lat = geocoding_result['geometry']['lat']
        lng = geocoding_result['geometry']['lng']
        row = [
            airport_code,
            city,
            region,
            country,
            lat,
            lng
        ]
        sys.stderr.write(f"Found airport: {airport_code}, city: {city}, region: {region}, country: {country}, lat: {lat}, lon: {lng}\n")
    else:
        row = [
            airport_code,
            '',
            '',
            'USA',  
            0,
            0
        ]
        sys.stderr.write(f"Airport not found or geocoding error: {airport_code}\n")
    
    csv_writer.writerow(row)

# Geocode a single airport code
async def geocode_airport_code(airport_code, city_region, csv_writer):
    async with OpenCageGeocode(API_KEY) as geocoder:
        try:
            query = f"{city_region}, USA"  
            geocoding_result = await geocoder.geocode_async(query)
            await write_one_geocoding_result(geocoding_result, airport_code, csv_writer)
        except Exception as e:
            sys.stderr.write(f"Error geocoding airport code {airport_code}: {e}\n")

# Worker function to process tasks
async def run_worker(worker_name, queue, csv_writer):
    sys.stderr.write(f"Worker {worker_name} started...\n")
    while True:
        work_item = await queue.get()
        if work_item is None:
            break
        airport_code = work_item['airport_code']
        city_region = work_item['city_region']
        await geocode_airport_code(airport_code, city_region, csv_writer)

        await asyncio.sleep(RATE_LIMIT_DELAY)  
        queue.task_done()

# Main function to run the workflow
async def main():
    assert sys.version_info >= (3, 7), "This script requires Python 3.7+."

    # Open the output CSV file for writing
    with open(OUTFILE, 'w', encoding='utf8', newline='') as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerow(['airport_code', 'city', 'region', 'country', 'latitude', 'longitude'])

        queue = asyncio.Queue(maxsize=MAX_ITEMS)

        # Read airport codes from the input file
        with open(INFILE, 'r', encoding='utf8') as input_file:
            csv_reader = csv.reader(input_file, delimiter=';')  
            next(csv_reader)  
            for row in csv_reader:
                if row:  
                    airport_code = row[0]  
                    city_region = row[1]  
                    work_item = {'airport_code': airport_code, 'city_region': city_region}
                    await queue.put(work_item)

        sys.stderr.write(f"{queue.qsize()} work_items in the queue\n")

        # Start workers
        tasks = []
        for i in range(NUM_WORKERS):
            task = asyncio.create_task(run_worker(f'worker {i}', queue, csv_writer))
            tasks.append(task)

        sys.stderr.write("Waiting for workers to complete...\n")
        await queue.join()  

        # Cancel tasks
        for task in tasks:
            task.cancel()

        sys.stderr.write("Processing completed.\n")

# Event loop handling
if __name__ == "__main__":
    asyncio.run(main())
