import os
import requests
import pandas as pd
from ..helper import write_log, parse_from_date, round_time

import logging

logger = logging.getLogger(__name__)

# --- Funzioni di retrieval dei dati ---

# Funzione per il fetch dei dati di un singolo sensore
def fetch_sensor_data(service_uri: str, from_date_time: str, to_date_time: str, access_token: str):
    parsed_from_date = parse_from_date(from_date_time, to_date_time)
    if not parsed_from_date:
        parsed_from_date = from_date_time
    base_url = os.getenv("BASE_URL","https://www.snap4city.org")
    api_url = (
        f"{base_url}/superservicemap/api/v1/?serviceUri={service_uri}"
        f"&fromTime={parsed_from_date}&toTime={to_date_time}&accessToken={access_token}"
    )
    header = {
      "Authorization": f"Bearer {access_token}"
    }
    try:
        res = requests.get(api_url,headers=header)
        sensors_data = res.json()

        write_log({
            "url": api_url,
            "response_status": res.status_code
        })

        if 'realtime' in sensors_data and 'results' in sensors_data['realtime']:
            results = sensors_data['realtime']['results']['bindings']

            if len(results) < 0:
                return None

            # Extract sensor information
            sensor_info = sensors_data['Service']['features'][0]
            coordinates = sensor_info['geometry']['coordinates']
            name = sensor_info['properties']['name'] or f"{service_uri}_sensor"

            # Process realtime data
            results = sensors_data['realtime']['results']['bindings']
            df = pd.DataFrame(results)

            if 'measuredTime' not in df.columns:
                return None

            variable_names = [col for col in df.columns if col != 'measuredTime']
            temp_df = df.copy()

            # Update variable names
            temp_df = temp_df.rename(columns={var: var for var in variable_names})

            # Convert measuredTime to datetime and adjust time slots
            temp_df['measuredTime'] = temp_df['measuredTime'].apply(lambda x: x['value'] if isinstance(x, dict) and 'value' in x else None)

            temp_df['measuredTime'] = pd.to_datetime(temp_df['measuredTime'], utc=True, errors='coerce')
            temp_df = temp_df.dropna(subset=['measuredTime'])
            temp_df['time'] = temp_df['measuredTime'].dt.strftime('%H:%M')
            temp_df['date'] = temp_df['measuredTime'].dt.strftime('%Y-%m-%d')
            temp_df['day'] = temp_df['measuredTime'].dt.day_name()

            # Round time slots every 10 minutes

            temp_df['time'] = temp_df['time'].apply(round_time)

            # Adjust date and time columns
            temp_df['dateTime'] = temp_df.apply(lambda row: f"{row['date']}T{row['time']}:00", axis=1)
            temp_df = temp_df.sort_values(by='measuredTime')
            temp_df['sensorName'] = name

            return {
                'sensorCoordinates': coordinates,
                'sensorName': name,
                'sensorRealtimeData': temp_df
            }

        return None

    except Exception as e:
        logger.error(f"Error fetching data for ServiceUri: {service_uri}, Error: {e}")
        write_log({ "exception": f"Error fetching data for ServiceUri: {service_uri}, Error: {e}" })

# Funzione per ottenere i sensori in un'area specifica
def get_sensors_in_area(lat_min, long_min, lat_max, long_max, sensor_category, token, base_url=None):
    """
    Restituisce la lista dei serviceUri dei sensori trovati nel bounding box specificato.
    """
    base_url = base_url or os.getenv("BASE_URL", "https://www.snap4city.org")
    query = (f"{base_url}/superservicemap/api/v1/?selection="
             f"{lat_min};{long_min};{lat_max};{long_max}"
             f"&categories={sensor_category}"
             f"&maxResults=100&maxDists=5&format=json")

    response = requests.get(query)

    write_log({
        "url": query,
        "token": token,
        "response_status": response.status_code
    })

    if response.status_code != 200:
        raise RuntimeError("Error fetching data from the service")

    sensor_category_json = response.json()

    service_uris = [
        service['properties']['serviceUri']
        for service in sensor_category_json.get('Services', {}).get('features', [])
    ]

    if not service_uris:
        raise ValueError("No Sensor Station in the selected area. Please, correct the coordinates")

    return service_uris

# Funzione per caricare i dati di tutti i sensori
def load_sensors_data(service_uris, from_date_time, to_date_time, token):
    """
    Data una lista di serviceUri, scarica i dati per ciascun sensore.
    Restituisce una lista di dataset (uno per sensore).
    """
    sensors_data = []

    for service_uri in service_uris:
        data = fetch_sensor_data(service_uri, from_date_time, to_date_time, token)
        if data:
            sensors_data.append(data)

    return sensors_data
