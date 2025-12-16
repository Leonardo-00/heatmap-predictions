import os
import time
import json
import requests
from urllib.parse import quote
from ..helper import write_log

import logging

logger = logging.getLogger(__name__)

# --- Funzioni di upload dei dati ---

# Funzione per creare il dizionario dei dati interpolati
def create_interpolated_heatmap(interpolated_data, heatmap_name, metric_name, from_date_time, to_date_time, clustered, step_size, epsg_projection, file):
    interpolated_heatmap = {
        'attributes': [],
        'saveStatus': []
    }

    id = 1
    for _, row in interpolated_data.iterrows():
        x = row['X']
        y = row['Y']
        value = row['Z']

        # Create a dictionary for the current record
        list_attrib_temp = {
            'id': id,
            'mapName': heatmap_name,
            'metricName': metric_name,
            'description': f"Average from {from_date_time} to {to_date_time}",
            'clustered': clustered,
            'latitude': x,
            'longitude': y,
            'value': value,
            'date': f"{to_date_time}Z",
            'Side_Length': step_size,
            'projection': int(epsg_projection),
            'file': file,
            'org': 'DISIT'
        }
        id += 1

        # Append the dictionary to the attributes list
        interpolated_heatmap['attributes'].append(list_attrib_temp)

    # Print the length of the attributes list
    logger.debug(f"interpolatedHeatmap list length: {len(interpolated_heatmap['attributes'])}")
    return interpolated_heatmap

# Funzione per caricare la heatmap su Snap4City
def upload_heatmap_to_snap4city(token, heat_map_model_name, broker, subnature, device_name, heatmap_name, color_map, 
                                coordinates, interpolated_data, from_date_time, to_date_time):
    base_url = os.getenv("BASE_URL", "https://www.snap4city.org")
    orionfilter_base_url = os.getenv("ORIONFILTER_BASE_URL", "https://www.snap4city.org/orionfilter")
    if broker is None:
        broker = os.getenv("DEFAULT_BROKER", "orionUNIFI")

    # bounding box in WKT
    long_min, long_max = coordinates['long'].min(), coordinates['long'].max()
    lat_min, lat_max = coordinates['lat'].min(), coordinates['lat'].max()
    wkt = (
        f"POLYGON(({long_min} {lat_min}, {long_max} {lat_min}, "
        f"{long_max} {lat_max}, {long_min} {lat_max}, {long_min} {lat_min}))"
    )
    wkt_geo_json = {
        "type": "Polygon",
        "coordinates": [[
            [long_min, lat_min],
            [long_max, lat_min],
            [long_max, lat_max],
            [long_min, lat_max],
            [long_min, lat_min]
        ]]
    }

    config = {
        "token": token,
        "model": {
            "model_name": heat_map_model_name,
            "model_type": "Heatmap",
            "model_kind": "sensor",
            "model_frequency": "600",
            "model_kgenerator": "normal",
            "model_contextbroker": broker,
            "model_protocol": "ngsi",
            "model_format": "json",
            "model_hc": "refresh_rate",
            "model_hv": "300",
            "model_subnature": subnature,
            "model_url": f"{base_url}/iot-directory/api/model.php?",
        },
        "producer": "DISIT",
        "k1": "cdfc46e7-75fd-46c5-b11e-04e231b08f37",
        "k2": "24f146b3-f2e8-43b8-b29f-0f9f1dd6cac5",
        "hlt": "Heatmap",
        "device_name": device_name,
        "heatmap_name": heatmap_name,
        "url": base_url,
        "device": {
            "device_url": f"{base_url}/iot-directory/api/device.php?"
        },
        "long": coordinates['long'].values[0],
        "lat": coordinates['lat'].values[0],
        "patch": f"{orionfilter_base_url}/{broker}/v2/entities/",
        "usernamedelegated": "",
        "passworddelegated": "",
        "color_map": color_map,
        "wkt": wkt,
        "bounding_box": wkt_geo_json,
        "size": len(interpolated_data),
        "date_observed": f"{to_date_time}Z",
        "description": f"Average from {from_date_time} to {to_date_time}",
        "maximum_date": to_date_time,
        "minimum_date": from_date_time
    }

    logger.debug(json.dumps(config))

    r = create_heatmap_device(config)
    info_heatmap = {}

    if r['status'] == 'ko' and r.get('error_msg'):
        info_heatmap['device'] = {
            'POSTStatus': "Error in creating device",
            'error': r['error_msg']
        }
    else:
        info_heatmap['device'] = {
            'POSTStatus': f"Stato creazione device {config['device_name']} : {r['status']}"
        }

    logger.info("Insert data in Device")
    time.sleep(5)

    response = send_heatmap_device_data(config)
    if response.status_code == 204:
        info_heatmap['device_data'] = {"POSTStatus": "Inserimento riuscito"}
    else:
        info_heatmap['device_data'] = {
            "POSTStatus": "Inserimento fallito",
            "error": response.text
        }

    return info_heatmap

# Funzione per creare il device della heatmap
def create_heatmap_device(conf: dict):
    logger.info("Create Device")
    token = conf['token']
    device_name = conf['device_name']
    device_url = conf["device"]["device_url"]
    type = conf['model']['model_type']
    kind = conf['model']['model_kind']
    context_broker = conf['model']['model_contextbroker']
    format = conf['model']['model_format']
    model = conf['model']['model_name']
    producer = conf['producer']
    lat = conf['lat']
    long = conf['long']
    frequency = conf['model']['model_frequency']
    k1 = conf['k1']
    k2 = conf['k2']
    subnature = conf['model']['model_subnature']
    hlt = conf['hlt']
    wkt = conf['wkt']

    attributes = quote(json.dumps([{"value_name": "dateObserved", "data_type": "string", "value_type": "timestamp", "editable": "0",
                                    "value_unit": "timestamp", "healthiness_criteria": "refresh_rate", "healthiness_value": "300",
                                    "real_time_flag": "false"},
                                   {"value_name": "mapName", "data_type": "string", "value_type": "Identifier", "editable": "0", "value_unit": "ID",
                                    "healthiness_criteria": "refresh_rate", "healthiness_value": "300", "real_time_flag": "false"},
                                   {"value_name": "colorMap", "data_type": "string", "value_type": "Identifier", "editable": "0", "value_unit": "ID",
                                    "healthiness_criteria": "refresh_rate", "healthiness_value": "300", "real_time_flag": "false"},
                                   {"value_name": "minimumDate", "data_type": "string", "value_type": "time", "editable": "0", "value_unit": "s",
                                    "healthiness_criteria": "refresh_rate", "healthiness_value": "300", "real_time_flag": "false"},
                                   {"value_name": "maximumDate", "data_type": "string", "value_type": "time", "editable": "0", "value_unit": "s",
                                    "healthiness_criteria": "refresh_rate", "healthiness_value": "300", "real_time_flag": "false"},
                                   {"value_name": "instances", "data_type": "integer", "value_type": "Count", "editable": "0", "value_unit": "#",
                                    "healthiness_criteria": "refresh_rate", "healthiness_value": "300", "real_time_flag": "false"},
                                   {"value_name": "description", "data_type": "string", "value_type": "status", "editable": "0",
                                    "value_unit": "status", "healthiness_criteria": "refresh_rate", "healthiness_value": "300",
                                    "real_time_flag": "false"},
                                   {"value_name": "boundingBox", "data_type": "json", "value_type": "Geometry", "editable": "0", "value_unit": "text",
                                    "healthiness_criteria": "refresh_rate", "healthiness_value": "300", "real_time_flag": "false"},
                                   {"value_name": "size", "data_type": "integer", "value_type": "status", "editable": "0", "value_unit": "status",
                                    "healthiness_criteria": "refresh_rate", "healthiness_value": "300", "real_time_flag": "false"}]))

    header = {
        "Content-Type": "application/json",
        "Accept": "application/x-www-form-urlencoded",
        "Authorization": f"Bearer {token}",
    }

    url = device_url + f"action=insert&attributes={attributes}&id={device_name}&type={type}&kind={kind}&contextbroker={context_broker}&format={format}&mac=&model={model}&producer={producer}&latitude={lat}&longitude={long}&visibility=&frequency={frequency}&accessToken={token}&k1={k1}&k2={k2}&edgegateway_type=&edgegateway_uri=&subnature={subnature}&static_attributes=&servicePath=&nodered=false&hlt={hlt}&wktGeometry={wkt}&nodered=false"

    response = requests.request("PATCH", url, headers=header)

    write_log({
        "url": url,
        "header": header,
        "response_status": response.status_code
    })

    r = response.text
    r = json.loads(r)
    time.sleep(2)

    return r

# Funzione per inviare i dati della heatmap al device
def send_heatmap_device_data(conf: dict):
    token = conf['token']
    heatmap_name = conf['heatmap_name']
    device_name = conf['device_name']
    payload = {
        "boundingBox": {"value": conf['bounding_box'], "type": "json"},
        "colorMap": {"value": conf['color_map'], "type": "string"},
        "dateObserved": {"value": conf['date_observed'], "type": "string"},
        "description": {"value": conf['description'], "type": "string"},
        "instances": {"value": 1, "type": "integer"},
        "mapName": {"value": heatmap_name, "type": "string"},
        "maximumDate": {"value": conf['maximum_date'], "type": "string"},
        "minimumDate": {"value": conf['minimum_date'], "type": "string"},
        "size": {"value": conf['size'], "type": "integer"}
    }

    header = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
    }

    # timestamp = datetime.now().isoformat()
    # timestamp = timestamp[0:20] + "000Z"

    url = conf["patch"] + device_name + '/attrs?elementid=' + device_name + '&type=' + conf['model']['model_type']
    response = requests.request("PATCH", url, data=json.dumps(payload), headers=header)

    write_log({
        "url": url,
        "header": header,
        "params": payload,
        "response_status": response.status_code
    })
    return response

# Funzione per salvare i dati interpolati su Snap4City
def save_interpolated_data(interpolated_heatmap, heatmap_name, metric_name, to_date_time, info_heatmap=None):
    """
    Salva i dati interpolati su Snap4City tramite /insertArray e setMap.php.

    Args:
        interpolated_heatmap (dict): dati interpolati, deve contenere 'attributes'
        heatmap_name (str): nome della heatmap
        metric_name (str): metrica associata
        to_date_time (str): timestamp finale (ISO)
        info_heatmap (dict, opzionale): dizionario su cui aggiornare lo stato

    Returns:
        dict: info_heatmap aggiornato con lo stato del salvataggio
    """

    if info_heatmap is None:
        info_heatmap = {}

    # Convert the data to JSON format
    request_body_json = json.dumps(interpolated_heatmap['attributes'], indent=4)

    # Define the URL for the POST request
    heatmap_insert_base_url = os.getenv("HEATMAP_INSERT_BASE_URL", "http://192.168.0.59:8000")
    heatmap_setmap_url = os.getenv("HEATMAP_SETMAP_URL", "http://192.168.0.59/setMap.php")
    url = heatmap_insert_base_url + "/insertArray"
    logger.info("Sending POST to " + url)

    # Send the POST request
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(url, data=request_body_json, headers=headers)
        logger.debug(response)
        save_status = response.status_code
    except requests.RequestException as e:
        logger.error(f"Request failed: {e}")
        save_status = None

    # Update the save status
    interpolated_heatmap['saveStatus'] = save_status

    if save_status == 200:
        logger.debug("StatusCode 200")
        completed = 1
    else:
        logger.debug("Completed Status = 0")
        completed = 0

    # Construct the final API URL for GET request
    api_final = f"{heatmap_setmap_url}?mapName={heatmap_name}&metricName={metric_name}&date={to_date_time}Z&completed={completed}"
    logger.info(api_final)

    # Send the GET request
    try:
        response = requests.get(api_final)
        api_status_code = response.status_code
    except requests.RequestException as e:
        logger.error(f"Request failed: {e}")
        api_status_code = 0

    # Update info_heatmap with result
    if api_status_code == 200:
        info_heatmap['interpolation'] = {
            "POSTstatus": "Interpolated data saved correctly"
        }
        logger.info("Interpolated data saved correctly")
    else:
        info_heatmap['interpolation'] = {
            "POSTstatus": "Problems on saving interpolated data",
            "error": response.text if 'response' in locals() else "No response"
        }
        logger.error("Problems on saving interpolated data")

    return info_heatmap