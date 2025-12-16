''' Snap4city Computing HEATMAP.
   Copyright (C) 2024 DISIT Lab http://www.disit.org - University of Florence

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU Affero General Public License as
   published by the Free Software Foundation, either version 3 of the
   License, or (at your option) any later version.
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Affero General Public License for more details.
   You should have received a copy of the GNU Affero General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import pyproj
import requests
import math
from datetime import datetime
import json 
import os
import re
from pyproj import Transformer

from shapely import Point

def safe_float_conversion(x):
    if isinstance(x, dict) and 'value' in x and x['value'] != '':
        return float(x['value'])
    return np.nan  # Return NaN if the condition is not met

def convert_bbox_to_utm(lat_min, lat_max, long_min, long_max, epsg_projection):
    """
    Converte bounding box da WGS84 (lat/long) a UTM (EPSG dato).
    Restituisce city_bbox come lista di shapely.Point e DataFrame con X/Y.
    """
    bbox_coordinates = pd.DataFrame({
        'X': [long_min, long_max],
        'Y': [lat_min, lat_max]
    })

    # Definisci CRS
    wgs84 = pyproj.CRS("EPSG:4326")  
    utm = pyproj.CRS(f"EPSG:{epsg_projection}")  

    # Trasforma coordinate
    project = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True).transform
    utm_bbox_coordinates = bbox_coordinates.apply(
        lambda row: project(row['X'], row['Y']), axis=1
    )
    utm_bbox_coordinates = pd.DataFrame(
        utm_bbox_coordinates.tolist(), columns=['X', 'Y']
    )

    # Prepara punti shapely
    x = [utm_bbox_coordinates['X'][1], utm_bbox_coordinates['X'][0]]
    y = [utm_bbox_coordinates['Y'][1], utm_bbox_coordinates['Y'][0]]
    city_bbox = [Point(coord) for coord in zip(x, y)]

    return city_bbox, utm_bbox_coordinates


def compute_to_date(start_date: str, prevision_type: str, prevision_value: int) -> str:
    start_date_obj = datetime.strptime(start_date, '%Y-%m-%dT%H:%M:%S')

    if prevision_type == 'days':
        new_date_obj = start_date_obj + timedelta(days=prevision_value)
    elif prevision_type == 'months':
        new_date_obj = start_date_obj + relativedelta(months=prevision_value)
    elif prevision_type == 'hours':
        new_date_obj = start_date_obj + timedelta(hours=prevision_value)
    else:
        raise ValueError("Invalid prevision type")

    new_date = new_date_obj.strftime('%Y-%m-%dT%H:%M:%S')
    
    return new_date

def write_log(data: dict):
    file_name = 'log.txt'
    if not os.path.exists(file_name):
        with open('log.txt', 'w') as file:
            pass  
    
    with open(file_name, 'a') as file:
        file.write(json.dumps(data, indent=4))      
        file.write('\n') 

def get_sensor_real_time_data(params: object):
    base_url = os.getenv("BASE_URL", "https://www.snap4city.org") + "/superservicemap/api/v1"
    
    try:
        # Make the API request
        response = requests.get(base_url, params=params)
        
        log_data = {
            'url': base_url, 
            "params": json.dumps(params),
            "response_status": response.status_code
        }

        write_log(log_data)
        
        # Intercept raise_for_status to handle errors gracefully
        response.raise_for_status()
    
    except requests.exceptions.HTTPError as http_err:
        error_log = {
            'url': base_url, 
            "params": json.dumps(params),
            "error": str(http_err),
            "response_status": response.status_code if response else 'No Response'
        }
        write_log(error_log)

        return {
            "error": f"HTTP error occurred: {str(http_err)}",
            "status_code": response.status_code,
            "response": response.text
        }
    
    except Exception as err:
        error_log = {
            'url': base_url,
            "params": json.dumps(params),
            "error": str(err),
            "response_status": 'No Response'
        }
        write_log(error_log)
        
        return {
            "error": f"An error occurred: {str(err)}",
            "response": 'error'
        }

    # Return the successful API response
    return response.json()

def ISO_to_datetime(date: str):
    return  datetime.fromisoformat(date.replace('Z', ''))

def date_time_to_ISO(dateTime: datetime):
    return  dateTime.strftime('%Y-%m-%dT%H:%M:%S%z')

def parse_from_date(from_date: str, to_date: str) -> datetime:
    match_hour = re.match(r"(\d+)-hours", from_date)
    match_day = re.match(r"(\d+)-day", from_date)

    to_date_dt = datetime.strptime(to_date, "%Y-%m-%dT%H:%M:%S")
   
    if match_hour:
        hours = int(match_hour.group(1)) 
        return to_date_dt - timedelta(hours=hours)
    
    if match_day:
        days = int(match_day.group(1))
        return to_date_dt - timedelta(days=days)
    
    return None

def round_time(time_str):
    time_obj = datetime.strptime(time_str, '%H:%M')
    minute = (time_obj.minute // 10 + 1) * 10
    if minute == 60:
        time_obj += timedelta(hours=1)
        minute = 0
    return time_obj.replace(minute=minute).strftime('%H:%M')

def is_date(mydate: str, date_format: str="%Y-%m-%dT%H:%M:%S") -> bool:
    try:
        datetime.strptime(mydate, date_format)
        return True
    except ValueError:
        return False

def checkParameters(lat_min, long_min, lat_max, long_max, from_date_time, to_date_time):
    
    date_format = '%Y-%m-%dT%H:%M:%S'
    
    if not (-180 <= long_min <= 180 and -180 <= long_max <= 180):
        raise ValueError("Longitude values (long_min, long_max) must be between -180 and 180.")

    if not (-90 <= lat_min <= 90 and -90 <= lat_max <= 90):
        raise ValueError("Latitude values (lat_min, lat_max) must be between -90 and 90.")

    if long_max <= long_min:
        raise ValueError("Please insert 'long_max' and 'long_min' parameters correctly. 'long_max' must be greater than 'long_min'.")

    if lat_max <= lat_min:
        raise ValueError("Please insert 'lat_max' and 'lat_min' parameters correctly. 'lat_max' must be greater than 'lat_min'.")

    parsed_from_date = parse_from_date(from_date_time, to_date_time)

    if parsed_from_date == None and is_date(from_date_time) == False:
        raise ValueError("Please insert 'from_date_time' parameter correctly. The parameter has to be in a date and time format (yyyy-mm-ddThh:mm:ss) or in the format 'X-hours' or 'Y-day'.")

    if parsed_from_date == None:
        parsed_from_date = datetime.strptime(from_date_time, date_format)
        
    if is_date(to_date_time) == False:
        raise ValueError("Please insert 'to_date_time' parameter correctly. The parameter has to be in a date and time format (yyyy-mm-ddThh:mm:ss).")
    
    parsed_to_date = datetime.strptime(to_date_time, date_format)
    
    if parsed_to_date <= parsed_from_date:
        raise ValueError("Please insert 'to_date_time' and 'from_date_time' parameters correctly. 'to_date_time' must be greater than 'from_date_time'.")