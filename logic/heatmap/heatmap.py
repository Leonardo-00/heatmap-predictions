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

from datetime import datetime
import logging

import geopandas as gpd
import pandas as pd
from pyproj import Proj, Transformer
from shapely.geometry import Point

# --- Import delle funzioni dai moduli interni ---


from .data_upload import upload_heatmap_to_snap4city, save_interpolated_data, create_interpolated_heatmap
from .data_interpolation import build_square_grid, interpolate
from .data_preprocessing import preprocess_sensors_data
from .data_retrieval import get_sensors_in_area, load_sensors_data
from ..helper import checkParameters, convert_bbox_to_utm

try:
    pd.set_option("future.no_silent_downcasting", True)
except (KeyError, AttributeError):
    pass

logger = logging.getLogger(__name__)

def generate_heatmap(params: dict):
    
    city = params.get("city")
    long_min = params.get("long_min")
    long_max = params.get("long_max")
    lat_min = params.get("lat_min")
    lat_max = params.get("lat_max")
    epsg_projection = params.get("epsg_projection")
    value_types = params.get("value_types")
    subnature = params.get("subnature")
    scenario = params.get("scenario")
    color_map = params.get("color_map")
    from_date_time = params.get("from_date_time")
    to_date_time = params.get("to_date_time")
    token = params.get("token")
    heat_map_model_name = params.get("heat_map_model_name")
    model = params.get("model")
    clustered = params.get("clustered", 0)
    file = params.get("file", 1)
    broker = params.get("broker", None)
    max_cells = params.get("max_cells", 10000)
    
    logger.debug("--------- CHECK ON PARAMETERS START ---------")
    logger.debug(datetime.now())
    logger.debug("------------------------------------------------")
    
    checkParameters(lat_min, long_min, lat_max, long_max, from_date_time, to_date_time)

    logger.debug("--------- CHECK ON PARAMETERS END ---------")
    logger.debug(datetime.now())
    logger.debug("------------------------------------------------")
    

    # Via nodered the valueType parameter is not entered as an array but as a string.
    # Splitting of the elements and creation of the array:
    
    if isinstance(value_types, str):
        value_types = [vt.strip() for vt in value_types.split(",")]

    heatmap_name = f"{scenario}_" + "_".join(value_types)
    device_name = heatmap_name
    
    logger.info(f"Heatmap name: {heatmap_name}")
    logger.info(f"Value types: {value_types}")

    metric_name = color_map
    sensor_category = subnature
    
    logger.debug("--------- UPLOAD ALL SENSOR STATIONS IN THE AREA OF INTEREST - START---------")
    logger.debug(datetime.now())

    try:
        service_uris = get_sensors_in_area(lat_min, long_min, lat_max, long_max, sensor_category, token)
        sensors_data = load_sensors_data(service_uris, from_date_time, to_date_time, token)
    except Exception as e:
        raise e
    logger.debug("--------- SensorData List Creation Completed ---------")
    logger.debug("--------- UPLOAD DATA FOR EACH SENSOR STATION - END ---------")
    logger.debug(datetime.now())
    logger.debug("------------------------------------------------")

    info_heatmap = {
        "heatmapName": heatmap_name,
        "dateTime": to_date_time,
        "message": []
    }
    
    # -------------------------------------------------------
    # 3. Preprocessing
    # -------------------------------------------------------
    logger.debug("--------- PREPROCESSING -- START ---------")
    logger.debug(datetime.now())
    try:
        data = preprocess_sensors_data(sensors_data, value_types, info_heatmap)
    except ValueError as e:
        raise e

    logger.debug("--------- DATA MANIPULATION -- END ---------")
    logger.debug(datetime.now())
    logger.debug("------------------------------------------------")

    # --------------------------------------------------------------------------------------------------------------------------------------#
    logger.debug("--------- LAT-LONG BBOX CONVERSION TO UTM - START---------")
    city_bbox, utm_bbox_coordinates = convert_bbox_to_utm(
        float(lat_min), float(lat_max),
        float(long_min), float(long_max),
        epsg_projection
    )
    logger.debug("--------- LAT-LONG BBOX CONVERSION TO UTM - END ---------")
    logger.debug("------------------------------------------------")

    logger.debug("--------- DATA INTERPOLATION - START ---------")


    if len(data) >= 3:
        data = data.dropna().drop_duplicates().reset_index(drop=True)
        values = data['value'].astype(float).to_frame(name='value')
        coordinates = data[['long', 'lat']]

        # Transform to UTM coordinates
        wgs84 = Proj("EPSG:4326")  # WGS84 Lat/Lon
        utm = Proj(f"EPSG:{epsg_projection}")  # UTM with the given EPSG code
        transformer = Transformer.from_proj(wgs84, utm)

        coordinates['X'], coordinates['Y'] = transformer.transform(coordinates['lat'].values, coordinates['long'].values)

        # Create a GeoDataFrame
        geometry = [Point(xy) for xy in zip(coordinates['X'], coordinates['Y'])]
        gdf = gpd.GeoDataFrame(values, geometry=geometry, crs=f"EPSG:{epsg_projection}")

        # Ensure that the bounding box is aligned with the grid
        city_bbox = gpd.GeoSeries(city_bbox, crs=f"EPSG:{epsg_projection}")
        xmin, ymin, xmax, ymax = city_bbox.total_bounds
        
        grid_x, grid_y, step_size, nx, ny = build_square_grid(xmin, xmax, ymin, ymax, max_cells=max_cells)

        logger.info("Grid parameters:")
        logger.info(f"Step size: {step_size:.1f}m")
        logger.info(f"#cells: {nx} x {ny} = {nx * ny}")

        logger.info(f"Grid shape (centri): {grid_x.shape} -> punti totali {grid_x.size}")

        # --- interpolation ---
        xy_known = coordinates[['X','Y']].values
        val_known = values['value'].values
        
        args = {}

        args.update({'cell_size': step_size, 'hull_expand_cells': 1, 'power': 2})
        
        grid_z = interpolate(xy_known, val_known, grid_x, grid_y, model, **args)

        interpolated_data = pd.DataFrame({
            'X': grid_x.flatten(),
            'Y': grid_y.flatten(),
            'Z': grid_z.flatten()
        }).dropna()

        logger.info(f"interpolatedData dim: {len(interpolated_data)}")
    else:
        raise ValueError("Not enough data points for interpolation. At least 3 valid data points are required.")
    logger.debug("--------- DATA INTERPOLATION - END ---------")

    logger.debug("--------- INTERPOLATED DATA LIST CREATION - START ---------")
    logger.debug(datetime.now())
    
    interpolated_heatmap = create_interpolated_heatmap(interpolated_data, heatmap_name, metric_name, from_date_time, to_date_time, 
                                                    clustered, step_size, epsg_projection, file)
    
    logger.debug("--------- INTERPOLATED DATA LIST CREATION - END ---------")
    logger.debug(datetime.now())
    logger.debug("------------------------------------------------")
    
    logger.debug("--------- UPLOAD HEATMAP DEVICE AND DATA - START ---------")
    logger.debug(datetime.now())
    info_heatmap.update(upload_heatmap_to_snap4city(token, heat_map_model_name, broker, subnature, device_name, heatmap_name, color_map, 
                                                    coordinates, interpolated_data, from_date_time, to_date_time))

    logger.debug("--------- UPLOAD HEATMAP DEVICE AND DATA - END ---------")
    logger.debug(interpolated_heatmap['attributes'][:5])
    
    logger.debug("--------- SAVING INTERPOLATED DATA LIST - START ---------")
    logger.debug(datetime.now())
    logger.debug("------------------------------------------------")
    
    info_heatmap.update(save_interpolated_data(interpolated_heatmap, heatmap_name, metric_name, to_date_time))
    logger.debug("--------- SAVING INTERPOLATED DATA LIST - END ---------")
    
    return info_heatmap


