import pandas as pd
import numpy as np
from ..helper import safe_float_conversion

import logging

logger = logging.getLogger(__name__)

# --- Funzioni di preprocessing dei dati ---

# Funzione principale di preprocessing
def preprocess_sensors_data(sensors_data, value_types, info_heatmap):
    # 1. Rimuovi sensori vuoti
    sensors_data = drop_empty_sensors(sensors_data)
    
    if not sensors_data:
        raise ValueError(f"No Available Data for {value_types}")

    # 2. Controlla colonne richieste
    names_matrix = validate_value_types(sensors_data, value_types)

    # 3. Calcola medie
    data = compute_average_values(sensors_data, names_matrix)

    # 4. Pulizia finale
    return clean_data_values(data, info_heatmap, value_types)

# Rimuove i sensori senza dati validi
def drop_empty_sensors(sensors_data):
    """
    Rimuove i sensori senza dati validi (tutte le colonne NaN).
    """
    logger.debug("--------- 1. Drop empty ServiceUris ---------")
    return [
        entry for entry in sensors_data
        if not entry['sensorRealtimeData'].isnull().all().all()
    ]

# Controlla che i sensori abbiano le colonne richieste
def validate_value_types(sensors_data, value_types):
    """
    Controlla che i sensori abbiano le colonne richieste.
    Restituisce un names_matrix (sensorDataIndex, varNameList, valTypeIndex).
    """
    logger.debug("--------- 2. Check on ValueType ---------")
    sensors_data_index, val_type_index, var_name_list = [], [], []
    for i, entry in enumerate(sensors_data):
        dat_temp = entry['sensorRealtimeData']
        for value in value_types:
            if value in dat_temp.columns:
                sensors_data_index.append(i)
                val_type_index.append(dat_temp.columns.get_loc(value))
                var_name_list.append(value)

    if not var_name_list:
        raise ValueError(f"The valueType is incorrect. Expected {value_types}")

    return pd.DataFrame({
        'sensorDataIndex': sensors_data_index,
        'varNameList': var_name_list,
        'valTypeIndex': val_type_index
    })

# Calcola la media dei valori per ogni sensore
def compute_average_values(sensors_data, names_matrix):
    """
    Calcola media dei valori per ogni sensore.
    Restituisce un DataFrame con lat, long, value.
    """
    logger.debug("--------- 3. Average Values Matrix creation ---------")
    data = pd.DataFrame(columns=["lat", "long", "value"], index=range(len(sensors_data)))

    for j, entry in enumerate(sensors_data):
        temp = entry['sensorRealtimeData'].drop(
            columns=["measuredTime", "time", "date", "dateObserved",
                     "reliability", "source", "day", "sensorName"],
            errors='ignore'
        )

        if not names_matrix[names_matrix['sensorDataIndex'] == j].empty:
            var_name = names_matrix.loc[
                names_matrix['sensorDataIndex'] == j, 'varNameList'
            ].values[0]

            if "dateTime" in temp.columns:
                temp_new = temp.loc[:, [var_name, "dateTime"]]
            else:
                temp_new = temp[[var_name]]

            temp_new[var_name] = temp_new[var_name].apply(safe_float_conversion)
            entry['indexValues'] = temp_new

            if temp_new[var_name].isna().all():
                data.at[j, "value"] = np.nan
            else:
                data.at[j, "value"] = temp_new[var_name].mean(skipna=True)

        # sempre assegna coordinate
        data.at[j, "lat"] = entry['sensorCoordinates'][1]
        data.at[j, "long"] = entry['sensorCoordinates'][0]

    return data

# Pulizia finale dei dati
def clean_data_values(data, info_heatmap, value_types):
    """
    Sostituisce valori sentinella (-9999, 9999) e gestisce il caso di tutto NaN.
    """
    logger.debug("--------- 4. Null values editing ---------")
    if data['value'].isna().all():
        info_heatmap["message"].append(
            f"No Available Data for {value_types}: All ServiceUris are empty"
        )
    else:
        data['value'] = data['value'].replace([-9999, 9999], np.nan)
        data = data.infer_objects(copy=False)
    return data

