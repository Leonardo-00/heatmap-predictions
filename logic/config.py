# config.py

# --- HEATMAP ---
# Default spatial projection (EPSG code)
DEFAULT_EPSG = 32632

# Default flags for the generation process
DEFAULT_CLUSTERED = 0  # 0=No, 1=Yes
DEFAULT_FILE_FLAG = 0  # 0=DB only, 1=File generation
DEFAULT_BROKER = None  # Let backend decide or use specific broker

# Minimum number of valid sensors required to attempt interpolation
MIN_SENSORS_REQUIRED = 3

# Coordinate Reference Systems
CRS_LATLON = "EPSG:4326"

# --- API & DATA RETRIEVAL ---
# URL base per le chiamate API Snap4City
SNAP4CITY_BASE_URL = "https://www.snap4city.org"

# Timeout in secondi per le richieste di rete (evita che il processo si appenda)
API_TIMEOUT = 30

# Limiti per la discovery dei sensori nell'area
DISCOVERY_MAX_RESULTS = 100
DISCOVERY_MAX_DIST = 5  # Parametro di distanza usato dall'API

# Formato data richiesto dalle API
DATE_FORMAT_API = '%Y-%m-%dT%H:%M:%S'

# --- PREPROCESSING SETTINGS ---
# Categorie di sensori che non possono avere valori negativi (es. PM10, traffico)
PHYSICAL_POSITIVE_CATS = {
    "Air_quality_monitoring_station",
    "Traffic_sensor",
    "Weather_sensor"
}

# Categorie da aggregare usando la media (Mean)
AGG_MEAN_CATS = {
    "Air_quality_monitoring_station", 
    "Weather_sensor"
}

# Categorie da aggregare usando il percentile (es. rumore)
AGG_PERCENTILE_CATS = {
    "Noise_monitoring_station"
}

# Valore del percentile da usare (es. 85 per Lden/Noise)
AGG_PERCENTILE_VALUE = 85

# Valori sentinella che indicano errore del sensore da convertire in NaN
SENTINEL_VALUES = [-9999, 9999, -999, 999]

# --- INTERPOLATION GENERAL ---
# Numero massimo di celle nella griglia (limita la complessità computazionale)
MAX_CELLS = 10000
# Dimensione iniziale della cella in metri (risoluzione di base)
BASE_CELL_SIZE = 10.0

# --- IDW SETTINGS ---
# Esponente per la ponderazione della distanza (più alto = influenza locale più forte)
IDW_POWER = 4
# Distanza in metri oltre la quale il valore sfuma a zero
IDW_FADE_DISTANCE = 800
# Parametro di smoothing per il fade-out (più alto = sfumatura più dolce)
IDW_FADE_SMOOTHING = 5.0

# --- AKIMA SETTINGS ---
# Fattore per il buffer del Convex Hull (moltiplicatore della cell_size)
AKIMA_HULL_BUFFER_FACTOR = 3.0
# Numero di punti di controllo per le spline (righe/colonne)
AKIMA_SPLINE_POINTS = 5
# Numero di vicini (k) per la ricerca cKDTree
AKIMA_K_NEIGHBORS = 6
# Esponente per la ponderazione inversa della distanza nel pre-calcolo Akima
AKIMA_WEIGHT_POWER = 3.5

# --- POST PROCESSING ---
# Deviazione standard per il filtro Gaussiano finale
GAUSSIAN_SIGMA = 1.5

# --- IOT DEVICE & UPLOAD SETTINGS ---
# URL per il Context Broker e filtri
ORION_BASE_URL = "https://www.snap4city.org/orionfilter"
DEFAULT_BROKER = "orionUNIFI"

# Chiavi API hardcodate (Legacy)
IOT_K1 = "cdfc46e7-75fd-46c5-b11e-04e231b08f37"
IOT_K2 = "24f146b3-f2e8-43b8-b29f-0f9f1dd6cac5"

# Parametri di default del Device
DEVICE_PRODUCER = "DISIT"
DEVICE_MODEL_TYPE = "Heatmap"
DEVICE_MODEL_KIND = "sensor"
DEVICE_MODEL_FREQ = "600"
DEVICE_MODEL_FORMAT = "json"

# Endpoint interni per il salvataggio della griglia (spesso diversi in dev/prod)
# Nota: Questi IP interni (192.168...) dovrebbero essere sovrascrivibili via env var
HEATMAP_INSERT_URL = "http://192.168.0.59:8000/insertArray"
HEATMAP_SETMAP_URL = "http://192.168.0.59/setMap.php"

# Headers standard
JSON_HEADER = {"Content-Type": "application/json"}