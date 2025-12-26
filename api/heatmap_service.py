import os
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from logic.heatmap.heatmap import generate_heatmap

router = APIRouter()

class HeatmapRequest(BaseModel):
    city: str
    long_min: float
    long_max: float
    lat_min: float
    lat_max: float
    epsg_projection: int
    value_types: object
    subnature: str
    scenario: str
    color_map: str
    from_date_time: str
    to_date_time: str
    token: str
    heat_map_model_name: str
    model: str
    clustered: int = 0
    file: int = 1
    broker: str = None
    max_cells: int = 200000
    
    

@router.post("/")
def heatmap(req: HeatmapRequest):
    try:
        return generate_heatmap(req.model_dump())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@router.get("/health")
def health_check():
    return {"status": "Heatmap service is running"}

@router.get("/logs")
def get_logs(
    lines: int = Query(None, description="Numero di righe da restituire, None per tutte"),
    level: str = Query(None, description="Filtra per livello di log, es. INFO, DEBUG, ERROR")
):
    log_file = "uvicorn.log"
    if not os.path.exists(log_file):
        return JSONResponse(content={"error": "Log file not found"}, status_code=404)

    with open(log_file, "r") as f:
        all_lines = f.readlines()

    if level:
        filtered_lines = [line for line in all_lines if f"| {level.upper()} |" in line]
    else:
        filtered_lines = all_lines

    if lines:
        filtered_lines = filtered_lines[-lines:]

    return {"logs": filtered_lines}