from fastapi import APIRouter, HTTPException
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
    broker: str | None = None
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