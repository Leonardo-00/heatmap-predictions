''' Snap4city Computing HEATMAP - FastAPI Router.
    Copyright (C) 2024 DISIT Lab http://www.disit.org - University of Florence
'''

import os
import logging
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Any

from logic.heatmap.heatmap import generate_heatmap

router = APIRouter()
logger = logging.getLogger(__name__)

class HeatmapRequest(BaseModel):
    """
    Data model for the Heatmap generation request.
    """
    city: str
    long_min: float
    long_max: float
    lat_min: float
    lat_max: float
    epsg_projection: int
    value_types: Any  # Can be a list or a comma-separated string
    subnature: str
    scenario: str
    color_map: str
    from_date_time: str
    to_date_time: str
    token: str
    heat_map_model_name: str
    model: str # 'idw' or 'akima'
    clustered: int = 0
    file: int = 1
    broker: Optional[str] = None
    max_cells: int = 10000

@router.post("/")
def heatmap(req: HeatmapRequest):
    """
    Endpoint to trigger the heatmap generation process.
    Returns a structured dictionary with the status of each phase.
    """
    try:
        # generate_heatmap returns a dictionary (via HeatmapStatus.to_dict())
        # containing 'heatmapName', 'message', 'device', 'device_data', etc.
        result = generate_heatmap(req.model_dump())
        return result
    except Exception as e:
        logger.error(f"Unexpected error in heatmap service: {e}")
        # This fallback is only for unhandled exceptions outside the internal logic
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.get("/health")
def health_check():
    """Service health status check."""
    return {"status": "Heatmap service is running"}

@router.get("/logs")
def get_logs(
    lines: Optional[int] = Query(None, description="Number of lines to return from the end"),
    level: Optional[str] = Query(None, description="Filter by log level (INFO, DEBUG, ERROR)")
):
    """
    Retrieves execution logs for monitoring and debugging.
    """
    log_file = "uvicorn.log"
    if not os.path.exists(log_file):
        return JSONResponse(content={"error": "Log file not found"}, status_code=404)

    try:
        with open(log_file, "r") as f:
            all_lines = f.readlines()

        if level:
            level_tag = f"| {level.upper()} |"
            filtered_lines = [line for line in all_lines if level_tag in line]
        else:
            filtered_lines = all_lines

        if lines:
            filtered_lines = filtered_lines[-lines:]

        return {"logs": filtered_lines}
    except Exception as e:
        return JSONResponse(content={"error": f"Failed to read logs: {str(e)}"}, status_code=500)