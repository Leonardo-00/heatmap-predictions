''' Snap4city Computing HEATMAP - Main Application Entry Point.
    Copyright (C) 2024 DISIT Lab http://www.disit.org - University of Florence
'''

import uvicorn
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from api.heatmap_service import router

# --- Logging Configuration ---
# Configurato per essere compatibile con il parsing dell'endpoint /logs
logging.basicConfig(
    filename="uvicorn.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# Assicuriamoci che i log di uvicorn e del servizio siano unificati
logger = logging.getLogger("uvicorn")

app = FastAPI(
    title="Snap4City Heatmap API",
    description="API Service for spatial interpolation and heatmap generation",
    version="1.0.0"
)

# --- Router Inclusion ---
app.include_router(router, prefix="/heatmap", tags=["heatmap"])

# --- Custom Exception Handlers ---
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handles Pydantic validation errors (e.g., missing fields in JSON payload).
    Returns a structured error response for the caller.
    """
    missing_fields = []
    for error in exc.errors():
        if error["type"] == "missing":
            # Extract the field name from the location tuple
            missing_fields.append(str(error["loc"][-1]))
    
    error_msg = "Request validation failed: missing required fields."
    logger.warning(f"{error_msg} {missing_fields}")
    
    return JSONResponse(
        status_code=422,
        content={
            "error": error_msg,
            "missing_fields": missing_fields
        }
    )

# --- Server Startup ---
if __name__ == "__main__":
    # Nota: Assicurati che il percorso "service.main:app" sia corretto 
    # rispetto alla struttura delle cartelle del tuo progetto.
    uvicorn.run(
        "service.main:app", 
        host="0.0.0.0", 
        port=8085, 
        reload=True,
        log_level="info"
    )