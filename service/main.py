from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from api.heatmap_service import router
import uvicorn
import logging

# Configura logging su file
logging.basicConfig(
    filename="uvicorn.log",        # file dei log
    level=logging.INFO,             # livello minimo da loggare
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("uvicorn")

app = FastAPI(title="Snap4City Heatmap API")


app.include_router(router, prefix="/heatmap", tags=["heatmap"])

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    missing = []
    for error in exc.errors():
        if error["type"] == "missing":
            missing.append(error["loc"][-1])
    return JSONResponse(
        status_code=422,
        content={
            "error": "Missing required fields",
            "missing_fields": missing
        }
    )

if __name__ == "__main__":

    uvicorn.run("service.main:app", host="0.0.0.0", port=8085, reload=True)

