from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import api.heatmap_service as heatmap_service
import uvicorn

app = FastAPI(title="Snap4City Heatmap API")


app.include_router(heatmap_service.router, prefix="/heatmap", tags=["heatmap"])

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

    uvicorn.run("main:app", host="0.0.0.0", port=8085, reload=True)

