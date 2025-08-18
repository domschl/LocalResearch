from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import json
import logging

from .config import settings

app = FastAPI()
logger = logging.getLogger("uvicorn")

# Mount static files last, after all API routes are defined.
# Serve files from the 'static' directory within the 'app' module.
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def read_root():
    # Redirect to the main visualization page or provide a simple welcome
    return FileResponse(static_dir / "visualization.html")

@app.get("/api/models")
async def get_models():
    """
    Scans the data_path to find available model subdirectories.
    Returns a list of model names.
    """
    models = []
    if settings.resources_data_path.exists() and settings.resources_data_path.is_dir():
        for item in settings.resources_data_path.iterdir():
            if item.is_dir():
                # Check if essential files exist for a model to be "valid"
                vis_data_path = item / "visualization_data.json"
                # lib_path = item / "vector_library.json" # For future search
                # tensor_path = item / "embeddings.pt"   # For future search
                if vis_data_path.exists(): # and lib_path.exists() and tensor_path.exists():
                    models.append(item.name)
        logger.info(f"Found models: {models} in {settings.resources_data_path}")
    else:
        logger.error(f"Data path {settings.resources_data_path} does not exist or is not a directory.")
        raise HTTPException(status_code=500, detail=f"Data path {settings.resources_data_path} is invalid.")
    if not models:
        logger.warning(f"No models found in {settings.resources_data_path}")
    return {"models": models}


@app.get("/api/visualization_data/{model_name}")
async def get_visualization_data(model_name: str):
    """
    Serves the visualization_data.json for the specified model.
    """
    model_data_path = settings.resources_data_path / model_name / "visualization_data.json"
    if not model_data_path.exists():
        logger.error(f"Visualization data not found for model '{model_name}' at {model_data_path}")
        raise HTTPException(status_code=404, detail=f"Visualization data not found for model '{model_name}'")
    try:
        with open(model_data_path, "r") as f:
            data = json.load(f)
        logger.info(f"Successfully read visualization data for model '{model_name}', {len(data)} items found.")
        return data
    except Exception as e:
        logger.error(f"Error reading visualization data for model '{model_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Error reading visualization data for model '{model_name}'")

# If visualization.html is the main entry point, you might want a route for it
@app.get("/visualization.html", include_in_schema=False)
async def visualization_html():
    return FileResponse(static_dir / "visualization.html")

if __name__ == "__main__":
    # This is for debugging purposes if you run main.py directly.
    # For production, use run_server.py or uvicorn command.
    import uvicorn
    logger.info(f"Starting Uvicorn server from main.py on {settings.server_host}:{settings.server_port}")
    logger.info(f"Expecting data in: {settings.resources_data_path}")
    uvicorn.run(app, host=settings.server_host, port=settings.server_port)
