"""FastAPI application entrypoint.

We should start the FastAPI application from this module running the script

    scripts/start_image_api.sh

Then, we can send a POST request to the /predict_image/ endpoint with a base64-encoded image.
An example of how to send a request to the API is shown below:

    API_URL = "http://localhost:8000"
    image_path = "path/to/image.jpg"

    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    payload = { "image": base64_image}

    response = requests.post(f"{API_URL}/predict_image/", json=payload)
    results = response.json()

    for pipeline_name, result in response.json():
        print(f"{pipeline_name}: {result}")

Another example is provided in

    notebooks/try_api.ipynb

Author: Mikel Sagardia
Date: 2024-12-04
"""
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn

from src.core import SRC_PATH
from src.core import APPLICATION_NAME, VERSION
from src.service import image_inference_service
from src.domain.shared import image_utils


# FastAPI application
app = FastAPI()

# Load environment variables
load_dotenv(override=True, dotenv_path=str(SRC_PATH / ".env"))

# Initialize the image inference service
image_service = image_inference_service


class PredictImageRequest(BaseModel):
    """Pydantic model for the predict_image endpoint request."""
    image: str  # Base64 encoded image string


@app.get("/", summary="Root endpoint", description="Returns application metadata.")
def root() -> dict:
    """Root endpoint."""
    return {
        "application": APPLICATION_NAME,
        "version": VERSION
    }


@app.post(
    "/predict_image/",
    summary="Predict image properties",
    description="Process a base64-encoded image using all available pipelines."
)
def predict_image(request: PredictImageRequest) -> dict:
    """Predict image properties.

    Args:
        request: Contains a base64-encoded image.

    Returns:
        A dictionary with predictions from all pipelines.
    """
    try:
        image_pil = image_utils.base64_to_image(request.image)
        result = image_service.predict(image_pil)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error") from e


def start():
    """Start the FastAPI application."""
    uvicorn.run(
        "src.entrypoints.api:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=True
    )
