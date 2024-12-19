import base64
import gc
import io

import torch
import uvicorn
from diffusers import DiffusionPipeline
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Load the model once when the server starts
try:
    pipe = DiffusionPipeline.from_pretrained(
        "path_to_sd3.5-medium",
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
except Exception as e:
    print(f"Error loading model: {e}")
    pipe = None


class GenerateRequest(BaseModel):
    prompt: str
    num_inference_steps: int = 50
    guidance_scale: float = 7.5


class GenerateResponse(BaseModel):
    image_base64: str


@app.post("/generate", response_model=GenerateResponse)
def generate_image(request: GenerateRequest):
    if pipe is None:
        raise HTTPException(status_code=500, detail="Model is not loaded properly.")
    # Generate image
    image = pipe(
        request.prompt,
    ).images[0]

    # Convert image to base64 string
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode()

    # Memory management
    gc.collect()
    torch.cuda.empty_cache()

    return GenerateResponse(image_base64=image_base64)


if __name__ == "__main__":
    uvicorn.run("script_name:app", host="0.0.0.0", port=8000, reload=True)
