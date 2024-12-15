from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from diffusers import StableDiffusionPipeline
import torch
from io import BytesIO
import base64
import os

# --- Configuration ---
MODEL_PATH = "/home/cpow/Desktop/api-endpoint/models/stable-diffusion-v1-5"
PRETRAINED_MODEL = "runwayml/stable-diffusion-v1-5"  # Hugging Face model hub path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# --- Global Pipeline ---
pipeline = None

# --- Load or Download Pipeline ---
def load_pipeline():
    global pipeline
    try:
        if os.path.exists(MODEL_PATH) and "model_index.json" in os.listdir(MODEL_PATH):
            print("Loading Stable Diffusion pipeline from local path...")
            pipeline = StableDiffusionPipeline.from_pretrained(
                MODEL_PATH, torch_dtype=DTYPE
            ).to(DEVICE)
        else:
            print("Local model not found. Downloading pre-trained model from Hugging Face...")
            pipeline = StableDiffusionPipeline.from_pretrained(
                PRETRAINED_MODEL, torch_dtype=DTYPE
            ).to(DEVICE)
            pipeline.save_pretrained(MODEL_PATH)  # Save downloaded model locally
            print(f"Model saved locally at {MODEL_PATH}")
        
        pipeline.enable_attention_slicing()  # Optimize memory usage
        if DEVICE == "cuda":
            pipeline.unet = torch.compile(pipeline.unet)  # Optimize U-Net (PyTorch 2.0+ required)
        print(f"Model loaded successfully on {DEVICE}")
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        raise RuntimeError("Failed to load or download the model pipeline.")

# --- API Models ---
class ImageGenerationRequest(BaseModel):
    prompt: str
    num_images: int = 1
    steps: int = 25
    guidance_scale: float = 7.5
    height: int = 512
    width: int = 512

    @field_validator("num_images")
    def validate_num_images(cls, value):
        if not 1 <= value <= 4:
            raise ValueError("num_images must be between 1 and 4.")
        return value

    @field_validator("height", "width")
    def validate_dimensions(cls, value):
        if value % 8 != 0 or value > 1024:
            raise ValueError("Dimensions must be multiples of 8 and less than or equal to 1024.")
        return value

class ImageGenerationResponse(BaseModel):
    images: list[str]

# --- FastAPI Application ---
app = FastAPI()

@app.on_event("startup")
def on_startup():
    try:
        load_pipeline()
    except RuntimeError as e:
        print(f"Pipeline failed to load during startup: {e}")

def generate_images(request: ImageGenerationRequest) -> list[str]:
    try:
        print(f"Generating images for prompt: '{request.prompt}'")
        output = pipeline(
            prompt=request.prompt,
            num_images_per_prompt=request.num_images,
            num_inference_steps=request.steps,
            guidance_scale=request.guidance_scale,
            height=request.height,
            width=request.width,
        )
        images = output.images
        base64_images = []
        for image in images:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            base64_images.append(img_str)
        return base64_images
    except Exception as e:
        print(f"Error during image generation: {e}")
        raise HTTPException(status_code=500, detail="Image generation failed")

@app.post("/generate", response_model=ImageGenerationResponse)
def generate_image_endpoint(request: ImageGenerationRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline is not ready")
    images = generate_images(request)
    return {"images": images}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app="api:app", host="127.0.0.1", port=8000, reload=True)
