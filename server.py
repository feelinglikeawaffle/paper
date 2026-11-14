import io
import os
import threading
from pathlib import Path

import torch
from PIL import Image, ImageOps
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
MODEL_ID = os.getenv("FLUX_MODEL_ID", "black-forest-labs/FLUX.2-klein-4B")
MAX_UPLOAD = 12 * 1024 * 1024
ALLOWED_TYPES = {"image/png", "image/jpeg", "image/webp"}

app = FastAPI(title="SHREDCO FLUX Image Engine")
pipe = None
load_lock = threading.Lock()
generation_lock = threading.Lock()


def get_pipeline():
    global pipe
    if pipe is not None:
        return pipe
    with load_lock:
        if pipe is not None:
            return pipe
        from diffusers import Flux2KleinPipeline
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dtype = torch.float32
        else:
            dtype = torch.float32
        pipe = Flux2KleinPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
        if torch.cuda.is_available():
            pipe.enable_model_cpu_offload()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            pipe.to("mps")
        else:
            pipe.to("cpu")
        return pipe


@app.get("/")
def home():
    return FileResponse(ROOT / "studio.html")


@app.get("/api/status")
def status():
    return {"ready": pipe is not None, "model": MODEL_ID,
            "cuda": torch.cuda.is_available(),
            "device": "cuda" if torch.cuda.is_available() else
                      ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")}


@app.post("/api/generate")
async def generate(prompt: str = Form(...), size: int = Form(1024),
                   seed: str = Form(""), image: UploadFile | None = File(None)):
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(400, "Please enter an image prompt.")
    if len(prompt) > 1200:
        raise HTTPException(400, "Prompt is too long (maximum 1200 characters).")
    if size not in (512, 768, 1024):
        raise HTTPException(400, "Unsupported output size.")
    reference = None
    if image is not None:
        if image.content_type not in ALLOWED_TYPES:
            raise HTTPException(415, "Use a PNG, JPG, or WEBP image.")
        raw = await image.read(MAX_UPLOAD + 1)
        if len(raw) > MAX_UPLOAD:
            raise HTTPException(413, "Image is larger than 12 MB.")
        try:
            reference = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
        except Exception:
            raise HTTPException(400, "The uploaded image could not be opened.")
        if reference.width * reference.height > 25_000_000:
            raise HTTPException(413, "Image dimensions are too large.")

    try:
        with generation_lock:
            model = get_pipeline()
            generator = None
            if seed.strip():
                try:
                    seed_value = int(seed)
                    if seed_value < 0 or seed_value > 2**63 - 1:
                        raise ValueError()
                except ValueError:
                    raise HTTPException(400, "Seed must be a non-negative integer.")
                generator_device = "cuda" if torch.cuda.is_available() else "cpu"
                generator = torch.Generator(device=generator_device).manual_seed(seed_value)

            kwargs = {"prompt": prompt, "generator": generator}
            if reference is not None:
                kwargs["image"] = reference
                kwargs["width"] = size
                kwargs["height"] = size
            else:
                kwargs.update({"width": size, "height": size,
                               "guidance_scale": 1.0, "num_inference_steps": 4})
            result = model(**kwargs).images[0]
            buffer = io.BytesIO()
            result.save(buffer, format="PNG")
            return Response(content=buffer.getvalue(), media_type="image/png",
                            headers={"Content-Disposition": 'inline; filename="shredco-flux.png"',
                                     "Cache-Control": "no-store"})
    except HTTPException:
        raise
    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(503, "Not enough GPU memory. Try a smaller image size or close other GPU-heavy apps.")
    except Exception as exc:
        raise HTTPException(500, f"FLUX error: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=7860, reload=False)
