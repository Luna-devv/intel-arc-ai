from fastapi import APIRouter, Response
from diffusers import DiffusionPipeline
import torch
import utils
from time import time
from io import BytesIO

router = APIRouter()
base_url = "https://ai.local.wamellow.com"

pipe = DiffusionPipeline.from_pretrained(
    "cagliostrolab/animagine-xl-3.1",
    torch_dtype=torch.float16,
    use_safetensors=True,
)

pipe.to('xpu')


@router.get("/generate/image/animagine-xl-v3")
def generate(
    prompt: str,
    negative_prompt: str = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, jpeg artifacts, signature, watermark, username, artist name",
    width: int = 1024,
    height: int = 1024,
    guidance_scale: int = 7,
    steps: int = 28
):
    start_time = time()*1000

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=steps
    ).images[0]

    image_bytes = BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes = image_bytes.getvalue()

    filename = utils.save_image(image_bytes, "animagine-xl-v3")

    return {
        "url": f"{base_url}/static/{filename}",
        "duration": round(time()*1000 - start_time),
    }
