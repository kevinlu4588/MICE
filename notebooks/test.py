from diffusers import StableDiffusionPipeline
import torch

pipeline = StableDiffusionPipeline.from_pretrained("/share/u/kevin/erasing/models/30_inpainting_erasure_esd_airliner", torch_dtype=torch.float16).to("cuda")
pipeline.load_textual_inversion("/share/u/kevin/erasing/diffusers/examples/textual_inversion/textual_inversion_airliner")
image = pipeline("a picture of an <airliner>", num_inference_steps=50).images[0]
image.save("airliner.png")