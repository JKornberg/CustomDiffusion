from __future__ import annotations

import gc
import pathlib
import sys

import gradio as gr
import PIL.Image
import torch
from diffusers import StableDiffusionPipeline

sys.path.insert(0, 'custom-diffusion')
from src import sample_diffuser, diffuser_training


class InferencePipeline:
    def __init__(self):
        self.pipe = None
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.weight_path = None

    def clear(self) -> None:
        self.weight_path = None
        del self.pipe
        self.pipe = None
        torch.cuda.empty_cache()
        gc.collect()

    @staticmethod
    def get_weight_path(name: str) -> pathlib.Path:
        curr_dir = pathlib.Path(__file__).parent
        return curr_dir / name

    def load_pipe(self, model_id: str, filename: str) -> None:
        weight_path = self.get_weight_path(filename)
        if weight_path == self.weight_path:
            return
        self.weight_path = weight_path
        weight = torch.load(self.weight_path, map_location=self.device)

        if self.device.type == 'cpu':
            pipe = StableDiffusionPipeline.from_pretrained(model_id)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id, torch_dtype=torch.float16)
            pipe = pipe.to(self.device)

        diffuser_training.load_model(pipe.text_encoder, pipe.tokenizer, pipe.unet, weight_path, '<new1>')

        self.pipe = pipe

    def run(
        self,
        base_model: str,
        weight_name: str,
        prompt: str,
        seed: int,
        n_steps: int,
        guidance_scale: float,
        eta: float,
    ) -> PIL.Image.Image:
        if not torch.cuda.is_available():
            raise gr.Error('CUDA is not available.')

        self.load_pipe(base_model, weight_name)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        out = self.pipe(prompt,
                        num_inference_steps=n_steps,
                        guidance_scale=guidance_scale,
                        eta = eta,
                        generator=generator)  # type: ignore
        return out.images[0]
