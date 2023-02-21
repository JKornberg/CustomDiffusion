from __future__ import annotations
import importlib
import gc
import pathlib
import sys

import gradio as gr
import PIL.Image
import numpy as np
from omegaconf import OmegaConf

import torch
from diffusers import StableDiffusionPipeline
sys.path.insert(0, './custom-diffusion')


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


    def get_obj_from_str(self, string, reload=False):
        module, cls = string.rsplit(".", 1)
        if reload:
            module_imp = importlib.import_module(module)
            importlib.reload(module_imp)
        return getattr(importlib.import_module(module, package=None), cls)
    
    def instantiate_from_config(self, config):
        if not "target" in config:
            if config == '__is_first_stage__':
                return None
            elif config == "__is_unconditional__":
                return None
            raise KeyError("Expected key `target` to instantiate.")
        return self.get_obj_from_str(config["target"])(**config.get("params", dict()))

    def load_model_from_config(self, config, ckpt, verbose=False):
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        model = self.instantiate_from_config(config.model)

        token_weights = sd["cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"]
        del sd["cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"]
        m, u = model.load_state_dict(sd, strict=False)
        model.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight.data[:token_weights.shape[0]] = token_weights
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)

        model.cuda()
        model.eval()
        return model

    def load_pipe(self, model_id: str, filename: str) -> None:
        print('start')
        weight_path = self.get_weight_path(filename)
        if weight_path == self.weight_path:
            return
        self.weight_path = weight_path
        print("loading")
        weight = torch.load(self.weight_path, map_location=self.device)
        print("loaded weight")
        #config = OmegaConf.load(f"config/finetune.yaml")
        #model = self.load_model_from_config(config, self.weight_path)
        if self.device.type == 'cpu':
            pipe = StableDiffusionPipeline.from_pretrained(model_id)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id, torch_dtype=torch.float16)
            pipe = pipe.to(self.device)
        from src import diffuser_training
        diffuser_training.load_model(pipe.text_encoder, pipe.tokenizer, pipe.unet, weight_path, compress=False)
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
        batch_size: int,
        resolution: int,
    ) -> PIL.Image.Image:
        if not torch.cuda.is_available():
            raise gr.Error('CUDA is not available.')
        print("running")
        self.load_pipe(base_model, weight_name)
        print("loaded pipe")
        generator = torch.Generator(device=self.device).manual_seed(seed)
        out = self.pipe([prompt]*batch_size,
                        num_inference_steps=n_steps,
                        guidance_scale=guidance_scale,
                        height=resolution, width=resolution,
                        eta = eta,
                        generator=generator)  # type: ignore
        torch.cuda.empty_cache()
        out = out.images
        out = PIL.Image.fromarray(np.hstack([np.array(x) for x in out]))
        return out
