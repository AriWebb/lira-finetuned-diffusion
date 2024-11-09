
import torch
from model_confidence import solution_1, solution_2
from diffusers import StableDiffusionPipeline, DDIMScheduler
from typing import Optional, List
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer
from scipy.stats import norm

def cosine(point_path: str, model_path: str, token: str):
    point = safe_open(point_path, "pt").get_tensor(token)
    model = safe_open(model_path, "pt").get_tensor(token)
    return 1 - F.cosine_similarity(point, model, dim=1).item()

counter = 0
def eval_lira_1(
        pipe: StableDiffusionPipeline, z_0, prompt, sigma_steps_cap, # y = a drawing in the style of <*> 
        target_path : str,
        shadow_paths : List[str],
        token : str,
        K : float,
    ):
    global counter
    counter += 1
    print(f"Number of eval_lira_1 calls made: {counter}")
    with torch.no_grad():
        pipe.load_textual_inversion(target_path)
        text_input = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt").input_ids.to("cuda")
        y = pipe.text_encoder(text_input)[0]
        target_val = solution_1(pipe, z_0, y, K, sigma_steps_cap).cpu()
        pipe.unload_textual_inversion()
        #print(f"target {target_val}")
        shadow_vals = []
        for path in shadow_paths:
            pipe.load_textual_inversion(path)
            text_input = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt").input_ids.to("cuda")
            y = pipe.text_encoder(text_input)[0]
            CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
            shadow_val = solution_1(pipe, z_0, y, K, sigma_steps_cap).cpu()
            shadow_vals.append(shadow_val)
            pipe.unload_textual_inversion()
            #print(f"shadow {shadow_val}")
        shadow_vals = np.array(shadow_vals)
        #print(norm.cdf(target_val, shadow_vals.mean(), shadow_vals.std()))
        return norm.cdf(target_val, shadow_vals.mean(), shadow_vals.std())

def threshold_attack_1(
                pipe: StableDiffusionPipeline, prompt, sigma_steps_cap,
                target_path : str, # filepath of target model
                shadow_paths : Optional[List[str]],
                ins : List[torch.Tensor], # list of latents
                outs : List[torch.Tensor], # list of latents\
                token : str,
                granularity : int,
                K : float,
            ):
    in_vals = np.array([eval_lira_1(pipe, z0, prompt, sigma_steps_cap, target_path, shadow_paths, token, K) for z0 in ins])
    out_vals = np.sort([eval_lira_1(pipe, z0, prompt, sigma_steps_cap, target_path, shadow_paths, token, K) for z0 in outs])
    thresholds = np.linspace(out_vals[0], out_vals[-1], granularity + 1)
    fprs = np.sum(out_vals[:, np.newaxis] < thresholds, axis=0) / len(out_vals)
    tprs = np.sum(in_vals[:, np.newaxis] < thresholds, axis=0) / len(in_vals)
    return fprs, tprs, in_vals, out_vals