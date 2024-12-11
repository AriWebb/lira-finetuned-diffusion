import torch
from model_confidence import solution_1, solution_2, pang_solution
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from typing import Optional, List
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer
from scipy.stats import norm
import gc

# def cosine(point_path: str, model_path: str, token: str):
#     point = safe_open(point_path, "pt").get_tensor(token)
#     model = safe_open(model_path, "pt").get_tensor(token)
#     return 1 - F.cosine_similarity(point, model, dim=1).item()

counter = 0
def eval_lira_1(
        pipe: StableDiffusionPipeline, z_0, prompt, sigma_steps_cap, # y = a drawing in the style of <*> 
        target_path : str,
        shadow_paths : List[str],
        token : str,
        K : float,
        db : bool = False
    ):
    global counter
    counter += 1
    print(f"Number of eval_lira_1 calls made: {counter}")
    with torch.no_grad():
        if db:
            del pipe.unet
            gc.collect()
            torch.cuda.empty_cache()
            pipe.unet = UNet2DConditionModel.from_pretrained(target_path).to("cuda")
        else:
            pipe.load_textual_inversion(target_path)
        text_input = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt").input_ids.to("cuda")
        y = pipe.text_encoder(text_input)[0]
        target_val = solution_1(pipe, z_0, y, K, sigma_steps_cap).cpu()
        if not db:
            pipe.unload_textual_inversion()
        #print(f"target {target_val}")
        shadow_vals = []
        for path in shadow_paths:
            if db:
                pipe.unet = UNet2DConditionModel.from_pretrained(path).half().to("cuda")
            else:
                pipe.load_textual_inversion(path)
            text_input = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt").input_ids.to("cuda")
            y = pipe.text_encoder(text_input)[0]
            shadow_val = solution_1(pipe, z_0, y, K, sigma_steps_cap).cpu()
            shadow_vals.append(shadow_val)
            if not db:
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
                outs : List[torch.Tensor], # list of latents
                token : str,
                granularity : int,
                K : float,
                db : float = False
            ):
    in_vals = np.array([eval_lira_1(pipe, z0, prompt, sigma_steps_cap, target_path, shadow_paths, token, K, db=db) for z0 in ins])
    out_vals = np.sort([eval_lira_1(pipe, z0, prompt, sigma_steps_cap, target_path, shadow_paths, token, K, db=db) for z0 in outs])
    thresholds = np.linspace(out_vals[0], out_vals[-1], granularity + 1)
    fprs = np.sum(out_vals[:, np.newaxis] < thresholds, axis=0) / len(out_vals)
    tprs = np.sum(in_vals[:, np.newaxis] < thresholds, axis=0) / len(in_vals)
    return fprs, tprs, in_vals, out_vals

def eval_lira_2(
        pipe: StableDiffusionPipeline, z_0, prompt, # y = a drawing in the style of <*> 
        target_path : str,
        shadow_paths : List[str],
        token : str,
        N : float,  # Using N instead of K for solution_2
        db : float = False
    ):
    global counter
    counter += 1
    print(f"Number of eval_lira_2 calls made: {counter}")
    with torch.no_grad():
        if db:
            UNet2DConditionModel.from_pretrained(target_path).half().to("cuda")
        else:
            pipe.load_textual_inversion(target_path)
        text_input = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt").input_ids.to("cuda")
        y = pipe.text_encoder(text_input)[0]
        target_val = solution_2(pipe, z_0, y, N).cpu()
        if not db:
            pipe.unload_textual_inversion()
        
        shadow_vals = []
        for path in shadow_paths:
            if db:
                pipe.unet = UNet2DConditionModel.from_pretrained(path).half().to("cuda")
            else:
                pipe.load_textual_inversion(path)
            text_input = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt").input_ids.to("cuda")
            y = pipe.text_encoder(text_input)[0]
            CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
            shadow_val = solution_2(pipe, z_0, y, N).cpu()
            shadow_vals.append(shadow_val)
            if not db:
                pipe.unload_textual_inversion()
            
        shadow_vals = np.array(shadow_vals)
        return norm.cdf(target_val, shadow_vals.mean(), shadow_vals.std())

def threshold_attack_2(
                pipe: StableDiffusionPipeline, prompt,
                target_path : str,
                shadow_paths : Optional[List[str]],
                ins : List[torch.Tensor],
                outs : List[torch.Tensor],
                token : str,
                granularity : int,
                N : float,
                db : float = False
            ):
    in_vals = np.array([eval_lira_2(pipe, z0, prompt, target_path, shadow_paths, token, N, db=db) for z0 in ins])
    out_vals = np.sort([eval_lira_2(pipe, z0, prompt, target_path, shadow_paths, token, N, db=db) for z0 in outs])
    thresholds = np.linspace(out_vals[0], out_vals[-1], granularity + 1)
    fprs = np.sum(out_vals[:, np.newaxis] < thresholds, axis=0) / len(out_vals)
    tprs = np.sum(in_vals[:, np.newaxis] < thresholds, axis=0) / len(in_vals)
    return fprs, tprs, in_vals, out_vals

def eval_pang(
        pipe: StableDiffusionPipeline, z_0, prompt, # y = a drawing in the style of <*> 
        target_path : str,
        shadow_paths : List[str],
        db : float = False
    ):
    pipe.safety_checker = None
    global counter
    counter += 1
    print(f"Number of eval_pang calls made: {counter}")
    with torch.no_grad():
        if db:
            del pipe.unet
            gc.collect()
            torch.cuda.empty_cache() 
            pipe.unet = UNet2DConditionModel.from_pretrained(target_path).half().to("cuda")
        else:
            pipe.load_textual_inversion(target_path)
        text_input = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt").input_ids.to("cuda")
        y = pipe.text_encoder(text_input)[0]
        target_val = pang_solution(pipe, z_0, prompt)
        if not db:
            pipe.unload_textual_inversion()

        shadow_vals = []
        for path in shadow_paths:
            if db:
                pipe.unet = UNet2DConditionModel.from_pretrained(path).half().to("cuda")
            else:
                pipe.load_textual_inversion(path)
            text_input = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt").input_ids.to("cuda")
            y = pipe.text_encoder(text_input)[0]
            #CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
            shadow_val = pang_solution(pipe, z_0, prompt)
            shadow_vals.append(shadow_val)
            #print(shadow_val)
            if not db:
                pipe.unload_textual_inversion()
            
        shadow_vals = np.array(shadow_vals)
        #print(shadow_vals)
        return norm.cdf(target_val, shadow_vals.mean(), shadow_vals.std())

def pang_attack(
                pipe: StableDiffusionPipeline, prompt,
                target_path : str,
                shadow_paths : Optional[List[str]],
                ins : List[torch.Tensor],
                outs : List[torch.Tensor],
                granularity : int,
                db : float = False
            ):
    #in_vals = np.array([eval_pang(pipe, z0, prompt, target_path, shadow_paths, db=db) for z0 in ins])
    #print(f"ti in_vals: {in_vals}")
    #out_vals = np.sort([eval_pang(pipe, z0, prompt, target_path, shadow_paths, db=db) for z0 in outs])
    #print(f"ti out_vals: {out_vals}")
    in_vals = np.array([
    0.96759047, 0.84303861, 0.86026406, 0.92900905, 0.95961135, 0.4871217,
    0.1817246,  0.58613382, 0.97858553, 0.8855633,  0.74422046, 0.6350456,
    0.85199599, 0.06579911, 0.02450977, 0.48686161, 0.83381088, 0.13461913,
    0.60678533, 0.69305774
])
    out_vals = np.array([
    0.05769038, 0.07048656, 0.11803179, 0.13881855, 0.25163015, 0.2730273,
    0.43803629, 0.53248056, 0.56526263, 0.63273586, 0.64053884, 0.70360048,
    0.71998736, 0.72125448, 0.77120405, 0.77342168, 0.78024825, 0.82835787,
    0.96087936
])
    thresholds = np.linspace(out_vals[0], out_vals[-1], granularity + 1)
    fprs = np.sum(out_vals[:, np.newaxis] < thresholds, axis=0) / len(out_vals)
    tprs = np.sum(in_vals[:, np.newaxis] < thresholds, axis=0) / len(in_vals)
    return fprs, tprs, in_vals, out_vals
